import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

import numpy as np

from network import *
from loss import *
from data import DataGenerator
import match

class UniNet:
    def __init__(
            self,
            input_w=512,
            input_h=64,
            train_txt='./dataset/Train/NDtrain.txt',
            evaL_txt='./dataset/Train/NDeval.txt',
            snapshots='./snapshots/',
            alpha=0.2,
            epoch=1000,
            batch_size=90,
            people_per_batch=45,
            imgs_per_person=40,
            lr=1e-3,
            lr_decay=1e-6
                 ):
        self.input_w = input_w
        self.input_h = input_h
        self.train_txt = train_txt
        self.eval_txt = evaL_txt
        self.snapshots = snapshots

        self.alpha = alpha
        self.epoch = epoch
        self.batch_size = batch_size
        self.people_per_batch = people_per_batch
        self.imgs_per_person = imgs_per_person
        self.lr = lr
        self.lr_decay = lr_decay

    ''' calculate the gradient: ETL/fp, ETL/fa, ETL/fn '''
    def get_grad(self, loss, fp, fa, fn, fp_mask, fa_mask, fn_mask, b_AP, b_AN):
        if loss == 0.:
            return torch.zeros_like(fp), torch.zeros_like(fa), torch.zeros_like(fn)

        batch_size = fp.shape[0]
        zero = torch.tensor(0.).to(self.device)

        fp_mask_offset = torch.zeros_like(fp_mask)
        fn_mask_offset = torch.zeros_like(fn_mask)

        fa_b_AP = torch.zeros_like(fa)
        fa_b_AN = torch.zeros_like(fa)
        ''' shifted fp, fp with -b_AP and -b_AN '''
        fp_AP_b = torch.zeros_like(fp)
        fn_AN_b = torch.zeros_like(fn)

        for i in range(batch_size):
            fp_mask_offset[i, :, :] = match.shiftbits_torch(fp_mask[i], b_AP[i])
            fa_b_AP[i, 0, :, :] = match.shiftbits_torch(fa[i, 0, :, :], b_AP[i])
            fp_AP_b[i, 0, :, :] = match.shiftbits_torch(fp[i, 0, :, :], -b_AP[i])

            fn_mask_offset[i, :, :] = match.shiftbits_torch(fn_mask[i], b_AN[i])
            fa_b_AN[i, 0, :, :] = match.shiftbits_torch(fa[i, 0, :, :], b_AN[i])
            fn_AN_b[i, 0, :, :] = match.shiftbits_torch(fn[i, 0, :, :], -b_AN[i])

        M_ap = (fa_mask == fp_mask_offset) & (fp_mask_offset == 1.)
        M_an = (fa_mask == fn_mask_offset) & (fn_mask_offset == 1.)
        norm_M_ap = torch.sum(M_ap, dim=[1, 2])
        norm_M_an = torch.sum(M_an, dim=[1, 2])
        norm_M_ap = norm_M_ap.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        norm_M_an = norm_M_an.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
        grad_etl2fp = 2. * (fa_b_AP - fp) / (batch_size * norm_M_ap)
        grad_etl2fn = 2. * (fa_b_AN - fn) / (batch_size * norm_M_an)
        grad_etl2fp_AP_b = - 2. * (fa_b_AP - fp_AP_b) / (batch_size * norm_M_ap)
        grad_etl2fn_AN_b = 2. * (fa_b_AN - fn_AN_b) / (batch_size * norm_M_an)

        grad_etl2fp_ = grad_etl2fp.clone()
        grad_etl2fn_ = grad_etl2fn.clone()
        grad_etl2fp_AP_b_ = grad_etl2fp_AP_b.clone()
        grad_etl2fn_AN_b_ = grad_etl2fn_AN_b.clone()

        for i in range(batch_size):
            grad_etl2fp_[i] = torch.where(M_ap[i] == True, grad_etl2fp[i], zero)
            grad_etl2fp_AP_b_[i] = torch.where(M_ap[i] == True, grad_etl2fp_AP_b[i], zero)
            grad_etl2fn_[i] = torch.where(M_an[i] == True, grad_etl2fn[i], zero)
            grad_etl2fn_AN_b_[i] = torch.where(M_an[i] == True, grad_etl2fn_AN_b[i], zero)

        grad_etl2fa_ = grad_etl2fp_AP_b_ + grad_etl2fn_AN_b_

        return grad_etl2fp_, grad_etl2fa_, grad_etl2fn_

    def train(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.featnet = FeatNet()
        self.featnet = torch.nn.DataParallel(self.featnet)
        self.featnet.to(self.device)
        self.etl_loss = ETLoss(self.alpha, self.device)

        ''' set data generator '''
        self.train_dataGenerator = DataGenerator(
            txt=self.train_txt,
            batch_size=self.batch_size,
            people_per_batch=self.people_per_batch,
            imgs_per_person=self.imgs_per_person
        )

        ''' set optimizer '''
        self.optimizer = torch.optim.Adam(self.featnet.parameters(), self.lr)

        epoch_loss_list = []

        for epoch in range(self.epoch):
            print('INFO: Epoch={}'.format(epoch+1))
            self.train_dataGenerator.reset(self.featnet, self.device)
            train_g = self.train_dataGenerator.gen()

            epoch_loss = 0.0

            for step in range(self.train_dataGenerator.batches):
                self.optimizer.zero_grad()

                triplet_batch = next(train_g)
                img_ps = triplet_batch['ps'].to(self.device)
                img_ps_mask = triplet_batch['ps_mask'].to(self.device)
                img_as = triplet_batch['as'].to(self.device)
                img_as_mask = triplet_batch['as_mask'].to(self.device)
                img_ns = triplet_batch['ns'].to(self.device)
                img_ns_mask = triplet_batch['ns_mask'].to(self.device)

                fp, fa, fn = self.featnet(img_ps), self.featnet(img_as), self.featnet(img_ns)

                etl_loss, b_AP, b_AN = self.etl_loss.forward(
                    fp, fa, fn, img_ps_mask, img_as_mask, img_ns_mask)
                epoch_loss += etl_loss.item()

                fp.retain_grad()
                fa.retain_grad()
                fn.retain_grad()

                etl_loss.backward()

                if (step + 1) % 10 == 0:
                    print('INFO: Steps={}, ETL loss={}'.format(step+1, etl_loss.item()))
                    torch.save(
                        self.featnet,
                        self.snapshots + 'featnet_epoch{}_steps{}.pth'.format(epoch+1, step+1))

                grad_etl2fp, grad_etl2fa, grad_etl2fn = self.get_grad(
                    etl_loss,
                    fp, fa, fn,
                    img_ps_mask, img_as_mask, img_ns_mask,
                    b_AP, b_AN
                )

                ''' replace gradients '''
                fp.grad.data = grad_etl2fp.data
                fa.grad.data = grad_etl2fa.data
                fn.grad.data = grad_etl2fn.data

                self.optimizer.step()

            epoch_loss_list.append(epoch_loss)
            np.save('static/epoch_loss.npy', epoch_loss_list)
            print('INFO: Epoch {} done, total loss={}'.format(epoch+1, epoch_loss))





