import torch
from torch.autograd import Function
import numpy as np

''' Extended Triplet Loss '''
class ETLoss(Function):
    def __init__(self, alpha, device):
        super(ETLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.device = device

    def shiftbits(self, fa, noshifts):
        fnew = fa.clone()
        width = fa.shape[2]
        s = 2 * np.abs(noshifts)
        p = width - s

        # Shift
        if noshifts == 0:
            return fa

        elif noshifts < 0:
            fnew[:, :, 0:p] = fa[:, :, s:p + s]
            fnew[:, :, p:width] = fa[:, :, 0:s]

        else:
            fnew[:, :, s:width] = fa[:, :, 0:p]
            fnew[:, :, 0:s] = fa[:, :, p:width]

        return fnew

    ''' Fractional Distance '''
    def fd(self, f1, f2, mask1, mask2):
        batch_size = f1.shape[0]
        batch_fd = torch.zeros(size=(batch_size, ))
        zero = torch.tensor(0.).to(self.device)

        for i in range(batch_size):
            M = torch.sum((mask1[i] == mask2[i]) & (mask1[i] == 1))
            fd = torch.where(
                ((mask1[i] == mask2[i]) & (mask1[i] == 1)),
                torch.square(f1[i] - f2[i]),
                zero)

            fd = torch.sum(fd) / M
            batch_fd[i] = fd

        return batch_fd

    ''' Minimum Shifted and Masked Distance '''
    def mmsd(self, f1, f2, mask1, mask2):
        batch_size = f1.shape[0]
        fd_set = torch.zeros(size=(17, batch_size))

        for shifts in range(-8, 9):
            f1_s = self.shiftbits(f1, shifts)
            mask1_s = self.shiftbits(mask1, shifts)

            fd_set[shifts + 8] = self.fd(f1_s, f2, mask1_s, mask2)

        batch_min_fd = torch.min(fd_set, dim=0)

        return batch_min_fd.values, batch_min_fd.indices - 8

    def forward(self, fp, fa, fn, fp_mask, fa_mask, fn_mask):
        mmsd_fa_fp, offset_ap = self.mmsd(fa[:,0,:,:], fp[:,0,:,:], fa_mask, fp_mask)
        mmsd_fa_fn, offset_an = self.mmsd(fa[:,0,:,:], fn[:,0,:,:], fa_mask, fn_mask)

        etl_loss = mmsd_fa_fp - mmsd_fa_fn + self.alpha

        zero = torch.tensor(0.)
        etl_loss = torch.maximum(etl_loss, zero)
        etl_loss = torch.mean(etl_loss)

        return etl_loss, offset_ap, offset_an

