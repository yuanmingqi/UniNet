import torch
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image

import pandas as pd
import numpy as np
import time

class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

class DataGenerator():
    def __init__(
            self,
            txt,
            batch_size, 
            people_per_batch,
            imgs_per_person,
            alpha=0.2
    ):
        self.img_list = pd.read_csv(txt, sep=' ', header=None, names=['path', 'label'])
        self.img_list = self.img_list[self.img_list['label'] < 50]
        self.classes = self.img_list['label'].value_counts().shape[0]
        self.batch_size = batch_size
        self.batches = None
        self.people_per_batch = people_per_batch
        self.images_per_person = imgs_per_person
        self.alpha = alpha

        self.dataset = self.get_dataset()
        self.triplets = None

    def get_dataset(self):
        dataset = []
        norf_classes = self.classes

        for i in range(norf_classes):
            class_name = 'UE' + str(i)
            img_paths = self.img_list[self.img_list['label'] == i]['path'].tolist()
            dataset.append(ImageClass(class_name, img_paths))

        return dataset

    def sample_people(self, dataset, people_per_batch, images_per_person):
        nrof_images = people_per_batch * images_per_person

        # Sample classes from the dataset
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)

        i = 0
        image_paths = []
        num_per_class = []
        sampled_class_indices = []
        # Sample images from these classes until we have enough
        while len(image_paths) < nrof_images:
            class_index = class_indices[i]
            nrof_images_in_class = len(dataset[class_index])
            image_indices = np.arange(nrof_images_in_class)
            np.random.shuffle(image_indices)
            nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
            idx = image_indices[0:nrof_images_from_class]
            image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
            sampled_class_indices += [class_index] * nrof_images_from_class
            image_paths += image_paths_for_class
            num_per_class.append(nrof_images_from_class)
            i += 1

        return image_paths, num_per_class

    def select_triplets(self, embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha=0.2):
        """ Select the triplets for training
        """
        trip_idx = 0
        emb_start_idx = 0
        num_trips = 0
        triplets = []

        for i in range(people_per_batch):
            nrof_images = int(nrof_images_per_class[i])
            for j in range(1, nrof_images):
                a_idx = emb_start_idx + j - 1
                neg_dists_sqr = torch.sum(torch.square(embeddings[a_idx] - embeddings), dim=[1, 2])
                for pair in range(j, nrof_images):  # For every possible positive pair.
                    p_idx = emb_start_idx + pair
                    pos_dist_sqr = torch.sum(torch.square(embeddings[a_idx] - embeddings[p_idx]))
                    neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
                    all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]
                    nrof_random_negs = all_neg.shape[0]
                    if nrof_random_negs > 0:
                        rnd_idx = np.random.randint(nrof_random_negs)
                        n_idx = all_neg[rnd_idx]
                        triplets.append([image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]])
                        trip_idx += 1

                    num_trips += 1

            emb_start_idx += nrof_images

        np.random.shuffle(triplets)

        return triplets, num_trips, len(triplets)

    def reset(self, model, device):
        t_s = time.perf_counter()

        image_paths, num_per_class = self.sample_people(self.dataset, self.people_per_batch, self.images_per_person)
        nrof_examples = self.people_per_batch * self.images_per_person
        nrof_batches = int(np.ceil(nrof_examples / self.batch_size))

        embeddings = torch.zeros(nrof_batches * self.batch_size, 64, 512)
        for i in range(nrof_batches):
            imgs = torch.zeros(size=(self.batch_size, 1, 64, 512))
            for j in range(self.batch_size):
                img = Image.open(image_paths[i * self.batch_size + j])
                img = transforms.ToTensor()(img)
                imgs[j, 0, :, :] = img

            imgs = imgs.to(device)
            feat = model(imgs)
            embeddings[i*self.batch_size:(i+1)*self.batch_size, ...] = feat[:, 0, :, :]

        embeddings = F.normalize(embeddings)

        self.triplets, nrof_random_negs, nrof_triplets = self.select_triplets(
            embeddings, num_per_class, image_paths, self.people_per_batch, self.alpha)
        self.batches = int(np.floor(nrof_triplets / self.batch_size))

        t_e = time.perf_counter()
        print('INFO: Batch data generated, time cost={}s'.format(t_e - t_s))

    def gen(self):
        for i in range(self.batches):
            pos_sample = torch.zeros(self.batch_size, 1, 64, 512)
            anc_sample = torch.zeros(self.batch_size, 1, 64, 512)
            neg_sample = torch.zeros(self.batch_size, 1, 64, 512)
            pos_mask_sample = torch.zeros(self.batch_size, 1, 64, 512)
            anc_mask_sample = torch.zeros(self.batch_size, 1, 64, 512)
            neg_mask_sample = torch.zeros(self.batch_size, 1, 64, 512)

            for j in range(self.batch_size):
                ps_img_file = self.triplets[i*self.batch_size+j][0]
                as_img_file = self.triplets[i*self.batch_size+j][1]
                ns_img_file = self.triplets[i*self.batch_size+j][2]

                pos_sample[j, ...] = transforms.ToTensor()(Image.open(ps_img_file))
                anc_sample[j, ...] = transforms.ToTensor()(Image.open(as_img_file))
                neg_sample[j, ...] = transforms.ToTensor()(Image.open(ns_img_file))

                pos_mask_sample[j, ...] = transforms.ToTensor()(
                    Image.open(ps_img_file.replace('.bmp', '_mask.png'))) * 255.0
                anc_mask_sample[j, ...] = transforms.ToTensor()(
                    Image.open(as_img_file.replace('.bmp', '_mask.png'))) * 255.0
                neg_mask_sample[j, ...] = transforms.ToTensor()(
                    Image.open(ns_img_file.replace('.bmp', '_mask.png'))) * 255.0

            triplet_data = {
                'ps': pos_sample,
                'ps_mask': torch.ceil(pos_mask_sample[:,0,:,:]),
                'as': anc_sample,
                'as_mask': torch.ceil(anc_mask_sample[:,0,:,:]),
                'ns': neg_sample,
                'ns_mask': torch.ceil(neg_mask_sample[:,0,:,:])
            }

            yield triplet_data
