from match import *
from network import FeatNet
from PIL import Image
from torchvision import transforms

import cv2
import torch
import pandas as pd

test_txt = './dataset/Train/NDeval.txt' # './dataset/Test/NDtest.txt'
img_list = pd.read_csv(test_txt, sep=' ', header=None, names=['path', 'label'])
img_list = img_list[img_list['label'] < 50]

ava_idx = img_list['label'].value_counts().index
classes = ava_idx.shape[0] # img_list['label'].value_counts().shape[0]

sample_num = 5000
binary_threshold = 0.6
triplet_pairs = []
people = np.random.randint(0, classes, size=(sample_num, 2))
model_path = './snapshots/featnet.pth'
feanet = torch.load(model_path)

for i in range(sample_num):
    pspa_img_file = img_list[img_list['label'] == ava_idx[people[i][0]]].sample(n=2)['path'].tolist()
    pn_img_file = img_list[img_list['label'] == ava_idx[people[i][1]]].sample(n=1)['path'].tolist()[0]

    triplet_pairs.append([pspa_img_file[0], pspa_img_file[1], pn_img_file])

pos_pair_result = []
neg_pair_result = []

for i in range(sample_num):
    img_pos = transforms.ToTensor()(Image.open(triplet_pairs[i][0])).unsqueeze(0)
    img_anc = transforms.ToTensor()(Image.open(triplet_pairs[i][1])).unsqueeze(0)
    img_neg = transforms.ToTensor()(Image.open(triplet_pairs[i][2])).unsqueeze(0)

    fp_mask = cv2.imread(triplet_pairs[i][0].replace('.bmp', '_mask.png'), 0)
    fa_mask = cv2.imread(triplet_pairs[i][1].replace('.bmp', '_mask.png'), 0)
    fn_mask = cv2.imread(triplet_pairs[i][2].replace('.bmp', '_mask.png'), 0)

    fp = feanet(img_pos).cpu().detach().numpy()[0, 0, :, :]
    fa = feanet(img_anc).cpu().detach().numpy()[0, 0, :, :]
    fn = feanet(img_neg).cpu().detach().numpy()[0, 0, :, :]

    binary_fp, binary_fp_mask = binarization(fp, fp_mask, binary_threshold)
    binary_fa, binary_fa_mask = binarization(fa, fa_mask, binary_threshold)
    binary_fn, binary_fn_mask = binarization(fn, fn_mask, binary_threshold)

    hd_fpfa = calHammingDist(binary_fp, binary_fp_mask, binary_fa, binary_fa_mask)
    hd_fafn = calHammingDist(binary_fa, binary_fa_mask, binary_fn, binary_fn_mask)

    print('Sample={}, P-A={}, A-N={}'.format(
        i + 1, hd_fpfa, hd_fafn))

    pos_pair_result.append(list())
    neg_pair_result.append(list())

    for hd_threshold in np.linspace(1.0, 1.5, 50):
        if hd_fpfa < hd_threshold:
            fp_fa = True
        else:
            fp_fa = False

        if hd_fafn < hd_threshold:
            fa_fn = True
        else:
            fa_fn = False

        pos_pair_result[i].append(fp_fa)
        neg_pair_result[i].append(fa_fn)

TAR = np.sum(pos_pair_result, axis=0) / sample_num
FAR = np.sum(neg_pair_result, axis=0) / sample_num

print(TAR)
print(FAR)

np.savez('static/roc.npz', tar=TAR, far=FAR)







