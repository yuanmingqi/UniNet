import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def shiftbits_torch(fa, noshifts):
    fanew = fa.clone()
    width = fa.shape[1]
    s = 2 * torch.abs(noshifts)
    p = width - s

    # Shift
    if noshifts == 0:
        return fa

    elif noshifts < 0:
        fanew[:, 0:p] = fa[:, s:p + s]
        fanew[:, p:width] = fa[:, 0:s]

    else:
        fanew[:, s:width] = fa[:, 0:p]
        fanew[:, 0:s] = fa[:, p:width]

    return fanew

def shiftbits(fa, noshifts):
    fanew = np.zeros(fa.shape)
    width = fa.shape[1]
    s = 2 * np.abs(noshifts)
    p = width - s

    # Shift
    if noshifts == 0:
        fanew = fa

    elif noshifts < 0:
        fanew[:, 0:p] = fa[:, s:p + s]
        fanew[:, p:width] = fa[:, 0:s]

    else:
        fanew[:, s:width] = fa[:, 0:p]
        fanew[:, 0:s] = fa[:, p:width]

    return fanew

def calHammingDist(template1, mask1, template2, mask2):
    # Initialize
    hd = np.nan

    # Shift template left and right, use the lowest Hamming distance
    for shifts in range(-8, 9):
        template1s = shiftbits(template1, shifts)
        mask1s = shiftbits(mask1, shifts)

        mask = np.logical_or(mask1s, mask2)
        nummaskbits = np.sum(mask == 1)
        totalbits = template1s.size - nummaskbits

        C = np.logical_xor(template1s, template2)
        C = np.logical_and(C, mask)
        bitsdiff = np.sum(C == 1)

        if totalbits == 0:
            hd = np.nan
        else:
            hd1 = bitsdiff / totalbits
            if hd1 < hd or np.isnan(hd):
                hd = hd1

    return hd

def binarization(feature, mask, threshold=0.6):
    f_mean = np.mean(feature)
    feature[np.where(feature > f_mean)] = 1.
    feature[np.where(feature < f_mean)] = 0.

    mask[np.where(np.abs(feature - f_mean) < threshold)] = 1.

    return feature, mask


