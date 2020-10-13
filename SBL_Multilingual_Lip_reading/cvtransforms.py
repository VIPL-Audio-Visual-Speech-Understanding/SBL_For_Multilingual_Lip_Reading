# coding: utf-8
import random
import cv2
import numpy as np


def CenterCrop(batch_img, size):
    
    w, h = batch_img[0].shape[1], batch_img[0].shape[0]
    th, tw = size
    img = np.zeros((len(batch_img), th, tw))
    for i in range(len(batch_img)):
        x1 = int(round((w - tw))/2.)
        y1 = int(round((h - th))/2.)
        #print(batch_img.shape)
        #r, g, b = batch_img[:, :, :, :, 0], batch_img[:, :, :, :, 1], batch_img[:, :, :, :, 2]
        #batch_img_gray = 0.299*r + 0.587*g + 0.114*b
        img[i] = batch_img[i,  y1:y1+th, x1:x1+tw]
    return img


def RandomCrop(batch_img, size):
    #print(batch_img.shape)
    w, h = batch_img[0].shape[1], batch_img[0].shape[0]
    th, tw = size
    img = np.zeros((len(batch_img), th, tw))
    for i in range(len(batch_img)):
        x1 = random.randint(0, 8)
        y1 = random.randint(0, 8)
        #r, g, b = batch_img[:, :, :, :, 0], batch_img[:, :, :, :, 1], batch_img[:, :, :, :, 2]
        #batch_img_gray = 0.299*r + 0.587*g + 0.114*b
        img[i] = batch_img[i, y1:y1+th, x1:x1+tw]
    return img


def HorizontalFlip(batch_img):
    for i in range(len(batch_img)):
        if random.random() > 0.5:
            for j in range(len(batch_img[i])):
                batch_img[i][j] = cv2.flip(batch_img[i][j], 1)
    return batch_img


def ColorNormalize(batch_img):
    mean = 0.413621
    std = 0.1700239
    batch_img = (batch_img - mean) / std
    return batch_img
