#load one SEC data(single)

import os
import math
import random
import numpy as np
from skimage import io, exposure
from torch.utils import data
from skimage.transform import rescale
from torchvision.transforms import functional as F
from PIL import Image
num_classes = 1
MEAN = np.array([123.675, 116.28, 103.53])
STD = np.array([58.395, 57.12, 57.375])

root = '/path/to/pretraining/dataset' # data path


# def showIMG(img):
#     plt.imshow(img)
#     plt.show()
#     return 0

def normalize_image(im):
    # im = (im - MEAN) / STD
    im = im / 255
    return im.astype(np.float32)


def normalize_images(imgs):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im)
    return imgs
def del_dim(label):
    if label.ndim == 3:
        label = label[:,:,0]
    return label

def Color2Index(ColorLabel):
    IndexMap = ColorLabel.clip(max=1)
    return IndexMap


def Index2Color(pred):
    pred = exposure.rescale_intensity(pred, out_range=np.uint8)
    return pred


def sliding_crop_CD(imgs1, imgs2, labels, size):
    crop_imgs1 = []
    crop_imgs2 = []
    crop_labels = []
    label_dims = len(labels[0].shape)
    for img1, img2, label in zip(imgs1, imgs2, labels):
        h = img1.shape[0]
        w = img1.shape[1]
        c_h = size[0]
        c_w = size[1]
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            crop_imgs1.append(img1)
            crop_imgs2.append(img2)
            crop_labels.append(label)
            continue
        h_rate = h / c_h
        w_rate = w / c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        if h_times == 1:
            stride_h = 0
        else:
            stride_h = math.ceil(c_h * (h_times - h_rate) / (h_times - 1))
        if w_times == 1:
            stride_w = 0
        else:
            stride_w = math.ceil(c_w * (w_times - w_rate) / (w_times - 1))
        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j * c_h - j * stride_h)
                if (j == (h_times - 1)): s_h = h - c_h
                e_h = s_h + c_h
                s_w = int(i * c_w - i * stride_w)
                if (i == (w_times - 1)): s_w = w - c_w
                e_w = s_w + c_w
                # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
                # print('%d %d %d %d'%(s_h_s, e_h_s, s_w_s, e_w_s))
                crop_imgs1.append(img1[s_h:e_h, s_w:e_w, :])
                crop_imgs2.append(img2[s_h:e_h, s_w:e_w, :])
                if label_dims == 2:
                    crop_labels.append(label[s_h:e_h, s_w:e_w])
                else:
                    crop_labels.append(label[s_h:e_h, s_w:e_w, :])

    print('Sliding crop finished. %d pairs of images created.' % len(crop_imgs1))
    return crop_imgs1, crop_imgs2, crop_labels


def rand_crop_CD(img1, img2, label, size):
    # print(img.shape)
    h = img1.shape[0]
    w = img1.shape[1]
    c_h = size[0]
    c_w = size[1]
    if h < c_h or w < c_w:
        print("Cannot crop area {} from image with size ({}, {})"
              .format(str(size), h, w))
    else:
        s_h = random.randint(0, h - c_h)
        e_h = s_h + c_h
        s_w = random.randint(0, w - c_w)
        e_w = s_w + c_w

        crop_im1 = img1[s_h:e_h, s_w:e_w, :]
        crop_im2 = img2[s_h:e_h, s_w:e_w, :]
        crop_label = label[s_h:e_h, s_w:e_w]
        # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
        return crop_im1, crop_im2, crop_label


def rand_flip_CD(img1, img2, data_labelA, data_labelB, data_labels):
    r = random.random()
    # showIMG(img.transpose((1, 2, 0)))
    if r < 0.25:
        return img1, img2, data_labelA, data_labelB, data_labels
    elif r < 0.5:
        return np.flip(img1, axis=0).copy(), np.flip(img2, axis=0).copy(), np.flip(data_labelA, axis=0).copy(), np.flip(data_labelB, axis=0).copy(), np.flip(data_labels, axis=0).copy()
    elif r < 0.75:
        return np.flip(img1, axis=1).copy(), np.flip(img2, axis=1).copy(), np.flip(data_labelA, axis=1).copy(), np.flip(data_labelB, axis=1).copy(), np.flip(data_labels, axis=1).copy()
    else:
        return img1[::-1, ::-1, :].copy(), img2[::-1, ::-1, :].copy(), data_labelA[::-1, ::-1].copy(), data_labelB[::-1, ::-1].copy(), data_labels[::-1, ::-1].copy()


def read_RSimages(mode, read_list=False):
    assert mode in ['train', 'val', 'test']
    # img_A_dir = os.path.join(root, mode, 'A')
    # img_B_dir = os.path.join(root, mode, 'A')
    # label_dir = os.path.join(root, mode, 'label')
    img_A_dir = os.path.join(root, 'A')
    img_B_dir = os.path.join(root, 'A')
    label_dir = os.path.join(root, 'label2')

    if mode == 'train' and read_list:
        list_path = os.path.join(root, mode + '0.4_info.txt')
        list_info = open(list_path, 'r')
        data_list = list_info.readlines()
        data_list = [item.rstrip() for item in data_list]
    else:
        data_list = os.listdir(img_A_dir)
    A_list = random.sample(data_list, 3000)
    B_list = random.sample(data_list, 3000)
    data_A, data_B, data_labelA, data_labelB, data_labels = [], [], [], [], []
    for i in range(len(A_list)):
        # if (it[-4:]=='.png'):

        img_A_path = os.path.join(img_A_dir, A_list[i])
        img_B_path = os.path.join(img_B_dir, B_list[i])
        labelA_path = os.path.join(label_dir, A_list[i])
        labelB_path = os.path.join(label_dir, B_list[i])

        # img_A = io.imread(img_A_path)
        img_A = np.array(Image.open(img_A_path))
        img_A = normalize_image(img_A)
        img_B = np.array(Image.open(img_B_path))
        # img_B = io.imread(img_B_path)
        img_B = normalize_image(img_B)
        # labelA = Color2Index(io.imread(labelA_path))
        # labelB = Color2Index(io.imread(labelB_path))
        labelA = Color2Index(np.array(Image.open(labelA_path)))
        labelB = Color2Index(np.array(Image.open(labelB_path)))
        label = np.where(labelA == labelB, 0, 1)

        # label = np.zeros_like(labelA)
        # label[(labelA == 1) ^ (labelB == 1)] = 1

        labelA = del_dim(labelA)
        labelB = del_dim(labelB)
        label = del_dim(label)

        data_A.append(img_A)
        data_B.append(img_B)
        data_labelA.append(labelA)
        data_labelB.append(labelB)
        data_labels.append(label)

        # if idx>10: break
        if not i % 1000: print('%d/%d images loaded.' % (i, len(A_list)))

    print(data_A[0].shape)
    print(str(len(data_A)) + ' ' + mode + ' images loaded.')
    return data_A, data_B, data_labelA, data_labelB, data_labels


class RS(data.Dataset):
    def __init__(self, mode, random_crop=False, crop_nums=6, sliding_crop=False, crop_size=512, random_flip=False,):
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.crop_nums = crop_nums
        self.crop_size = crop_size
        data_A, data_B, data_labelA, data_labelB, data_labels = read_RSimages(mode, read_list=False)
        self.data_A, self.data_B, self.data_labelA, self.data_labelB, self.data_labels = data_A, data_B, data_labelA, data_labelB, data_labels
        self.len = len(self.data_A)

    def __getitem__(self, idx):
        data_A = self.data_A[idx]
        data_B = self.data_B[idx]
        data_labelA = self.data_labelA[idx]
        data_labelB = self.data_labelB[idx]
        data_labels = self.data_labels[idx]
        if self.random_flip:
            data_A, data_B, data_labelA, data_labelB, data_labels = rand_flip_CD(data_A, data_B, data_labelA, data_labelB, data_labels)
        return F.to_tensor(data_A), F.to_tensor(data_B), data_labelA, data_labelB, data_labels

    def __len__(self):
        return self.len

