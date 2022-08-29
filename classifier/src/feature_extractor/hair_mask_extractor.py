#!/usr/bin/python
# -*- encoding: utf-8 -*-

from .face_parsing_PyTorch.model import BiSeNet
import sys
sys.path.append('./')
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

def parsing_hair_mask(parsing_anno, stride,include_hat=False,include_ear=False):
    # Colors for all 20 parts
    part_colors = [255, 255, 255]

    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    #
    # for pi in range(19):
    #     index = np.where(vis_parsing_anno == pi)
    #     temp = np.zeros_like(vis_parsing_anno_color)
    #     temp[index[0], index[1], :] = part_colors
    #
    #     temp = temp.astype(np.uint8)
    #     cv2.imshow(str(pi),temp)
    #     cv2.waitKey(0)


   # 0 17 18
    indexes = [17]
    if include_hat:
        indexes .append(18)
    if include_ear:
        indexes.append(7)
        indexes.append(8)
    hair_mask = np.zeros_like(vis_parsing_anno_color, np.uint8)
    for pi in indexes:
        index = np.where(vis_parsing_anno == pi)
        temp = np.zeros_like(vis_parsing_anno_color)
        temp[index[0], index[1], :] = part_colors

        temp = temp.astype(np.uint8)
        hair_mask = np.bitwise_or(temp, hair_mask)



    return hair_mask



def parsing_app_mask(parsing_anno, stride,include_hat=False,include_ear=False):
    # Colors for all 20 parts
    part_colors = [255, 255, 255]

    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    indexes = [17]
    if include_hat:
        indexes .append(18)
    if include_ear:
        indexes.append(7)
        indexes.append(8)
    hair_mask = np.zeros_like(vis_parsing_anno_color, np.uint8)
    for pi in indexes:
        index = np.where(vis_parsing_anno == pi)
        temp = np.zeros_like(vis_parsing_anno_color)
        temp[index[0], index[1], :] = part_colors

        temp = temp.astype(np.uint8)
        hair_mask = np.bitwise_or(temp, hair_mask)

    indexes = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 15]
    face_mask = np.zeros_like(vis_parsing_anno_color, np.uint8)
    for pi in indexes:
        index = np.where(vis_parsing_anno == pi)
        temp = np.zeros_like(vis_parsing_anno_color)
        temp[index[0], index[1], :] = part_colors

        temp = temp.astype(np.uint8)
        face_mask = np.bitwise_or(temp, face_mask)

    pi = 0
    bg_mask = np.zeros_like(vis_parsing_anno_color, np.uint8)
    index = np.where(vis_parsing_anno == pi)
    temp = np.zeros_like(vis_parsing_anno_color)
    temp[index[0], index[1], :] = part_colors

    temp = temp.astype(np.uint8)
    bg_mask = np.bitwise_or(temp, bg_mask)

    return face_mask,bg_mask,hair_mask

def parsing_face_mask(parsing_anno, stride,include_ear=False):
    # Colors for all 20 parts
    part_colors = [255, 255, 255]

    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    #
    # for pi in range(19):
    #     index = np.where(vis_parsing_anno == pi)
    #     temp = np.zeros_like(vis_parsing_anno_color)
    #     temp[index[0], index[1], :] = part_colors
    #
    #     temp = temp.astype(np.uint8)
    #     cv2.imshow(str(pi),temp)
    #     cv2.waitKey(0)


    # 14 neck
    indexes = [1,2,3,4,5,6,9,10,11,12,13,15]
    if include_ear:
        indexes.append(7)
        indexes.append(8)
    face_mask = np.zeros_like(vis_parsing_anno_color, np.uint8)
    for pi in indexes:
        index = np.where(vis_parsing_anno == pi)
        temp = np.zeros_like(vis_parsing_anno_color)
        temp[index[0], index[1], :] = part_colors

        temp = temp.astype(np.uint8)
        face_mask = np.bitwise_or(temp, face_mask)



    return face_mask

def parsing_bg_mask(parsing_anno, stride):
    # Colors for all 20 parts
    part_colors = [255, 255, 255]

    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    # 14 neck
    bg_mask = np.zeros_like(vis_parsing_anno_color, np.uint8)
    pi=0
    index = np.where(vis_parsing_anno == pi)
    temp = np.zeros_like(vis_parsing_anno_color)
    temp[index[0], index[1], :] = part_colors

    temp = temp.astype(np.uint8)
    bg_mask = np.bitwise_or(temp, bg_mask)

    return bg_mask

def get_parsingNet(save_pth=''):#(cp='79999_iter.pth'):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    if save_pth=='':
        cur_dir = os.path.dirname(__file__)
        save_pth = os.path.join(cur_dir, os.path.join('face_parsing_PyTorch/res/cp', '79999_iter.pth'))



    print('load face_parsing model from: ', save_pth)
    net.load_state_dict(torch.load(save_pth))
    net.eval()
    return net


def get_hair_mask(img_path,net=None, cp='79999_iter.pth',dilate_kernel=10,include_hat=False,blur=False,include_ear=False):
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path
    img_shape = img.shape
    if net==None:

        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        cur_dir = os.path.dirname(__file__)
        save_pth = os.path.join(cur_dir, os.path.join('face_parsing_PyTorch/res/cp', cp))
        net.load_state_dict(torch.load(save_pth))
        net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])



    with torch.no_grad():
        if isinstance(img_path, str):
            image =Image.open(img_path)
        else:
            image =Image.fromarray(img_path)

        image = image.resize((512, 512), Image.BILINEAR)
        img_input= to_tensor(image)
        img_input = torch.unsqueeze(img_input, 0)
        img_input = img_input.cuda()
        out = net(img_input)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        hair_mask=parsing_hair_mask(parsing, stride=1,include_hat=include_hat,include_ear=include_ear)

    if dilate_kernel>0:
        hair_mask = cv2.dilate(hair_mask, kernel=np.ones((dilate_kernel, dilate_kernel), np.uint8))
    hair_mask=cv2.resize(hair_mask,(img_shape[1],img_shape[0]))

    if blur:
        hair_mask = cv2.dilate(hair_mask, kernel=np.ones((10, 10), np.uint8))
        hair_mask = cv2.blur(hair_mask, ksize=(25, 25))

    return hair_mask


def get_back_ground(img_path,net=None, cp='79999_iter.pth'):
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path
    img_shape = img.shape
    if net==None:

        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        cur_dir = os.path.dirname(__file__)
        save_pth = os.path.join(cur_dir, os.path.join('face_parsing_PyTorch/res/cp', cp))
        net.load_state_dict(torch.load(save_pth))
        net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])



    with torch.no_grad():
        if isinstance(img_path, str):
            image =Image.open(img_path)
        else:
            image =Image.fromarray(img_path)

        image = image.resize((512, 512), Image.BILINEAR)
        img_input= to_tensor(image)
        img_input = torch.unsqueeze(img_input, 0)
        img_input = img_input.cuda()
        out = net(img_input)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        bg_mask=parsing_bg_mask(parsing, stride=1)


    bg_mask=cv2.resize(bg_mask,(img_shape[1],img_shape[0]))

    return bg_mask


def get_face_mask(img_path,net=None, cp='79999_iter.pth',include_ear=False):
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path
    img_shape = img.shape
    if net==None:

        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        cur_dir = os.path.dirname(__file__)
        save_pth = os.path.join(cur_dir, os.path.join('face_parsing_PyTorch/res/cp', cp))
        net.load_state_dict(torch.load(save_pth))
        net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])



    with torch.no_grad():
        if isinstance(img_path, str):
            image =Image.open(img_path)
        else:
            image =Image.fromarray(img_path)

        image = image.resize((512, 512), Image.BILINEAR)
        img_input= to_tensor(image)
        img_input = torch.unsqueeze(img_input, 0)
        img_input = img_input.cuda()
        out = net(img_input)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        face_mask=parsing_face_mask(parsing, stride=1,include_ear=include_ear)

    # if erode_kernel>0:
    #     face_mask = cv2.erode(face_mask, kernel=np.ones((erode_kernel, erode_kernel), np.uint8))
    face_mask=cv2.resize(face_mask,(img_shape[1],img_shape[0]))

    # if blur:
    #     face_mask = cv2.dilate(face_mask, kernel=np.ones((10, 10), np.uint8))
    #     face_mask = cv2.blur(face_mask, ksize=(25, 25))

    return face_mask

def get_app_mask(img_path,net=None, cp='79999_iter.pth',dilate_kernel=10,include_hat=False,blur=False,include_ear=False):
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path
    img_shape = img.shape
    if net==None:

        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        cur_dir = os.path.dirname(__file__)
        save_pth = os.path.join(cur_dir, os.path.join('face_parsing_PyTorch/res/cp', cp))
        net.load_state_dict(torch.load(save_pth))
        net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])



    with torch.no_grad():
        if isinstance(img_path, str):
            image =Image.open(img_path)
        else:
            image =Image.fromarray(img_path)

        image = image.resize((512, 512), Image.BILINEAR)
        img_input= to_tensor(image)
        img_input = torch.unsqueeze(img_input, 0)
        img_input = img_input.cuda()
        out = net(img_input)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        face_mask,face_bg_mask,hair_mask=parsing_app_mask(parsing, stride=1,include_hat=include_hat,include_ear=include_ear)


    hair_mask=cv2.resize(hair_mask,(img_shape[1],img_shape[0]))
    face_bg_mask = cv2.resize(face_bg_mask, (img_shape[1], img_shape[0]))
    face_mask = cv2.resize(face_mask, (img_shape[1], img_shape[0]))

    return face_mask,face_bg_mask,hair_mask