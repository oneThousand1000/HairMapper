from torch.utils.data import Dataset
import numpy as np
import cv2
import os


class LatentsDataset(Dataset):

    def __init__(self, data_dir,mode='train'):
        self.data_dir = data_dir
        self.data_path = os.path.join(data_dir,f'{mode}.txt')
        self.data = open(self.data_path, 'r').readlines()

        self.original_code = np.load(os.path.join(self.data_dir,'original_wp.npy'))
        print('dataset size: ', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        words=self.data[index].split(' ')
        origin_latent = np.reshape(self.original_code[int(words[0]),:,:], (18, 512))

        res_path=words[1]

        mask_path= words[2].replace('\n', '')

        res_latent = np.reshape(np.load(res_path),(18,512))
        mask= cv2.imread(mask_path)
        mask=mask.transpose(2,0,1)
        mask=1-mask//255

        return origin_latent, res_latent, mask


class LatentsTestDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        mode='test'
        self.data_path = os.path.join(data_dir, f'{mode}.txt')
        self.data = open(self.data_path, 'r').readlines()
        print('dataset size: ', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        words=self.data[index].split(' ')
        origin_img_path=words[0]
        origin_wp_path=words[1]
        mask_path= words[2].replace('\n', '')

        origin_latent = np.reshape(np.load(origin_wp_path),(18,512))
        #print(no_hair_latent.shape)
        print(mask_path)
        mask= cv2.imread(mask_path)
        mask=mask.transpose(2,0,1)
        mask=1-mask//255

        origin_image = cv2.imread(origin_img_path)
        origin_image = (origin_image / 255.0 - 0.5) / 0.5
        origin_image = origin_image[:, :, ::-1].copy()
        origin_image = origin_image.transpose(2, 0, 1)


        return origin_latent,mask,origin_image

# from torch.utils.data import Dataset
# import numpy as np
# import cv2
# import os
# class LatentsDataset(Dataset):
#
#     def __init__(self, data_path):
#         self.data_path = data_path
#         self.data = open(self.data_path, 'r').readlines()
#         print('dataset size: ', len(self.data))
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         words=self.data[index].split(' ')
#         origin_path=words[0]
#
#         res_path=words[1]
#
#         mask_path= words[2].replace('\n', '')
#
#         res_latent = np.reshape(np.load(res_path),(18,512))
#         origin_latent = np.reshape(np.load(origin_path),(18,512))
#         #print(no_hair_latent.shape)
#         mask= cv2.imread(mask_path)
#         mask=mask.transpose(2,0,1)
#         mask=1-mask//255
#
#         return origin_latent, res_latent, mask
#
#
# class LatentsTestDataset(Dataset):
#
#     def __init__(self, data_path):
#         self.data_path = data_path
#
#         self.data = open(self.data_path, 'r').readlines()
#         print('dataset size: ', len(self.data))
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         # origin_image_path + ' ' + codepath + ' ' + mask + '\n'
#         words=self.data[index].split(' ')
#         origin_img_path=words[0]
#         origin_wp_path=words[1]
#         mask_path= words[2].replace('\n', '')
#
#         origin_latent = np.reshape(np.load(origin_wp_path),(18,512))
#         #print(no_hair_latent.shape)
#         print(mask_path)
#         mask= cv2.imread(mask_path)
#         mask=mask.transpose(2,0,1)
#         mask=1-mask//255
#
#         origin_image = cv2.imread(origin_img_path)
#         origin_image = (origin_image / 255.0 - 0.5) / 0.5
#         origin_image = origin_image[:, :, ::-1].copy()
#         origin_image = origin_image.transpose(2, 0, 1)
#
#
#         return origin_latent,mask,origin_image