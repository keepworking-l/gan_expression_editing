from .base_dataset import BaseDataset
import os
import random
import numpy as np
import csv

class CelebADataset(BaseDataset):
    """docstring for CelebADataset"""
    def __init__(self):
        super(CelebADataset, self).__init__()
        
    def initialize(self, opt):
        super(CelebADataset, self).initialize(opt)

    def get_aus_by_path(self, img_path):
        assert os.path.isfile(img_path), "Cannot find image file: %s" % img_path
        img_id = str(os.path.splitext(os.path.basename(img_path))[0])
        # print(img_id)
        # print(self.aus_dict[img_id].dtype)#<class 'numpy.ndarray'>
        return self.aus_dict[img_id] / 5.0   # norm to [0, 1]

    def make_dataset(self):
        # return all image full path in a list
        imgs_path = []
        assert os.path.isfile(self.imgs_name_file), "%s does not exist." % self.imgs_name_file
        with open(self.imgs_name_file, 'r') as f:
            lines = f.readlines()
            imgs_path = [os.path.join(self.imgs_dir, line.strip()) for line in lines]
            imgs_path = sorted(imgs_path)
        return imgs_path

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        if not self.opt.tes:
            # load source image
            src_img = self.get_img_by_path(img_path)
            src_img_tensor = self.img2tensor(src_img)
            src_aus = self.get_aus_by_path(img_path)
        else:
            with open('datasets/celebA/au/test.csv', 'r') as f:
                reader = csv.reader(f)
                re = list(reader)
                src_aus = np.array(re[self.opt.tes][-35:-18], dtype=np.float64)
                src_aus = src_aus / 5.0
                src_img_path = 'datasets/celebA\\imgs\\'+str(self.opt.tes)+'.png'
                # print(tar_img_path)
                src_img = self.get_img_by_path(src_img_path)
                src_img_tensor = self.img2tensor(src_img)
        # list1 = [0]
        # list1.append(random.randint(1, 8))
        # self.opt.func = random.choice(list1)
        # print(self.opt.func)
        if self.opt.func:
            with open('datasets/celebA/au/all.csv', 'r') as f:
                reader = csv.reader(f)
                re = list(reader)
                tar_aus = np.array(re[(self.opt.func - 1) * 3][-35:-18], dtype=np.float64)
                tar_aus = tar_aus / 5.0
                tar_img_path = 'datasets/celebA\\imgs\\b'+str((self.opt.func - 1) * 3)+'.jpg'
                # print(tar_img_path)
                tar_img = self.get_img_by_path(tar_img_path)
                tar_img_tensor = self.img2tensor(tar_img)
        else:
            # load target image
            #path = ['datasets/celebA\\imgs\\000072.jpg']
            #tar_img_path = random.choice(path)
            tar_img_path = random.choice(self.imgs_path)
            # print(type(tar_img_path))
            tar_img = self.get_img_by_path(tar_img_path)
            tar_img_tensor = self.img2tensor(tar_img)
            tar_aus = self.get_aus_by_path(tar_img_path)
            if self.is_train and not self.opt.no_aus_noise:
                tar_aus = tar_aus + np.random.uniform(-0.1, 0.1, tar_aus.shape)

        # record paths for debug and test usage
        data_dict = {'src_img':src_img_tensor, 'src_aus':src_aus, 'tar_img':tar_img_tensor, 'tar_aus':tar_aus, \
                        'src_path':img_path, 'tar_path':tar_img_path}

        return data_dict
