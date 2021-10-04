import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import cv2
from image_util import *
from util import util
import torch.nn as nn
from torchvision import transforms as tf

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        ### self.A는 Thermal, self.B는 RGB임.
        self.opt = opt
        self.root = opt.dataroot    

        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))
        
        self.A_paths = [path for path in self.A_paths if not 'ipynb_checkpoints' in path]

        ### input B (real images)
        if opt.isTrain:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.A_paths) 

        self.thm_gamma_low = 0.5
        self.thm_gamma_high = 1.5

        contrast_param = (0.3, 1)
        self.colorjitter1 = tf.ColorJitter(contrast=contrast_param)
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]
        
        if self.opt.gray_only:
            A = Image.open(A_path)
        else:
            A = gen_ther_color_pil(A_path)
        params = get_params(self.opt, A.size)

        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A)
        A_tensor = self.colorjitter1(A_tensor)

        if self.opt.isTrain and self.opt.photometric:
            #Photometric agumentation
            thm_random_gamma = np.random.uniform(self.thm_gamma_low, self.thm_gamma_high)
            A_tensor = A_tensor ** thm_random_gamma
            A_tensor = torch.clamp(A_tensor,0,1)

        inst_tensor = feat_tensor = 0
        
        ### input B (real images)
        if self.opt.isTrain:
            B_path = self.B_paths[index]

            B = Image.open(B_path) #load RGB

            transform_B = get_transform(self.opt, params)
            B_tensor = transform_B(B)

            input_dict = {'label': A_tensor, 'image': B_tensor, 'path': A_path}
        else:
            input_dict = {'label': A_tensor, 'path':A_path}
            
        return input_dict
        
    def __len__(self):
        return len(self.A_paths) // self.opt.batch_size * self.opt.batch_size

    def name(self):
        return 'AlignedDataset'