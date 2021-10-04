import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import cv2

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        if not self.opt.isTrain and self.opt.gray_only:
            dir_A = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)  
            self.A_paths = sorted(make_dataset(self.dir_A))
            if '.ipynb_checkpoints' in self.A_paths[0]:
                print("delete_.ipynb_checkpoints")
                self.A_paths = self.A_paths[1:]
        else:
            dir_A = '_A' if self.opt.label_nc == 0 else '_label'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A))
            if '.ipynb_checkpoints' in self.A_paths[0]:
                print("delete_.ipynb_checkpoints")
                self.A_paths = self.A_paths[1:]

        ### input B (real images)
        # if opt.isTrain or opt.use_encoded_image:
        dir_B = '_B' if self.opt.label_nc == 0 else '_img'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
        self.B_paths = sorted(make_dataset(self.dir_B))
        if '.ipynb_checkpoints' in self.B_paths[0]:
                print("delete_.ipynb_checkpoints")
                self.B_paths = self.B_paths[1:]
        
        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 

        self.thm_gamma_low = 0.5
        self.thm_gamma_high = 1.5
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]
        A = cv2.imread(A_path)[:,:,[0]]
        tmp = cv2.imread(self.B_paths[index])
        tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(tmp)
        Y, Cr, Cb = np.expand_dims(Y,2), np.expand_dims(Cr, 2), np.expand_dims(Cb,2)
        A = np.concatenate([A, Cr, Cb], -1)
        A = Image.fromarray(A)

        # A = Image.open(A_path).convert('YCbCr')
        # import pdb;pdb.set_trace()
        params = get_params(self.opt, A.size)

        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A)
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0
            
        # thm_random_gamma = np.random.uniform(self.thm_gamma_low, self.thm_gamma_high)
        # A_tensor = A_tensor ** thm_random_gamma
        # A_tensor = torch.clamp(A_tensor,0,1)
        # B_tensor = A_tensor[[0],:,:]

        # A_tensor = A_tensor[1:,:,:]
        inst_tensor = feat_tensor = 0
#         B_tensor = inst_tensor = feat_tensor = 0

        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]
            if self.opt.output_nc==1:
                # B = Image.open(B_path).convert('L')
                B = Image.fromarray(Y[:,:,0])
            else:
                B = tmp
                B = Image.fromarray(B)
            
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)
        else:
            B_tensor=0
            B_path = self.B_paths[index]
            # rgb = Image.open(B_path).convert('RGB')
            rgb = cv2.imread(B_path)
            rgb = Image.fromarray(rgb)
            transform_rgb = get_transform(self.opt, params)
            rgb = transform_rgb(rgb)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}
        if not self.opt.isTrain:
            input_dict['real_RGB'] = rgb
        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batch_size * self.opt.batch_size

    def name(self):
        return 'AlignedDataset'