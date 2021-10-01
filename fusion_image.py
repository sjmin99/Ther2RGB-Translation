import subprocess
import os
from os.path import join, isfile, isdir
from os import listdir
from subprocess import call
from tqdm import tqdm
from os.path import join

DATASET_DIR1='./datasets/MTN'
DATASET_DIR2='./datasets/MTN2'
save_path = './datasets/MTN_fusion'

sets_list = os.listdir(DATASET_DIR1)

for set in tqdm(sets_list):
    if 'train' in set:
        MTN1 = os.listdir(join(DATASET_DIR1,set))
        MTN1.sort()
        MTN2 = os.listdir(join(DATASET_DIR2,set))
        MTN2.sort()
        total_imgs = MTN1 + MTN2
    elif 'test' in set:
        MTN2 = os.listdir(join(DATASET_DIR2,set))
        MTN2.sort()
        total_imgs = MTN2

    for i,name in enumerate(total_imgs):
        if 'jpg' in name:
            fold = DATASET_DIR1
        elif 'png' in name:
            fold = DATASET_DIR2
        else:
            print(name)
            continue
        
        img_name = join(fold, set,name)

        if not os.path.isdir(join(save_path, set)):
            print('make folder'+save_path+set)
            os.makedirs(join(save_path,set))

        call(['cp','-p',join(img_name),join(save_path,set,'I'+str(i).zfill(5)+'.jpg')])
