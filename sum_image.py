import subprocess
import os
from os.path import join, isfile, isdir
from os import listdir
from subprocess import call
from tqdm import tqdm
from os.path import join

DATASET_DIR='/raid/datasets/kaist-rgbt/'
save_path = './datasets/kaist/'
    
input_dir = join(DATASET_DIR,'imageSets')
# image_list = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]

# txt_path = [p for p in image_list if p=='test.txt']
txt_path = join(input_dir,'train-day-04.txt')
# txt_path = join(input_dir,'test-day-20.txt')
f = open(join(txt_path), mode='rt')

for i, line in enumerate(tqdm(f)):
    # import pdb;pdb.set_trace()
    # if not 'V000' in line:
    #     import pdb;pdb.set_trace()
    img_list = line.split('/')
    img_path = join(DATASET_DIR,'images',img_list[0],img_list[1])
    rgb_name = join(img_path,'visible',img_list[2][:-1])+'.jpg'
    ther_name = join(img_path,'lwir',img_list[2][:-1])+'.jpg'
    # if 'Campus' in rgb_name:
    #     scene = 'Campus'
    # elif 'Residential' in rgb_name:
    #     scene = 'Residential'
    # elif 'Urban' in rgb_name:
    #     scene = 'Urban'
    # else:
    #     print("Error")
    # import pdb;pdb.set_trace()

    if not os.path.isdir(join(save_path)):
        print('make folder'+save_path)
        os.makedirs(join(save_path,'train_A'))
        os.makedirs(join(save_path,'train_B'))
        os.makedirs(join(save_path,'test_A'))
        os.makedirs(join(save_path,'test_B'))

    # import pdb;pdb.set_trace()
    if 'train' in txt_path:
        call(['cp','-p',join(DATASET_DIR,ther_name),join(save_path,'train_A','I'+str(i).zfill(5)+'.jpg')])
        call(['cp','-p',join(DATASET_DIR,rgb_name), join(save_path,'train_B','I'+str(i).zfill(5)+'.jpg')])
    elif 'test' in txt_path:
        call(['cp','-p',join(DATASET_DIR,ther_name),join(save_path,'test_A','I'+str(i).zfill(5)+'.jpg')])
        call(['cp','-p',join(DATASET_DIR,rgb_name), join(save_path,'test_B','I'+str(i).zfill(5)+'.jpg')])
    else:
        print("Choose train or test")
