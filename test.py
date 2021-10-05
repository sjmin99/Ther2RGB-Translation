import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
from image_util import *
from tqdm import tqdm
import math
import numpy as np
import cv2

opt = TestOptions().parse(save=False)
opt.nThreads = 0   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test

model = create_model(opt)

if opt.verbose:
    print(model)

predict_list = []

for i, data in enumerate(dataset):

    ############## Image Processing ##################
    data['label'] = data['label'].cuda()
       
    generated = model.inference(data['label'], data['image'])
    
    
    visuals = OrderedDict([('real_RGB', util.tensor2im(data['real_RGB'], \
                                                        normalize=opt.normalize)),
                        ('fake_RGB', util.tensor2im(generated, \
                                                    normalize=opt.normalize)),        
                        ('input_label', util.tensor2label(data['label'], opt.label_nc))
                        ])
    predict_list.append(visuals['fake_RGB'])
    gt_list.append(visuals['real_RGB'])

    img_path = data['path']
    print('process image... %s' % img_path)

    if opt.save_result is True:
        visualizer.save_images(webpage, visuals, img_path)

predict_list = np.stack(predict_list)

np.save(os.path.join(web_dir,"predict"), predict_list)

webpage.save()
