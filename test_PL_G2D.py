# -*- coding: utf-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
from torchvision.transforms import InterpolationMode

import argparse
import numpy as np
import time
import os
import scipy.io
import yaml
import math
import h5py
from tqdm import tqdm

from utils_PLCD import load_network

#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='./data/test',type=str, help='./test_data')
parser.add_argument('--name', default='three_view_long_share_d0.75_256_s1_google', type=str, help='save model path')
parser.add_argument('--pool', default='avg', type=str, help='avg|max')
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='views')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--low_altitude', action='store_true', help='use low altitude drone-view gallery images' )

opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
opt.fp16 = config['fp16']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']
opt.views = config['views']

if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 729

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


data_dir = test_dir

if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery_satellite','gallery_drone', 'gallery_street',
                                                                                                    'query_satellite', 'query_drone', 'query_street',
                                                                                                    'gallery_drone_low','gallery_drone_middle','gallery_drone_high']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery_satellite', 'gallery_drone','gallery_street',
                                                                                      'query_satellite', 'query_drone', 'query_street',
                                                                                      'gallery_drone_low',
                                                                                      'gallery_drone_middle',
                                                                                      'gallery_drone_high']}
use_gpu = torch.cuda.is_available()

length1 = len(image_datasets['gallery_drone'])
length2 = len(image_datasets['query_street'])

if opt.low_altitude:
    length1 = len(image_datasets['gallery_drone_low'])
print(length1)

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def which_view(name):
    if 'satellite' in name:
        return 1
    elif 'street' in name:
        return 2
    elif 'drone' in name:
        return 3
    else:
        print('unknown view')
    return -1

def extract_feature_h5(model,dataloaders, save_path):
    view_index = 3
    features = torch.FloatTensor()
    count = 0
    with h5py.File(save_path + '/pytorch.hdf5', mode='w') as h5:
        with torch.no_grad():
            print('====> Extracting Gallery Descriptors')
            dbFeat = h5.create_dataset('gallery_feature',
                                       [length1, 512],
                                       dtype=np.float32)
        if opt.low_altitude:
            data_iterator_gallery_drone = tqdm(dataloaders['gallery_drone_low'], desc='Gallery-drone Testing...')
        else:
            data_iterator_gallery_drone = tqdm(dataloaders['gallery_drone'], desc='Gallery-drone Testing...')
        for iteration, data in enumerate(data_iterator_gallery_drone):
            img, _ = data
            n, c, h, w = img.size()
            count += n
            # print(count)
            ff = torch.FloatTensor(n,512).zero_().cuda()

            for i in range(2):
                if(i==1):
                    img = fliplr(img)
                input_img = Variable(img.cuda())
                for scale in ms:
                    if scale != 1:
                        # bicubic is only  available in pytorch>= 1.1
                        input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
                    if opt.views == 2:
                        _, _,  outputs, _, _ = model(None, input_img)
                    elif opt.views ==3:
                        if view_index == 1:
                            outputs, _, _ = model(input_img, None, None)
                        elif view_index ==2:
                            _, outputs, _ = model(None, input_img, None)
                        elif view_index ==3:
                            _, _, outputs = model(None, None, input_img)
                    ff += outputs
            # norm feature
            if opt.PCB:
                # feature size (n,2048,6)
                # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
                # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
                ff = ff.div(fnorm.expand_as(ff))
                ff = ff.view(ff.size(0), -1)
            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))

                ff = ff.data.cpu()

                #print(ff.shape)

            batchix = iteration * opt.batchsize
            dbFeat[batchix:batchix + opt.batchsize, :] = ff[:, :].detach().cpu().numpy()

            del input_img, ff

        view_index = 2

        with torch.no_grad():
            print('====> Extracting query Descriptors')
            dbFeat = h5.create_dataset('query_feature',
                                       [length2, 512],
                                       dtype=np.float32)

        data_iterator_query_street = tqdm(dataloaders['query_street'], desc='Query-ground Testing...'.format(epoch))
        for iteration, data in enumerate(data_iterator_query_street):
            img, _ = data
            n, c, h, w = img.size()
            count += n
            # print(count)
            ff = torch.FloatTensor(n,512).zero_().cuda()

            for i in range(2):
                if(i==1):
                    img = fliplr(img)
                input_img = Variable(img.cuda())
                for scale in ms:
                    if scale != 1:
                        # bicubic is only  available in pytorch>= 1.1
                        input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
                    if opt.views == 2:
                        outputs, _, _, _, _ = model(input_img, None)
                    elif opt.views ==3:
                        if view_index == 1:
                            outputs, _, _ = model(input_img, None, None)
                        elif view_index ==2:
                            _, outputs, _ = model(None, input_img, None)
                        elif view_index ==3:
                            _, _, outputs = model(None, None, input_img)
                    ff += outputs
            # norm feature
            if opt.PCB:
                # feature size (n,2048,6)
                # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
                # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
                ff = ff.div(fnorm.expand_as(ff))
                ff = ff.view(ff.size(0), -1)
            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))

                ff = ff.data.cpu()

            batchix = iteration * opt.batchsize
            dbFeat[batchix:batchix + opt.batchsize, :] = ff[:, :].detach().cpu().numpy()

            del input_img, ff

    print('====> Done!')


def get_id(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths

######################################################################
# Load Collected data Trained model
print('-------test-----------')

model, _, epoch = load_network(opt.name, opt)
#model.classifier.classifier = nn.Sequential()
model = model.eval()
if use_gpu:
    model = model.cuda()
    #model = nn.DataParallel(model, device_ids=gpu_ids)

# Extract feature
since = time.time()

#gallery_name = 'gallery_street'
#query_name = 'query_satellite'

gallery_name = 'gallery_drone'
query_name = 'query_street'

if opt.low_altitude:
    gallery_name = 'gallery_drone_low'

#gallery_name = 'gallery_street'
#query_name = 'query_drone'
#gallery_name = 'gallery_drone'

which_gallery = which_view(gallery_name)
which_query = which_view(query_name)
print('%d -> %d:'%(which_query, which_gallery))

gallery_path = image_datasets[gallery_name].imgs
f = open('gallery_name.txt','w')
for p in gallery_path:
    f.write(p[0]+'\n')
query_path = image_datasets[query_name].imgs
f = open('query_name.txt','w')
for p in query_path:
    f.write(p[0]+'\n')

gallery_label, gallery_path  = get_id(gallery_path)
query_label, query_path  = get_id(query_path)

save_path = os.path.join('./model',opt.name)

if __name__ == "__main__":
    with torch.no_grad():
        extract_feature_h5(model, dataloaders, save_path)


    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Save to Matlab for check
    result = {'gallery_label':gallery_label,'gallery_path':gallery_path,'query_label':query_label, 'query_path':query_path}
    scipy.io.savemat(save_path + '/pytorch_result.mat',result)
