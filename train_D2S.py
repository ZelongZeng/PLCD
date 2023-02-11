# -*- coding: utf-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torchvision.transforms import InterpolationMode

import matplotlib
matplotlib.use('agg')
import argparse
import copy
import time
import os
import random
import yaml
from tqdm import tqdm
import numpy as np
from shutil import copyfile

from model_rmac import two_view_net_drone2sate
from random_erasing import RandomErasing
from autoaugment import ImageNetPolicy, CIFAR10Policy
from utils_PLCD import update_average, load_network, save_network
from multi_folder import Multi_Folder



version = torch.__version__
# fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError:  # will be 3.x series
    print(
        'This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='two_view', type=str, help='output model name')
parser.add_argument('--old_name', default='two_view', type=str, help='output model name')
parser.add_argument('--pool', default='max', type=str, help='pool avg')
parser.add_argument('--data_dir', default='./data/train', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--pad', default=10, type=int, help='padding')
parser.add_argument('--h', default=384, type=int, help='height')
parser.add_argument('--w', default=384, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='the number of views')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_NAS', action='store_true', help='use NAS')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation')
parser.add_argument('--resume', action='store_true', help='use resume trainning')
parser.add_argument('--share', action='store_true', help='share weight between different view')
parser.add_argument('--extra_Google', action='store_true', help='using extra noise Google')
parser.add_argument('--fp16', action='store_true',
                    help='use float16 instead of float32, which will save about 50% memory')
parser.add_argument('--margin', default=0.3, type=float, help='the margin value of triplet loss')
parser.add_argument('--gen', default=0, type=int, help='the generation')
parser.add_argument('--tau', default=0.1, type=float, help='tau')
parser.add_argument('--hiera', default=0, type=int, help='the hierarchy')

parser.add_argument('--seed', default=0, type=int, help='the seed')

parser.add_argument('--use_detection', action='store_true', help='use detection')

parser.add_argument('--easy_pos', action='store_true', help='hard_negative')

parser.add_argument('--lambda1', default=1., type=float, help='lambda1')
parser.add_argument('--lambda2', default=1., type=float, help='lambda2')
parser.add_argument('--lambda3', default=1., type=float, help='lambda2')

opt = parser.parse_args()

if opt.resume:
    model, opt, start_epoch = load_network(opt.name, opt)
else:
    start_epoch = 0

fp16 = opt.fp16
data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

"""==================================================================================================="""
#################### SEEDS FOR REPROD. #####################
torch.backends.cudnn.deterministic=True; np.random.seed(opt.seed); random.seed(opt.seed)
torch.manual_seed(opt.seed); torch.cuda.manual_seed(opt.seed); torch.cuda.manual_seed_all(opt.seed)

######################################################################
# Load Data
# ---------
#

transform_train_list = [
    # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((opt.h, opt.w), interpolation=InterpolationMode.BICUBIC),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_satellite_list = [
    transforms.Resize((opt.h, opt.w), interpolation=InterpolationMode.BICUBIC),
    transforms.Pad(opt.pad, padding_mode='edge'),
    transforms.RandomAffine(90),
    transforms.RandomCrop((opt.h, opt.w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(opt.h, opt.w), interpolation=InterpolationMode.BICUBIC),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                   hue=0)] + transform_train_list
    transform_satellite_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                       hue=0)] + transform_satellite_list

if opt.DA:
    transform_train_list = [ImageNetPolicy()] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
    'satellite': transforms.Compose(transform_satellite_list)}

train_all = ''
if opt.train_all:
    train_all = '_all'

image_datasets = {}
image_datasets['satellite'] = Multi_Folder(os.path.join(data_dir, 'satellite'),
                                                data_transforms['train'], data_dir)


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=2, pin_memory=True)  # 8 workers may work faster
               for x in ['satellite']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['satellite']}
class_names = image_datasets['satellite'].classes
print(dataset_sizes)
use_gpu = torch.cuda.is_available()

######################################################################
# Training the model
# ------------------

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def kl_d(input, target):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    return kl_loss(input, target)


def train_model(model, model_teacher, model_test, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['satellite'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch

    for epoch in range(num_epochs - start_epoch):
        epoch = epoch + start_epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_loss1 = 0.0
            running_loss2 = 0.0
            running_loss3 = 0.0
            running_corrects = 0.0
            running_corrects2 = 0.0

            data_iterator = tqdm(dataloaders['satellite'], desc='Epoch {} Training...'.format(epoch))

            # Iterate over data.
            for i, data in enumerate(data_iterator):
                # get the inputs
                inputs, labels, pos, pos_labels = data

                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:  # skip the last batch
                    continue

                pos = pos.view(6 * opt.batchsize, c, h, w)
                # copy pos 6 times
                pos_labels = pos_labels.repeat(6).reshape(6, opt.batchsize)
                pos_labels = pos_labels.transpose(0, 1).reshape(6 * opt.batchsize)


                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    pos = Variable(pos.cuda().detach())
                    pos_labels = Variable(pos_labels.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                    # conrner = Variable(corner_output.cuda().detach())
                    if opt.extra_Google:
                        inputs4 = Variable(inputs4.cuda().detach())
                        labels4 = Variable(labels4.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if opt.gen != 0:
                    with torch.no_grad():
                        f_teacher, outputs_teacher, pf_teacher, outputs2_teacher, part_f_teacher = model_teacher(inputs, pos)
                    f, outputs, pf, outputs2, part_f = model(inputs, pos)
                else:
                    f, outputs, pf, outputs2, _ = model(inputs, pos)

                _, preds = torch.max(outputs.data, 1)
                _, preds2 = torch.max(outputs2.data, 1)


                # hard-pos
                # ----------------------------------
                pf_hard = torch.zeros(f.shape).cuda()
                for k in range(now_batch_size):
                    pf_data = pf[6 * k:6 * k + 6, :]
                    if opt.hiera != 0:
                        for l in range(opt.hiera):
                            pf_data = torch.cat((pf_data, part_f[l][6 * k:6 * k + 6, :]), 0)
                    pf_t = pf_data.transpose(0, 1)
                    ff = f.data[k, :].reshape(1, -1)
                    score = torch.mm(ff, pf_t)
                    score, rank = score.sort(dim=1, descending=opt.easy_pos)  # score low == hard
                    pf_hard[k, :] = pf_data[rank[0][0], :]

                # hard-neg
                # ----------------------------------
                neg_labels = pos_labels.cpu()
                nf_data = pf
                if opt.hiera != 0:
                    for l in range(opt.hiera):
                        nf_data = torch.cat((nf_data, part_f[l]), 0)
                    neg_labels = neg_labels.repeat(opt.hiera+1)

                nf_t = nf_data.transpose(0, 1)
                score = torch.mm(f.data, nf_t)
                score, rank = score.sort(dim=1, descending=True)  # score high == hard
                labels_cpu = labels.cpu()
                nf_hard = torch.zeros(10 * f.shape[0], f.shape[1]).cuda()
                for k in range(now_batch_size):
                    hard = rank[k, :]
                    kk_flag = 0
                    for kk in hard:     # select semi-hard negatives, use hard negatives is unstable.
                        now_label = neg_labels[kk]
                        anchor_label = labels_cpu[k]
                        if (now_label != anchor_label):
                            nf_hard[k*10+kk_flag, :] = nf_data[kk, :]
                            kk_flag = kk_flag + 1
                            if kk_flag == 10:
                                break

                # We don't use KD for learning D-S representation, it will decreases the performance.


                # loss
                fff = f.repeat(1, 10).reshape(10 * opt.batchsize, -1)
                pscore = torch.sum(f * pf_hard, dim=1)
                pscore = pscore.repeat(10).reshape(10 * opt.batchsize)
                nscore = torch.sum(fff * nf_hard, dim=1)

                catscore = torch.cat((pscore.unsqueeze(1), nscore.unsqueeze(1)), dim = 1)
                logsoftmaxscore = -1 * F.log_softmax(catscore, dim=1)[:,0]
                loss1 = torch.sum(logsoftmaxscore)/(10 * now_batch_size)
                loss2 = (criterion(outputs, labels) + criterion(outputs2, pos_labels))
                loss = opt.lambda1 * loss1 + opt.lambda2 * loss2


                if epoch < opt.warm_epoch and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':
                    if fp16:  # we use optimier to backward loss
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()
                    ##########
                    if opt.moving_avg < 1.0:
                        update_average(model_test, model, opt.moving_avg)

                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item()
                    running_loss1 += loss1.item()
                    running_loss2 += loss2.item()
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))
                running_corrects2 += float(torch.sum(preds2 == pos_labels.data))

            epoch_loss = running_loss / dataset_sizes['satellite']
            epoch_loss1 = running_loss1 / dataset_sizes['satellite']
            epoch_loss3 = running_loss3 / dataset_sizes['satellite']
            epoch_loss2 = running_loss2 / dataset_sizes['satellite']
            epoch_acc = running_corrects / dataset_sizes['satellite']
            epoch_acc2 = running_corrects2 / dataset_sizes['satellite'] / 6

            print('{} Loss: {:.4f} Loss1: {:.4f} Loss2: {:.4f} Loss3: {:.4f} satellite_Acc: '
                  '{:.4f}  Drone_Acc: {:.4f}'.format(phase, epoch_loss, epoch_loss1, epoch_loss2, epoch_loss3, epoch_acc,
                                                                                     epoch_acc2))
            # deep copy the model
            if phase == 'train':
                scheduler.step()
                save_network(model, opt.name, epoch, opt.seed)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model




######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#


model = two_view_net_drone2sate(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
                         share_weight=opt.share, gen=opt.gen)
if opt.gen != 0:
    model_teacher = two_view_net_drone2sate(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
                              share_weight=opt.share, gen=opt.gen)
    model_teacher.load_state_dict(torch.load('./model/%s/net_119_s%s.pth'%(opt.old_name, opt.seed)))

else:
    model_teacher = None


opt.nclasses = len(class_names)

#print(model)
# For resume:
if start_epoch >= 40:
    opt.lr = opt.lr * 0.1

ignored_params = list(map(id, model.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
#optimizer_ft = optim.SGD(model.parameters(), weight_decay=5e-4, momentum=0.9, nesterov=True, lr = opt.lr)

optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.classifier.parameters(), 'lr': opt.lr}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU.
#
dir_name = os.path.join('./model', name)
if not opt.resume:
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    # record every run
    copyfile('./train_D2S.py', dir_name + '/train_D2S.py')
    copyfile('./model_rmac.py', dir_name + '/model_rmac.py')
    # save opts
    with open('%s/opts.yaml' % dir_name, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
model = model.cuda()
if opt.gen !=0:
    model_teacher = model_teacher.cuda()
#model = nn.DataParallel(model,device_ids=gpu_ids)
if fp16:
    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level="O1")

criterion = nn.CrossEntropyLoss()
if opt.moving_avg < 1.0:
    model_test = copy.deepcopy(model)
    num_epochs = 140
else:
    model_test = None
    num_epochs = 120

model = train_model(model, model_teacher, model_test, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs=num_epochs)