import argparse
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x


class ft_net_VGG16(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_VGG16, self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)
        # avg pooling to global pooling
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.features(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), x.size(1))

        #x = self.classifier(x)
        return x


# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x_fm = x
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), x.size(1))
        #x = self.classifier(x)
        return x, x_fm


class three_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False, VGG16=False):
        super(three_view_net, self).__init__()
        if VGG16:
            self.model_1 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
        else:
            self.model_1 =  ft_net(class_num, stride = stride, pool = pool)
            self.model_2 =  ft_net(class_num, stride = stride, pool = pool)

        if share_weight:
            self.model_3 = self.model_1
        else:
            if VGG16:
                self.model_3 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            else:
                self.model_3 =  ft_net(class_num, stride = stride, pool = pool)
        self.classifier = ClassBlock(2048, class_num, droprate)
        if pool =='avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate)

    def forward(self, x1, x2, x3, x4 = None): # x4 is extra data
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            y2 = self.classifier(x2)

        if x3 is None:
            y3 = None
        else:
            x3 = self.model_3(x3)
            y3 = self.classifier(x3)

        if x4 is None:
            return y1, y2, y3
        else:
            x4 = self.model_2(x4)
            y4 = self.classifier(x4)
            return y1, y2, y3, y4


################################################



class two_view_net_rmac(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False, VGG16=False, gen = False):
        super(two_view_net_rmac, self).__init__()
        if VGG16:
            self.model_1 =  ft_net_VGG16(class_num, stride=stride, pool = pool)
        else:
            self.model_1 =  ft_net(class_num, stride=stride, pool = pool)
        if share_weight:
            self.model_2 = self.model_1
        else:
            if VGG16:
                self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            else:
                self.model_2 =  ft_net(class_num, stride = stride, pool = pool)

        self.classifier = ClassBlock(2048, class_num, droprate, return_f=True)
        if pool =='avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate)
        if VGG16:
            self.classifier = ClassBlock(512, class_num, droprate, return_f=True)
            if pool =='avg+max':
                self.classifier = ClassBlock(1024, class_num, droprate)

        self.maxpool88 = nn.MaxPool2d((8,8))
        self.maxpool66 = nn.MaxPool2d((6,6))
        self.maxpool44 = nn.MaxPool2d((4,4))
        self.gen = gen

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
            f1 = None

        else:
            x1, _ = self.model_1(x1)
            x1 = F.normalize(x1, p=2, dim=1)
            y1, f1 = self.classifier(x1)
            f1 = F.normalize(f1, p=2, dim=1)

        if x2 is None:
            y2 = None
            part_f = None
            f2 = None
        else:
            x2, x2_fm = self.model_2(x2)


            if self.gen:
                part_f = []

                # r-region 2 * 2
                for i in range(2):
                    for j in range(2):
                        step_h = 4
                        step_w = 4

                        part = self.maxpool88(x2_fm[:, :, 0 + i * step_h:8 + i * step_h, 0 + j * step_w:8 + j * step_w])
                        part = part.view(x2_fm.size(0), -1)  # flatten
                        part = F.normalize(part, p=2, dim=1)  # L2 normalize

                        _, part = self.classifier(part)
                        part = F.normalize(part, p=2, dim=1)  # L2 normalize

                        part_f.append(part)

                # r-region 3 * 3
                for i in range(3):
                    for j in range(3):
                        step_h = 3
                        step_w = 3
                        part = self.maxpool66(
                            x2_fm[:, :, 0 + i * step_h:6 + i * step_h, 0 + j * step_w:6 + j * step_w])
                        part = part.view(x2_fm.size(0), -1)  # flatten
                        part = F.normalize(part, p=2, dim=1)  # L2 normalize

                        _, part = self.classifier(part)
                        part = F.normalize(part, p=2, dim=1)  # L2 normalize

                        part_f.append(part)

                # r-region 4 * 4
                for i in range(4):
                    for j in range(4):
                        step_h = 3
                        step_w = 3
                        part = self.maxpool44(
                            x2_fm[:, :, int(0 + i * step_h - min(1, i)):int(4 + i * step_h - min(1, i)),
                            int(0 + j * step_w - min(1, j)):int(4 + j * step_w - min(1, j))])
                        part = part.view(x2_fm.size(0), -1)  # flatten
                        part = F.normalize(part, p=2, dim=1)  # L2 normalize

                        _, part = self.classifier(part)
                        part = F.normalize(part, p=2, dim=1)  # L2 normalize

                        part_f.append(part)

            else:
                part_f = None

            x2 = F.normalize(x2, p=2, dim=1)
            y2, f2 = self.classifier(x2)
            f2 = F.normalize(f2, p=2, dim=1)

        return f1, y1, f2, y2, part_f


class two_view_net_drone2sate(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False, VGG16=False, gen = False):
        super(two_view_net_drone2sate, self).__init__()
        if VGG16:
            self.model_1 =  ft_net_VGG16(class_num, stride=stride, pool = pool)
        else:
            self.model_1 =  ft_net(class_num, stride=stride, pool = pool)
        if share_weight:
            self.model_2 = self.model_1
        else:
            if VGG16:
                self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            else:
                self.model_2 =  ft_net(class_num, stride = stride, pool = pool)

        self.classifier = ClassBlock(2048, class_num, droprate, return_f=True)
        if pool =='avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate)
        if VGG16:
            self.classifier = ClassBlock(512, class_num, droprate, return_f=True)
            if pool =='avg+max':
                self.classifier = ClassBlock(1024, class_num, droprate)

        self.maxpool88 = nn.MaxPool2d((8,8))
        self.maxpool66 = nn.MaxPool2d((6,6))
        self.maxpool44 = nn.MaxPool2d((4,4))
        self.gen = gen

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
            f1 = None

        else:
            x1, _ = self.model_1(x1)
            x1 = F.normalize(x1, p=2, dim=1)
            y1, f1 = self.classifier(x1)
            f1 = F.normalize(f1, p=2, dim=1)

        if x2 is None:
            y2 = None
            part_f = None
            f2 = None
        else:
            x2, x2_fm = self.model_2(x2)
            part_f = None

            x2 = F.normalize(x2, p=2, dim=1)
            y2, f2 = self.classifier(x2)
            f2 = F.normalize(f2, p=2, dim=1)

        return f1, y1, f2, y2, part_f