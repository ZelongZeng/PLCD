from torchvision import datasets
import os
import numpy as np
import random
import torch

class TripletFolder(datasets.ImageFolder):

    def __init__(self, root, transform):
        super(TripletFolder, self).__init__(root, transform)
        targets = np.asarray([s[1] for s in self.samples]) #target:dataset中图片的class_index
        self.targets = targets
        # view_ids = []
        # for s in self.samples:
        #     view_ids.append( self._get_id_of_view(s[0]))
        # self.view_ids = np.asarray(view_ids)

    def _get_id_of_view(self, path):
        filename = os.path.basename(path)
        #illu_id = filename[-5:-4]
        view_id = filename.split('.')[0][-2:]
        return view_id

    def _get_id_sample(self, target, index, o_path): #taget: class_index 同类不同光
        pos_index = np.argwhere(self.targets == target)#去掉不同类的
        pos_illu = o_path[-5:-4]
        pos_illu_index = np.argwhere(self.illu_ids != pos_illu)
        pos_index = pos_index.flatten()
        pos_illu_index = pos_illu_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)#去掉自身
        pos_illu_index = np.setdiff1d(pos_illu_index, index)  # 去掉自身
        pos_index = np.intersect1d(pos_index, pos_illu_index)
        rand = np.random.permutation(len(pos_index))#把结果随机排列
        result_path = []
        for i in range(4):
           t = i%len(rand) # t = 0,1,2,3
           tmp_index = pos_index[rand[t]]
           result_path.append(self.samples[tmp_index][0])#随机选四个 然后读其path
        return result_path

    def _get_illu_sample(self, target, index, o_path): #同光不同类
        pos_index = np.argwhere(self.targets != target)
        pos_illu = o_path[-5:-4]
        pos_illu_index = np.argwhere(self.illu_ids == pos_illu)
        pos_index = pos_index.flatten()
        pos_illu_index = pos_illu_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)  # 去掉自身
        pos_illu_index = np.setdiff1d(pos_illu_index, index)  # 去掉自身
        pos_index = np.intersect1d(pos_index, pos_illu_index)
        rand = np.random.permutation(len(pos_index))  # 把结果随机排列
        result_path = []
        for i in range(4):
            t = i % len(rand)  # t = 0,1,2,3
            tmp_index = pos_index[rand[t]]
            result_path.append(self.samples[tmp_index][0])  # 随机选四个 然后读其path
        return result_path

    def __getitem__(self, index):
        path, target = self.samples[index]
        view_id = self._get_id_of_view(path)
        # pos_path, neg_path



        return target, path, view_id



class Triplet(datasets.ImageFolder):

    def __init__(self, root, transform):
        super(Triplet, self).__init__(root, transform)
        targets = np.asarray([s[1] for s in self.samples]) #target:dataset中图片的class_index
        self.targets = targets
        illu_ids = []
        for s in self.samples:
            illu_ids.append( self._get_illu(s[0]) )
        self.illu_ids = np.asarray(illu_ids)
        find_path = '/net/per610a/export/das18a/satoh-lab/zelong/dataset/University/University-Release/train/'
        find_targets = np.loadtxt(find_path + 'drone_id.txt', dtype=int)
        find_paths = np.loadtxt(find_path + 'drone_path.txt', dtype=str)

        self.find_targets = find_targets
        self.find_paths = find_paths


    def _get_illu(self, path):
        filename = os.path.basename(path)
        illu_id = filename[-5:-4]
        #camera_id = filename.split('_')[2][0:2]
        return illu_id

    def _get_pos_sample(self, target, index, o_path): #taget: class_index 同类不同光
        pos_index = np.argwhere(self.find_targets == target)#去掉不同类的
        pos_index = pos_index.flatten()
        rand = np.random.permutation(len(pos_index))#把结果随机排列
        result_path = []
        for i in range(1):
           #t = i%len(rand) # t = 0,1,2,3
           tmp_index = pos_index[rand[i]]
           result_path.append(self.find_paths[tmp_index])#随机选四个 然后读其path
        return result_path

    def _get_neg_sample(self, target, index, o_path): #taget: class_index 不同类
        pos_index = np.argwhere(self.find_targets != target)#去掉同类的
        pos_index = pos_index.flatten()
        rand = np.random.permutation(len(pos_index))#把结果随机排列
        result_path = []
        for i in range(8):
           #t = i%len(rand) # t = 0,1,2,3
           tmp_index = pos_index[rand[i]]
           result_path.append(self.find_paths[tmp_index])#随机选四个 然后读其path
        return result_path

    def __getitem__(self, index):
        path, target = self.samples[index]
        # pos_path, neg_path
        pos_path_c = self._get_pos_sample(target, index, path)
        neg_path_c = self._get_neg_sample(target, index, path)
        #pos_path_i = self._get_illu_sample(target, index, path)


        sample = self.loader(path)

        posc0 = self.loader(pos_path_c[0])
        # posc1 = self.loader(pos_path_c[1])
        # posc2 = self.loader(pos_path_c[2])
        # posc3 = self.loader(pos_path_c[3])

        negc0 = self.loader(neg_path_c[0])
        negc1 = self.loader(neg_path_c[1])
        negc2 = self.loader(neg_path_c[2])
        negc3 = self.loader(neg_path_c[3])
        negc4 = self.loader(neg_path_c[4])
        negc5 = self.loader(neg_path_c[5])
        negc6 = self.loader(neg_path_c[6])
        negc7 = self.loader(neg_path_c[7])

        if self.transform is not None:
            sample = self.transform(sample)

            posc0 = self.transform(posc0)
            # posc1 = self.transform(posc1)
            # posc2 = self.transform(posc2)
            # posc3 = self.transform(posc3)

            negc0 = self.transform(negc0)
            negc1 = self.transform(negc1)
            negc2 = self.transform(negc2)
            negc3 = self.transform(negc3)
            negc4 = self.transform(negc4)
            negc5 = self.transform(negc5)
            negc6 = self.transform(negc6)
            negc7 = self.transform(negc7)


        c, h, w = posc0.shape
        posc = posc0.view(1, c, h, w)
        negc = torch.cat((negc0.view(1, c, h, w), negc1.view(1, c, h, w), negc2.view(1, c, h, w), negc3.view(1, c, h, w),
                          negc4.view(1, c, h, w), negc5.view(1, c, h, w), negc6.view(1, c, h, w), negc7.view(1, c, h, w)), 0)


        return sample, target, posc, negc

# class Triplet(datasets.ImageFolder):
#
#     def __init__(self, root, transform):
#         super(Triplet, self).__init__(root, transform)
#         targets = np.asarray([s[1] for s in self.samples]) #target:dataset中图片的class_index
#         self.targets = targets
#         illu_ids = []
#         for s in self.samples:
#             illu_ids.append( self._get_illu(s[0]) )
#         self.illu_ids = np.asarray(illu_ids)
#         find_path = '/net/per610a/export/das18a/satoh-lab/zelong/dataset/University/University-Release/train/'
#         find_targets = np.loadtxt(find_path + 'drone_id.txt', dtype=int)
#         find_paths = np.loadtxt(find_path + 'drone_path.txt', dtype=str)
#
#         self.find_targets = find_targets
#         self.find_paths = find_paths
#
#
#     def _get_illu(self, path):
#         filename = os.path.basename(path)
#         illu_id = filename[-5:-4]
#         #camera_id = filename.split('_')[2][0:2]
#         return illu_id
#
#     def _get_pos_sample(self, target, index, o_path): #taget: class_index 同类不同光
#         pos_index = np.argwhere(self.find_targets == target)#去掉不同类的
#         pos_index = pos_index.flatten()
#         rand = np.random.permutation(len(pos_index))#把结果随机排列
#         result_path = []
#         for i in range(1):
#            #t = i%len(rand) # t = 0,1,2,3
#            tmp_index = pos_index[rand[i]]
#            result_path.append(self.find_paths[tmp_index])#随机选四个 然后读其path
#         return result_path
#
#     def _get_neg_sample(self, target, index, o_path): #taget: class_index 不同类
#         pos_index = np.argwhere(self.find_targets != target)#去掉同类的
#         pos_index = pos_index.flatten()
#         rand = np.random.permutation(len(pos_index))#把结果随机排列
#         result_path = []
#         for i in range(4):
#            #t = i%len(rand) # t = 0,1,2,3
#            tmp_index = pos_index[rand[i]]
#            result_path.append(self.find_paths[tmp_index])#随机选四个 然后读其path
#         return result_path
#
#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         # pos_path, neg_path
#         pos_path_c = self._get_pos_sample(target, index, path)
#         neg_path_c = self._get_neg_sample(target, index, path)
#         #pos_path_i = self._get_illu_sample(target, index, path)
#
#
#         sample = self.loader(path)
#
#         posc0 = self.loader(pos_path_c[0])
#         # posc1 = self.loader(pos_path_c[1])
#         # posc2 = self.loader(pos_path_c[2])
#         # posc3 = self.loader(pos_path_c[3])
#
#         negc0 = self.loader(neg_path_c[0])
#         negc1 = self.loader(neg_path_c[1])
#         negc2 = self.loader(neg_path_c[2])
#         negc3 = self.loader(neg_path_c[3])
#
#         if self.transform is not None:
#             sample = self.transform(sample)
#
#             posc0 = self.transform(posc0)
#             # posc1 = self.transform(posc1)
#             # posc2 = self.transform(posc2)
#             # posc3 = self.transform(posc3)
#
#             negc0 = self.transform(negc0)
#             negc1 = self.transform(negc1)
#             negc2 = self.transform(negc2)
#             negc3 = self.transform(negc3)
#
#
#         c, h, w = posc0.shape
#         posc = posc0.view(1, c, h, w)
#         negc = torch.cat((negc0.view(1, c, h, w), negc1.view(1, c, h, w), negc2.view(1, c, h, w), negc3.view(1, c, h, w)), 0)
#
#
#         return sample, target, posc, negc