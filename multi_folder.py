from torchvision import datasets
import os
import numpy as np
import random
import torch

from IPython import embed

class Multi_Folder(datasets.ImageFolder):

    def __init__(self, root, transform, data_dir):
        super(Multi_Folder, self).__init__(root, transform, data_dir)
        targets = np.asarray([s[1] for s in self.samples]) #target:dataset中图片的class_index
        self.targets = targets
        self.data_dir = data_dir
        find_targets = np.loadtxt(self.data_dir + '/drone_id.txt', dtype=int)
        find_paths = np.loadtxt(self.data_dir + '/drone_path.txt', dtype=str)

        self.find_targets = find_targets
        self.find_paths = find_paths

        data_paths_dict = {}

        unique_id = np.unique(self.find_targets)
        for i in unique_id:
            unique_id_index = np.argwhere(self.find_targets == i)
            data_paths_dict[i] = self.find_paths[unique_id_index].flatten()

        self.data_paths_dict = data_paths_dict

    def _get_id_sample(self, target): #taget: class_index 同类不同光

        pos_paths = self.data_paths_dict[target]

        view_1_range = torch.Tensor([1, 2, 3, 19, 20, 21, 37, 38, 39])
        result_path = []
        for i in range(6):
            view_range = view_1_range + 3 * i - 1
            view_id = int(view_range[random.randint(0, 8)])
            result_path.append(pos_paths[view_id])
        return result_path

    def __getitem__(self, index):
        path, target = self.samples[index]
        # pos_path, neg_path
        pos_path_c = self._get_id_sample(target)

        sample = self.loader(path)
        posc0 = self.loader(self.data_dir + pos_path_c[0])
        posc1 = self.loader(self.data_dir + pos_path_c[1])
        posc2 = self.loader(self.data_dir + pos_path_c[2])
        posc3 = self.loader(self.data_dir + pos_path_c[3])
        posc4 = self.loader(self.data_dir + pos_path_c[4])
        posc5 = self.loader(self.data_dir + pos_path_c[5])

        if self.transform is not None:
            sample = self.transform(sample)
            posc0 = self.transform(posc0)
            posc1 = self.transform(posc1)
            posc2 = self.transform(posc2)
            posc3 = self.transform(posc3)
            posc4 = self.transform(posc4)
            posc5 = self.transform(posc5)

        c, h, w = posc0.shape
        posc = torch.cat((posc0.view(1, c, h, w), posc1.view(1, c, h, w), posc2.view(1, c, h, w), posc3.view(1, c, h, w),
                          posc4.view(1, c, h, w), posc5.view(1, c, h, w)), 0)
        target_of_posc = target

        return sample, target, posc, target_of_posc
