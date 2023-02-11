import numpy as np
import os
import torch


def CalculateReceptiveBoxes(height, width, rf=291., stride=16., padding=145., get_center=False):
    """Calculate receptive boxes for each feature point. (from delf.feature_extractor.py

    Args:
      height: The height of feature map.
      width: The width of feature map.
      rf: The receptive field size.
      stride: The effective stride between two adjacent feature points.
      padding: The effective padding size.
    Returns:
      rf_boxes: [N, 4] receptive boxes tensor. Here N equals to height x width.
      Each box is represented by [ymin, xmin, ymax, xmax].
    """
    x, y = np.meshgrid(range(width), range(height))
    coordinates = np.reshape(np.stack([y, x], axis=2), [-1, 2])
    point_boxes = np.concatenate([coordinates, coordinates], 1).astype(np.float32)
    bias = [-padding, -padding, -padding + rf - 1, -padding + rf - 1]
    rf_boxes = stride * point_boxes + bias
    if get_center:
        return np.vstack(((rf_boxes[:, 0] + rf_boxes[:, 2]) / 2.0, (rf_boxes[:, 1] + rf_boxes[:, 3]) / 2.0)).T
    else:
        return rf_boxes

def coord_cal_train(opt, path, coord, flip_flag):
    map_tab = CalculateReceptiveBoxes(12, 12, 483.0, 32.0, 241.0, True)
    corner_output = []

    for i in range(opt.batchsize):
        corner_path = '/net/per610a/export/das18a/satoh-lab/zelong/dataset/University/University-Release/detection/bbox_coord/' + \
                      path[i].split('/')[-2] + '_' + os.path.splitext(os.path.basename(path[i]))[0] + '.boxes.npy'
        corner = np.load(corner_path)

        coord_sample = [coord[0][i], coord[1][i]]

        flip_flag_sample = flip_flag[i]

        sample_coner = []
        for j in corner:

            if flip_flag_sample:
                corner_1_cache = np.minimum(np.maximum(j[0] + 10.0 - coord_sample, 0.0), 384.0)
                corner_2_cache = np.minimum(np.maximum(j[1] + 10.0 - coord_sample, 0.0), 384.0)

                corner_1 = np.array([corner_1_cache[0], 384.0 - corner_2_cache[1]])
                corner_2 = np.array([corner_2_cache[0], 384.0 - corner_1_cache[1]])

            else:
                corner_1 = np.minimum(np.maximum(j[0] + 10.0 - coord_sample, 0.0), 384.0)
                corner_2 = np.minimum(np.maximum(j[1] + 10.0 - coord_sample, 0.0), 384.0)

            fm_coner_topleft_sort = np.linalg.norm((map_tab - corner_1), axis=1).argsort()[0]
            fm_coner_bottomright_sort = np.linalg.norm((map_tab - corner_2), axis=1).argsort()[0]

            fm_coner = [fm_coner_topleft_sort, fm_coner_bottomright_sort]

            sample_coner.append(fm_coner)
        sample_coner = np.array(sample_coner)

        corner_output.append(torch.from_numpy(sample_coner).cuda().detach())

    return corner_output


def coord_cal_test(opt, path):
    map_tab = CalculateReceptiveBoxes(12, 12, 483.0, 32.0, 241.0, True)
    batchsize_now = opt.batchsize

    corner_output = []
    corner_output_flip =[]

    for i in range(batchsize_now):
        corner_path = '/net/per610a/export/das18a/satoh-lab/zelong/dataset/University/University-Release/detection/query_street_bbox/' + \
                      path[i].split('/')[-2] + '_' + os.path.splitext(os.path.basename(path[i]))[
                          0] + '.boxes.npy'
        corner = np.load(corner_path)

        ###############

        sample_coner = []
        sample_coner_flip = []
        for j in corner:
            corner_1_cache = j[0]
            corner_2_cache = j[1]

            corner_1_flip = np.array([corner_1_cache[0], 384.0 - corner_2_cache[1]])
            corner_2_flip = np.array([corner_2_cache[0], 384.0 - corner_1_cache[1]])

            corner_1 = corner_1_cache
            corner_2 = corner_2_cache

            fm_coner_topleft_sort = np.linalg.norm((map_tab - corner_1), axis=1).argsort()[0]
            fm_coner_bottomright_sort = np.linalg.norm((map_tab - corner_2), axis=1).argsort()[0]

            fm_coner_topleft_sort_flip = np.linalg.norm((map_tab - corner_1_flip), axis=1).argsort()[0]
            fm_coner_bottomright_sort_flip = np.linalg.norm((map_tab - corner_2_flip), axis=1).argsort()[0]

            fm_coner = [fm_coner_topleft_sort, fm_coner_bottomright_sort]
            fm_coner_flip = [fm_coner_topleft_sort_flip, fm_coner_bottomright_sort_flip]

            sample_coner.append(fm_coner)
            sample_coner_flip.append(fm_coner_flip)
        sample_coner = np.array(sample_coner)
        sample_coner_flip = np.array(sample_coner_flip)

        corner_output.append(torch.from_numpy(sample_coner).cuda().detach())
        corner_output_flip.append(torch.from_numpy(sample_coner_flip).cuda().detach())

        ###############

    return corner_output, corner_output_flip