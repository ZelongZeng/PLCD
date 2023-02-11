import scipy.io
import torch
import numpy as np
# import time
import os
import h5py

from diffusion import Diffusion
from tqdm import tqdm

import argparse

from IPython import embed

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--name_g2d', default='three_view_long_share_d0.75_256_s1_google', type=str, help='save model path')
parser.add_argument('--name_d2s', default='three_view_long_share_d0.75_256_s1_google', type=str, help='save model path')
parser.add_argument('--kq', default=20, type=int, help='kq')
parser.add_argument('--kd', default=30, type=int, help='kd')
parser.add_argument('--n_trunc', default=500, type=int, help='n_trunc')
parser.add_argument('--hiera', default=0, type=int, help='the hierarchy')
opt = parser.parse_args()


#######################################################################
# Evaluate
def evaluate(ql, gl, gl_d, gl_s):
    # good index
    query_index = np.argwhere(gl_s == ql)
    good_index = query_index

    # print(good_index)
    # print(index[0:10])
    junk_index = np.argwhere(gl_s == -1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros_like(index)
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


######################################################################

# load features
save_path_g2d =  os.path.join('../model',opt.name_g2d)
save_path_d2s =  os.path.join('../model',opt.name_d2s)
if opt.hiera == 0:
    with h5py.File(save_path_g2d +  '/pytorch.hdf5', mode='r') as h5:
        gallery_feature_d = h5.get("gallery_feature")[...]
        query_feature_g = h5.get("query_feature")[...]
    result_g2d_l = scipy.io.loadmat(save_path_g2d + '/pytorch_result.mat')
    gallery_label_d = result_g2d_l['gallery_label'][0]
    query_label_g = result_g2d_l['query_label'][0]
else:
    with h5py.File(save_path_g2d +  '/pytorch_part_h%s.hdf5'%(opt.hiera), mode='r') as h5:
        gallery_feature_d = h5.get("gallery_feature")[...]
        query_feature_g = h5.get("query_feature")[...]
    result_g2d_l = scipy.io.loadmat(save_path_g2d + '/pytorch_result_part_h%s.mat'%(opt.hiera))
    gallery_label_d = result_g2d_l['gallery_label'][0]
    query_label_g = result_g2d_l['query_label'][0]


with h5py.File(save_path_d2s +  '/pytorch_for_diffusion.hdf5', mode='r') as h5:
    gallery_feature_s = h5.get("gallery_feature")[...]
    query_feature_d = h5.get("query_feature")[...]
result_d2s_l = scipy.io.loadmat(save_path_d2s + '/pytorch_result_for_diffusion.mat')
gallery_label_s = result_d2s_l['gallery_label'][0]
query_label_d = result_d2s_l['query_label'][0]

print(query_feature_g.shape)
print(gallery_feature_d.shape)
print(query_feature_d.shape)
print(gallery_feature_s.shape)

print(query_label_g.shape)
print(gallery_label_d.shape)
print(query_label_d.shape)
print(gallery_label_s.shape)


CMC = np.zeros_like(gallery_label_s)
ap = 0.0
# print(query_label)


kq, kd = opt.kq, opt.kd
n_trunc = opt.n_trunc
gamma = 3

if not os.path.exists('./cache_%s_%s'%(opt.name_g2d, opt.name_d2s)):
    os.mkdir('./cache_%s_%s'%(opt.name_g2d, opt.name_d2s))

dfs = Diffusion(np.vstack((query_feature_d, gallery_feature_s)), './cache_%s_%s'%(opt.name_g2d, opt.name_d2s))
offline = dfs.get_offline_results(n_trunc, kd)

for i in tqdm(range(len(query_label_g)), desc='[online]'):
    # score = (gallery_feature_d @ query_feature_g[i]).reshape(-1, 30).max(-1)
    if opt.hiera == 0:
        score = gallery_feature_d @ query_feature_g[i]
    else:
        score = (gallery_feature_d @ query_feature_g[i]).reshape(-1, (opt.hiera + 1)).max(-1)

    index = np.argsort(-score)

    score = (score[index[:kq]] ** gamma) @ offline[index[:kq]]

    index = np.argsort(-score[-len(gallery_feature_s):])

    ap_tmp, CMC_tmp = evaluate(query_label_g[i], gallery_label_d, query_label_d, gallery_label_s)
    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    #print(i, CMC[9])

CMC = CMC / len(query_label_g)  # average CMC
print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label_s) * 0.01)] * 100, ap / len(query_label_g) * 100))




