from metrics import e_recall
import numpy as np
import faiss
import torch
from sklearn.preprocessing import normalize
from tqdm import tqdm
import copy
import os

import coordinate

def select(metricname, opt):
    #### Metrics based on euclidean distances
    if 'e_recall' in metricname:
        k = int(metricname.split('@')[-1])
        return e_recall.Metric(k)



class MetricComputer():
    def __init__(self, metric_names, opt):
        self.pars            = opt
        self.metric_names    = metric_names
        self.list_of_metrics = [select(metricname, opt) for metricname in metric_names]
        self.requires        = [metric.requires for metric in self.list_of_metrics]
        self.requires        = list(set([x for y in self.requires for x in y]))
        self.use_detection   = opt.use_detection

    def compute_standard(self, opt, model, dataloader_ground, dataloader_drone, evaltypes, device, **kwargs):
        evaltypes = copy.deepcopy(evaltypes)

        _ = model.eval()

        ###
        feature_colls  = {key:[] for key in evaltypes}

        ###
        with torch.no_grad():
            target_labels = []
            final_iter_ground = tqdm(dataloader_ground, desc='Ground Data...'.format(len(evaltypes)))
            final_iter_drone = tqdm(dataloader_drone, desc='Drone Data...'.format(len(evaltypes)))


            for idx,inp in enumerate(final_iter_ground):
                input_img,target = inp
                target_labels.extend(target.numpy().tolist())

                if opt.use_detection:
                    corner_output, corner_output_flip = coordinate.coord_cal_test(opt, path)

                out = model(input_img.cuda())
                #if isinstance(out, tuple): out, aux_f = out

                if isinstance(out, tuple): out, aux_f = out # out = embedding output

                ### Include embeddings of all output features
                for evaltype in evaltypes:
                    if isinstance(out, dict):
                        feature_colls[evaltype].extend(out[evaltype].cpu().detach().numpy().tolist())
                    else:
                        feature_colls[evaltype].extend(out.cpu().detach().numpy().tolist())


            target_labels = np.hstack(target_labels).reshape(-1,1)


        computed_metrics = {evaltype:{} for evaltype in evaltypes}
        extra_infos      = {evaltype:{} for evaltype in evaltypes}


        ###
        faiss.omp_set_num_threads(self.pars.kernels)
        # faiss.omp_set_num_threads(self.pars.kernels)
        res = None
        torch.cuda.empty_cache()
        if self.pars.evaluate_on_gpu:
            res = faiss.StandardGpuResources()


        import time
        for evaltype in evaltypes:
            features        = np.vstack(feature_colls[evaltype]).astype('float32')
            features_cosine = normalize(features, axis=1)

            start = time.time()


            """============ Compute Nearest Neighbours ==============="""
            if 'nearest_features' in self.requires:
                faiss_search_index  = faiss.IndexFlatL2(features.shape[-1])
                if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(features)

                max_kval            = np.max([int(x.split('@')[-1]) for x in self.metric_names if 'recall' in x])
                _, k_closest_points = faiss_search_index.search(features, int(max_kval+1))
                k_closest_classes   = target_labels.reshape(-1)[k_closest_points[:,1:]]

            if 'nearest_features_cosine' in self.requires:
                faiss_search_index  = faiss.IndexFlatIP(features_cosine.shape[-1])
                if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(normalize(features_cosine,axis=1))

                max_kval                   = np.max([int(x.split('@')[-1]) for x in self.metric_names if 'recall' in x])
                _, k_closest_points_cosine = faiss_search_index.search(normalize(features_cosine,axis=1), int(max_kval+1))
                k_closest_classes_cosine   = target_labels.reshape(-1)[k_closest_points_cosine[:,1:]]



            ###
            if self.pars.evaluate_on_gpu:
                features        = torch.from_numpy(features).to(self.pars.device)
                features_cosine = torch.from_numpy(features_cosine).to(self.pars.device)

            start = time.time()
            for metric in self.list_of_metrics:
                input_dict = {}
                if 'features' in metric.requires:         input_dict['features'] = features
                if 'target_labels' in metric.requires:    input_dict['target_labels'] = target_labels

                if 'kmeans' in metric.requires:           input_dict['centroids'] = centroids
                if 'kmeans_nearest' in metric.requires:   input_dict['computed_cluster_labels'] = computed_cluster_labels
                if 'nearest_features' in metric.requires: input_dict['k_closest_classes'] = k_closest_classes

                if 'features_cosine' in metric.requires:         input_dict['features_cosine'] = features_cosine

                if 'kmeans_cosine' in metric.requires:           input_dict['centroids_cosine'] = centroids_cosine
                if 'kmeans_nearest_cosine' in metric.requires:   input_dict['computed_cluster_labels_cosine'] = computed_cluster_labels_cosine
                if 'nearest_features_cosine' in metric.requires: input_dict['k_closest_classes_cosine'] = k_closest_classes_cosine

                computed_metrics[evaltype][metric.name] = metric(**input_dict)

            extra_infos[evaltype] = {'features':features, 'target_labels':target_labels,
                                     'image_paths': dataloader.dataset.image_paths,
                                     'query_image_paths':None, 'gallery_image_paths':None}

        torch.cuda.empty_cache()
        return computed_metrics, extra_infos
