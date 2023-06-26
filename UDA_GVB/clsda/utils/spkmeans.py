import torch
from torch.nn import functional as F
from math import ceil
import numpy as np


def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


def to_onehot(label, num_classes):
    identity = torch.eye(num_classes)
    onehot = torch.index_select(identity, 0, label)
    return onehot


class DIST(object):
    def __init__(self, dist_type='cos'):
        self.dist_type = dist_type

    def get_dist(self, pointA, pointB, cross=False):
        return getattr(self, self.dist_type)(
            pointA, pointB, cross)

    def cos(self, pointA, pointB, cross):
        pointA = F.normalize(pointA + 1e-8, dim=1)
        pointB = F.normalize(pointB + 1e-8, dim=1)
        a = torch.min(torch.sum(torch.abs(pointA), dim=1))
        b = torch.min(torch.sum(torch.abs(pointB), dim=1))
        if not cross:
            return 0.5 * (1.0 - torch.sum(pointA * pointB, dim=1))
        else:
            NA = pointA.size(0)
            NB = pointB.size(0)
            assert (pointA.size(1) == pointB.size(1))
            return 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))


class Clustering(object):
    def __init__(self, eps, cluster_num, max_len=5000, dist_type='cos', ):
        self.eps = eps
        self.Dist = DIST(dist_type)
        self.samples = {}
        self.path2label = {}
        self.center_change = None
        self.stop = False
        self.max_len = max_len
        self.cluster_num = cluster_num
        self.centers = [[] * self.cluster_num]



    def set_init_centers(self, init_centers):
        self.centers = init_centers
        self.init_centers = init_centers
        self.cluster_num = self.centers.size(0)

    def set_random_init_centers(self, features):
        select_ind = np.random.choice(features.shape[0], size=self.cluster_num, replace=False)
        return features[torch.from_numpy(select_ind), :].clone()

    def clustering_stop(self, centers, last_centers):
        if centers is None:
            return False
        dist = self.Dist.get_dist(centers, last_centers)
        dist = torch.mean(dist, dim=0)
        print('dist %.4f' % dist.item())
        return dist.item() < self.eps

    def assign_labels(self, feats, centers):
        dists = self.Dist.get_dist(feats, centers, cross=True)
        _, labels = torch.min(dists, dim=1)
        return dists, labels

    def feature_clustering(self, feature, init_centers=None):
        refs = torch.LongTensor(range(self.cluster_num)).unsqueeze(1)
        num_samples = feature.shape[0]
        num_split = ceil(1.0 * num_samples / self.max_len)
        #
        if init_centers is None:
            init_centers = self.set_random_init_centers(feature)
            print('center set')
        last_centers = init_centers
        centers = None

        if num_split > 0:
            while True:
                stop = self.clustering_stop(centers, last_centers)
                if stop:
                    start = 0
                    all_labels = torch.zeros(feature.shape[0])
                    for N in range(num_split):
                        cur_len = min(self.max_len, num_samples - start)
                        cur_feature = feature.narrow(0, start, cur_len)
                        dist2center, labels = self.assign_labels(cur_feature, last_centers)
                        all_labels[start:cur_len] = labels
                    # dist2center, labels = self.assign_labels(feature, centers)
                    return F.normalize(centers, dim=1), all_labels
                if centers is not None:
                    last_centers = centers

                centers = 0
                count = 0
                start = 0

                for N in range(num_split):
                    cur_len = min(self.max_len, num_samples - start)
                    cur_feature = feature.narrow(0, start, cur_len)
                    dist2center, labels = self.assign_labels(cur_feature, last_centers)
                    labels_onehot = to_onehot(labels, self.cluster_num)
                    count += torch.sum(labels_onehot, dim=0)
                    labels = labels.unsqueeze(0)
                    mask = (labels == refs).unsqueeze(2).type(torch.float)
                    reshaped_feature = cur_feature.unsqueeze(0)
                    # update centers
                    centers += torch.sum(reshaped_feature * mask, dim=1)
                    start += cur_len

                mask = (count.unsqueeze(1) > 0).type(torch.float)
                centers = mask * centers + (1 - mask) * last_centers
        else:
            return last_centers, torch.zeros(feature.shape[0])