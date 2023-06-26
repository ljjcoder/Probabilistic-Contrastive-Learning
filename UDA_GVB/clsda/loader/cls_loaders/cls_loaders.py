import numpy as np
import os
import os.path
from PIL import Image
import random
import torch
from .builder import CLS_DATASETS
from mmcls.datasets import BaseDataset
from .pipelines import Compose
import copy
import pickle


def pil_loader(path):
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert("RGB")
    return img


def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
    return image_index, label_list


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list


@CLS_DATASETS.register_module(name='ssda_cls_dataset')
class SSDA_CLS_Datasets(BaseDataset):
    def __init__(self, root, name, split, shot='', pipeline=None, min_len=0):
        name_split = name.split('_')
        task = name_split[0]
        dataset = name_split[1]
        img_root = os.path.join(root, task)
        if shot != '':
            shot = '_' + str(shot)
        image_list = os.path.join(root, 'txt', task, split + '_images_' + dataset + shot + '.txt')
        self.min_len = min_len
        #
        super(SSDA_CLS_Datasets, self).__init__(data_prefix=img_root, pipeline=pipeline, ann_file=image_list)
        #
        self.name = name
        self.split = split

    def load_annotations(self):
        imgs, labels = make_dataset_fromlist(self.ann_file)
        # 设置最大类别
        self.n_classes = max(labels) + 1
        #
        repeat_num = int(float(self.min_len) / len(imgs)) + 1
        imgs = imgs * repeat_num
        labels = labels * repeat_num
        #
        data_infos = []
        for img, label in zip(imgs, labels):
            info = {'img_prefix': self.data_prefix, 'img_info': {'filename': img}, 'gt_label': label}
            data_infos.append(info)
        return data_infos

    def get_classes(self, classes=None):
        class_names = []
        with open(self.ann_file, 'rb') as f:
            for line in f.readlines():
                line = line.decode()
                tmp_res = line.strip().split(' ')
                tmp_path = tmp_res[0]
                # print(tmp_res[0],tmp_res[1])
                tmp_ind = int(tmp_res[1])
                tmp_name = tmp_path.split('/')[1]
                if tmp_ind == len(class_names):
                    class_names.append(tmp_name)
        return class_names


@CLS_DATASETS.register_module(name='ssda_cls_double_dataset')
class SSDA_CLS_Double_Datasets(SSDA_CLS_Datasets):
    def __init__(self, root, name, split, shot='', pipeline=None, pipeline2=None, min_len=0):
        super(SSDA_CLS_Double_Datasets, self).__init__(root, name, split, shot, pipeline, min_len)
        self.pipeline_2 = Compose(pipeline2)

    def __getitem__(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        aug_data_1 = self.pipeline(results)
        aug_data_2 = self.pipeline_2(results)
        return (aug_data_1, aug_data_2)
    

@CLS_DATASETS.register_module(name='ssda_cls_triple_dataset')
class SSDA_CLS_Triple_Datasets(SSDA_CLS_Datasets):
    def __init__(self, root, name, split, shot='', pipeline=None, pipeline2=None, min_len=0):
        super(SSDA_CLS_Triple_Datasets, self).__init__(root, name, split, shot, pipeline, min_len)
        self.pipeline_2 = Compose(pipeline2)

    def __getitem__(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        aug_data_1 = self.pipeline(results)
        aug_data_2 = self.pipeline_2(results)
        aug_data_3 = self.pipeline_2(results)
        return (aug_data_1, aug_data_2,aug_data_3)

