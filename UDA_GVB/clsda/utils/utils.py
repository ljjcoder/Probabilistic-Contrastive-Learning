"""
Misc Utility functions
"""
import datetime
import functools
import math
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.cluster import MiniBatchKMeans
from torch.nn import DataParallel
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from mmcv.runner import get_dist_info
from collections.abc import Sequence


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def calc_mean_std(feat, eps=1e-5, detach_mean_std=True):
    # eps is a small value added to the variance to avoid divide-by-zero.
    # print('detach flat {}'.format(detach_mean_std))
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    # var will cause inf in amp mode, use std instead
    # feat_var = feat.view(N, C, -1).var(dim=2) + eps
    # feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_std = torch.std(feat.view(N, C, -1), dim=2).view(N, C, 1, 1) + eps
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    if detach_mean_std:
        # print('detach mean std back')
        return feat_mean.detach(), feat_std.detach()
    else:
        # print('with mean std back')
        return feat_mean, feat_std


def show_imgs_in_grid(dir_list, res_save_path, img_size=(100, 200), row_num=5, img_gap=10, img_name_list=None):
    if img_name_list == None:
        img_name_list = os.listdir(dir_list[0])
    img_num = len(img_name_list)
    column_num = len(dir_list)  # 最后一列表示图像的名字，方便查找
    # 根据行列数，以及图像大小，gap大小确定整个图像的大小
    total_height = img_size[0] * row_num + img_gap * (row_num - 1)
    total_width = img_size[1] * column_num + img_gap * (column_num - 1)
    total_img_array = np.zeros((total_height, total_width, 4), dtype=np.uint8)
    transparency_mask = np.ones((*img_size, 1), dtype=np.uint8) * 255
    # 用作
    temp_name_list = []
    for img_ind in range(img_num):
        if img_ind % row_num == 0:
            total_img_array = np.zeros((total_height, total_width, 4), dtype=np.uint8)
            temp_name_list = []
        start_height = (img_ind % row_num) * img_size[0] + img_gap * (img_ind % row_num)
        for img_path_ind in range(column_num):
            start_width = img_path_ind * img_size[1] + img_gap * img_path_ind
            temp_img_path = os.path.join(dir_list[img_path_ind], img_name_list[img_ind])
            if os.path.isfile(temp_img_path):
                temp_img_path = temp_img_path
            else:
                temp_img_path = temp_img_path[0:-10] + '.jpg'
                assert os.path.isfile(temp_img_path), 'wrong img path {}'.format(temp_img_path)
            # print('start height {}, start width {}'.format(start_height, start_width))
            temp_img = Image.open(temp_img_path)
            temp_resized_img = temp_img.resize((img_size[1], img_size[0]), Image.ANTIALIAS)
            temp_img_array = np.asarray(temp_resized_img)
            if temp_img_array.ndim == 3:
                if temp_img_array.shape[2] == 3:
                    temp_img_array = np.concatenate((temp_img_array, transparency_mask), axis=2)
            elif temp_img_array.ndim == 2:
                temp_img = Image.open(temp_img_path)
                rgbimg = Image.new('RGBA', temp_img.size)
                rgbimg.paste(temp_img)
                temp_resized_img = rgbimg.resize((img_size[1], img_size[0]), Image.ANTIALIAS)
                temp_img_array = np.asarray(temp_resized_img)

            total_img_array[start_height:start_height + img_size[0],
            start_width:start_width + img_size[1]] = temp_img_array
        temp_name_list.append(img_name_list[img_ind])
        if (img_ind + 1) % row_num == 0:
            total_img = Image.fromarray(total_img_array, mode='RGBA')

            temp_save_path = os.path.join(res_save_path, 'num_{}_res.PNG'.format(img_ind // row_num))
            total_img.save(temp_save_path)
            print('Here is {}'.format(img_ind // row_num))
            for i in range(row_num):
                print(temp_name_list[i])

            #
            # new_img = Image.open(temp_save_path)
            # text_overlay = Image.new("RGBA",new_img.size,(255,255,255,0))
            # image_draw = ImageDraw.Draw(text_overlay)
            #
            # assert len(temp_name_list) == row_num, 'wrong match of img name list and row number'
            # start_width = column_num * img_size[1] + column_num * img_gap
            # for i in range(row_num):
            #     start_height = row_num * img_size[0] + row_num * img_gap
            #     print(temp_name_list[i])
            #     image_draw.text((start_width,start_height), temp_name_list[i], fill=(76, 234, 124, 180))
            # image_with_text = Image.alpha_composite(new_img,text_overlay)
            # image_with_text.save(temp_save_path)

            # font = cv2.FONT_HERSHEY_SIMPLEX
            #
            # for ind in range(row_num):
            #     start_height = row_num * img_size[0] + row_num * img_gap
            #     new_img = cv2.putText(img=total_img_array, text=temp_name_list[ind], org=(start_width, start_height),
            #                           fontFace=font, fontScale=5, color=(0, 255, 0))
            # temp_save_path = os.path.join(res_save_path, 'num_{}_res.PNG'.format(img_ind // row_num))
            # cv2.imwrite(temp_save_path, total_img_array)


def cal_feat_distance(feat_1, feat_2, metric_type='cos_similarity', alpha=1.0):
    """
    计算特征之间的相似度，值越大，相似度越高
    feat_1和feat_2位于不同的设备时，不会报错，但是计算结果会有问题
    :param feat_1: MxD
    :param feat_2: NxD
    :return: MxN
    """
    assert metric_type in ['student_t', 'inner_product', 'cos_similarity'], "wrong metric type {}".format(
        metric_type)
    # print('metric type {}'.format(metric_type))
    if metric_type == 'student_t':
        # score表示 NxM的距离矩阵
        score = (1 + (feat_1.unsqueeze(1) - feat_2.unsqueeze(0)).pow(2).sum(2) / alpha).pow(-(alpha + 1) / 2.0)
    elif metric_type == 'inner_product':
        score = feat_1.mm(feat_2.t())
    else:
        feat_1_norm = torch.sqrt(torch.sum(feat_1 * feat_1, dim=1, keepdim=True) + 1e-8)
        feat_2_norm = torch.sqrt(torch.sum(feat_2 * feat_2, dim=1) + 1e-8).unsqueeze(1).t()
        score = feat_1.mm(feat_2.t()) / ((feat_1_norm.mm(feat_2_norm)) + 1e-8)
    return score


def deal_with_val_interval(val_interval, max_iters, trained_iteration=0):
    fine_grained_val_checkpoint = []

    def reduce_trained_iteration(val_checkpoint):
        new_val_checkpoint = []
        start_flag = False
        for tmp_checkpoint in val_checkpoint:
            if start_flag:
                new_val_checkpoint.append(tmp_checkpoint)
            else:
                if tmp_checkpoint >= trained_iteration:
                    if tmp_checkpoint > trained_iteration:
                        new_val_checkpoint.append(tmp_checkpoint)
                    start_flag = True
        return new_val_checkpoint

    if isinstance(val_interval, (int, float)):
        val_times = int(max_iters / val_interval)
        for i in range(1, val_times + 1):
            fine_grained_val_checkpoint.append(i * int(val_interval))
        if fine_grained_val_checkpoint[-1] != max_iters:
            fine_grained_val_checkpoint.append(max_iters)
        return reduce_trained_iteration(fine_grained_val_checkpoint)
    elif isinstance(val_interval, dict):
        current_checkpoint = 0
        milestone_list = sorted(val_interval.keys())
        assert milestone_list[0] > 0 and milestone_list[-1] <= max_iters, 'check val interval keys'
        # 如果最后一个不是max_iter，则按最后的interval计算
        if milestone_list[-1] != max_iters:
            val_interval[max_iters] = val_interval[milestone_list[-1]]
            milestone_list.append(max_iters)
        last_milestone = 0
        for milestone in milestone_list:
            tmp_interval = val_interval[milestone]
            tmp_val_times = int((milestone - last_milestone) / tmp_interval)
            for i in range(tmp_val_times):
                fine_grained_val_checkpoint.append(current_checkpoint + int(tmp_interval))
                current_checkpoint += int(tmp_interval)
            if fine_grained_val_checkpoint[-1] != milestone:
                fine_grained_val_checkpoint.append(milestone)
                current_checkpoint = milestone
            last_milestone = current_checkpoint
        return reduce_trained_iteration(fine_grained_val_checkpoint)
    else:
        raise RuntimeError('only single value or dict is acceptable for val interval')


def move_models_to_gpu(model, device, max_card=0, find_unused_parameters=False):
    #
    rank, world_size = get_dist_info()
    #
    tmp_rank = rank * max_card + device
    model = model.to('cuda:{}'.format(tmp_rank))
    model = DistributedDataParallel(model, device_ids=[tmp_rank],
                                    output_device=tmp_rank,
                                    find_unused_parameters=find_unused_parameters)
    return model


def move_data_to_gpu(cpu_data, gpu_id):
    relocated_data = cpu_data
    if isinstance(cpu_data, Sequence):
        for ind, item in enumerate(cpu_data):
            relocated_data[ind] = move_data_to_gpu(item, gpu_id)
    elif isinstance(cpu_data, dict):
        for key, item in cpu_data.items():
            relocated_data[key] = move_data_to_gpu(item, gpu_id)
    elif isinstance(cpu_data, torch.Tensor):
        if cpu_data.device == torch.device('cpu'):
            return cpu_data.to(gpu_id)
    return relocated_data


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def generate_different_class_index(label, num_samples=1):
    """
    输入标签，输出index，index和对应位置的原始label不一样，实现采用torch.multinomial
    :param label: Bx1
    :return:
    """
    num = label.shape[0]
    label_row = label.view(num, 1).expand(num, num)
    label_column = label.view(1, num).expand(num, num)
    diff_mat = (label_row != label_column) + 1e-8
    # print('label {}, mat {}'.format(label, diff_mat))
    idx = torch.multinomial(diff_mat,num_samples=num_samples).squeeze()
    return idx
