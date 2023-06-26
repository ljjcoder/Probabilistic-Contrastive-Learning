# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
from torchvision.utils import make_grid
import torch
# from clsda.loader.seg_loaders.data_utils import index2rgb
import random
from .logger import get_root_logger
from .writer import get_root_writer


class segScore(object):
    def __init__(self, n_classes, name=None):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.name = name
        self.score = None
        self.class_iou = None
        self.main_metrics = None

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, labels):
        label_trues, label_preds = labels
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        # for dataset synthia
        # iu_for_13 = iu.copy()
        # iu_for_16 = iu.copy()
        # exclude_ind_13 = [3, 4, 5, 9, 14, 16]
        # exclude_ind_16 = [9, 14, 16]
        # iu_for_13[exclude_ind_13] = np.nan
        # iu_for_16[exclude_ind_16] = np.nan
        # mean_iu_13 = np.nanmean(iu_for_13)
        # mean_iu_16 = np.nanmean(iu_for_16)
        # print('type of iu is {}'.format(iu.shape))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        self.main_metrics = mean_iu

        return (
            {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                # "Mean IoU(16)": mean_iu_16,
                # "Mean IoU(13)": mean_iu_13,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def log_to_writer(self, writer, iteration, name_prefix):
        score, class_iou = self.get_scores()
        log_str = 'Dataset: {}, {}\n'.format(name_prefix, str(self.name))
        for k, v in score.items():
            log_str += '{} : \t{}\n'.format(k, v)
            writer.add_scalar('{}/{}'.format(name_prefix, k + '_' + str(self.name)), v, iteration)
        for k, v in class_iou.items():
            log_str += "cls_{} : \t{}\n".format(k, v)
            writer.add_scalar('{}/{}'.format(name_prefix, 'cls_' + str(k) + str(self.name)), v, iteration)
        return log_str

    def metrics_for_comparision(self):
        if self.main_metrics is None:
            self.get_scores()
        return self.main_metrics


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name=None):
        self.reset()
        self.name = name

    def reset(self):
        self.val = 0.0
        # self._avg = 0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n

    @property
    def avg(self):
        # print('sum {}, count {}, mean {}'.format(self.sum, self.count, self.sum / self.count))
        # print('self name {}'.format(self.name))
        return self.sum / self.count

    # @avg.setter
    # def avg(self, val):
    #     self._avg = val  # should not be self.avg Or you will get: RecursionError: maximum recursion depth exceeded while calling a Python object

    def log_to_writer(self, writer, iteration, name_prefix):
        # print('avg type {}'.format(type(self.avg)))
        # if isinstance(self.avg, np.ndarray):
        #     print('shape is {}'.format(self.avg.shape))
        # print('self.name {}'.format(self.name))
        writer.add_scalar('{}/{}'.format(name_prefix, self.name), self.avg, iteration)
        return self.name + ':{:.4f}\t'.format(self.avg)


class lrRecoder(object):
    def __init__(self, scheduler_dict, name=None):
        self.scheduler_dict = scheduler_dict

    def update(self, val):
        pass

    def log_to_writer(self, writer, iteration, name_prefix):
        log_str = ''
        for name in self.scheduler_dict:
            temp_lr = self.scheduler_dict[name].get_lr()[0]
            writer.add_scalar('{}/{}'.format(name_prefix, name), temp_lr, iteration)
            log_str += '{}_lr: {:.2e}\t'.format(name, temp_lr)
        return log_str

    def reset(self):
        self.log_str = ''


class imageList(object):
    def __init__(self, max_num, name=None):
        self.name = name
        self.max_num = max_num
        self.img_list = []  # 保存图像数据，或者图像数据的序列（每一次的序列长度要一致）

    def __len__(self):
        return len(self.img_list)

    def update(self, img):
        """
        :param img: img tensor or sequence of img tensor
        :return:
        """
        if len(self.img_list) == self.max_num:
            self.img_list.pop(0)
        self.img_list.append(img)

    def log_to_writer(self, writer, iteration, name_prefix):
        if self.max_num > 0:
            processed_img_list = []
            for item in self.img_list:
                if isinstance(item, (tuple, list)):
                    temp_img_list = []
                    for img in item:
                        temp_img_list.append(self.process_single_img_data(img))
                    processed_img_list.append(temp_img_list)
                else:
                    processed_img_list.append(self.process_single_img_data(item))
            single_len = len(processed_img_list[0])
            # 重新排列，构造img_grid
            all_imgs = []
            for i in range(single_len):
                all_imgs.extend([item[i] for item in processed_img_list])
            # make_grid接口请参考 https://pytorch.org/docs/stable/torchvision/transforms.html
            imgs_grid = make_grid(all_imgs, nrow=len(processed_img_list), padding=5)
            writer.add_image('{}/{}'.format(name_prefix, self.name), imgs_grid, iteration)
            # add_image的接口请参考 https://tensorboardx.readthedocs.io/en/latest/tensorboard.html
        return ''

    def process_single_img_data(self, img_data):
        raise NotImplementedError

    def reset(self):
        self.img_list = []


class segimgList(imageList):
    def __init__(self, dataloader, max_num, name=None):
        super(segimgList, self).__init__(max_num=max_num, name=name)
        self.dataloader = dataloader

    def process_single_img_data(self, img_data):
        if isinstance(img_data, torch.Tensor):
            img_data = img_data.numpy()
        if img_data.ndim == 4:  # 包含了batch
            if img_data.shape[1] != 1:  # 没有经过arguemax
                img_data = np.argmax(img_data, axis=1)
            else:
                img_data = img_data[0]  # 经过了argmax, 只取第一张图像
        elif img_data.ndim == 3:  # 假设经过了argmax，而不是压缩了batch_dim导致的
            img_data = img_data[0]
        else:
            raise RuntimeError('ndim of img_data should be 3 or 4')
        img_tensor = torch.from_numpy(index2rgb(self.dataloader.dataset.decode_label(img_data)).squeeze(0))
        return img_tensor.permute(2, 0, 1)


class boundaryimgList(imageList):
    def __init__(self, max_num, name=None):
        super(boundaryimgList, self).__init__(max_num=max_num, name=name)

    def process_single_img_data(self, img_data):
        if isinstance(img_data, np.ndarray):
            img_data = torch.from_numpy(img_data)
        if img_data.ndimension() == 4:
            assert img_data.shape[1] == 1, 'boundary img should be 1 dim'
            img_data = img_data.squeeze(1)
        assert torch.max(img_data) <= 1.0 and torch.min(img_data) >= 0.0, 'img_data should be scaled to 0-1'
        img_data = (img_data * 255).to(torch.uint8)
        return img_data


class fixedSampleImgList(segimgList):
    def __init__(self, dataloader, max_num, name=None, sample_indices=None):
        super(fixedSampleImgList, self).__init__(dataloader=dataloader, max_num=max_num, name=name)
        if sample_indices is None:
            self.show_indices = random.sample(range(len(self.dataloader)), max_num)
        else:
            self.show_indices = sample_indices
        self.start_index = 0

    def update(self, img_data):
        if len(self.img_list) == self.max_num:
            return
        elif self.max_num > 0:
            if isinstance(img_data, (tuple, list)):
                batch_num = img_data[0].shape[0]
            else:
                batch_num = img_data.shape[0]

            current_indices_list = list(range(self.start_index, self.start_index + batch_num))
            saved_indices = sorted(list(set(self.show_indices) & set(current_indices_list)))

            if len(saved_indices) > 0:
                saved_indices = [i - self.start_index for i in saved_indices]
                for i in saved_indices:
                    self.img_list.append(img_data[i:i + 1])

            self.start_index += batch_num
            if self.start_index >= len(self.dataloader):
                self.start_index = 0


class fixedSampleSegImgList(fixedSampleImgList, segimgList):
    def __init__(self, dataloader, max_num, name=None, sample_indices=None):
        super().__init__(dataloader=dataloader, max_num=max_num, name=name, sample_indices=sample_indices)
        # super(fixedSampleSegImgList, self).__init__(dataloader=dataloader, max_num=max_num, name=name)


def _get_metric_instance(name):
    try:
        return {
            "segscore": segScore,
            'avgmeter': averageMeter,
            'imglist': imageList,
            'lrrecoder': lrRecoder,
            'segimglist': segimgList,
            'fixedsamplesegimglist': fixedSampleImgList,
            'boundaryimglist': boundaryimgList,
        }[name]
    except:
        raise RuntimeError("metric type {} not available".format(name))


class runningMetric(object):
    def __init__(self, ):
        self.metrics = {}
        self.log_intervals = {}
        self.log_str_flag = {}
        self.logger = get_root_logger()
        self.writer = get_root_writer()

    def add_metrics(self, metric_name, group_name, metric_type, log_interval=None, init_param_list=(),
                    init_param_dict=None, log_str_flag=True):
        if group_name not in self.metrics:
            assert log_interval is not None, 'you should specify log interval when first add metric'
            self.metrics[group_name] = {}
            self.log_intervals[group_name] = log_interval
            self.log_str_flag[group_name] = log_str_flag
        else:
            assert log_interval is None or log_interval == self.log_intervals[
                group_name], 'log interval {} is not consistent with {}'.format(log_interval,
                                                                                self.log_intervals[group_name])
            assert log_str_flag is None or log_str_flag == self.log_str_flag[group_name], 'log str flag not match'
        if isinstance(metric_name, (tuple, list, str)):
            metric_name = (metric_name,) if isinstance(metric_name, str) else metric_name
            for name in metric_name:
                temp_param_dict = {'name': name}
                if init_param_dict is not None:
                    temp_param_dict.update(init_param_dict)
                self.metrics[group_name][name] = _get_metric_instance(metric_type)(*init_param_list,
                                                                                   **temp_param_dict)
        else:
            raise RuntimeError('log name should be str or tuple list of str')

    def update_metrics(self, batch_metric):
        for group_name in batch_metric:
            if group_name in self.metrics.keys():  # 增加了group name是否在本running_metrics的判断
                temp_group = batch_metric[group_name]
                for name in temp_group:
                    self.metrics[group_name][name].update(temp_group[name])

    def log_metrics(self, iteration, force_log=False):
        for group_name in self.metrics:
            # print('group name {}, log interval {}'.format(group_name,self.log_intervals[group_name]))
            if (iteration % self.log_intervals[group_name] == 0 and iteration > 0) or force_log:
                log_str = 'iter:{}---'.format(iteration)
                for name in self.metrics[group_name]:
                    temp_log_str = self.metrics[group_name][name].log_to_writer(self.writer, iteration, group_name)
                    if self.log_str_flag[group_name]:
                        log_str += temp_log_str
                    # 重置
                    self.metrics[group_name][name].reset()
                if self.log_str_flag[group_name]:
                    self.logger.info(log_str)
