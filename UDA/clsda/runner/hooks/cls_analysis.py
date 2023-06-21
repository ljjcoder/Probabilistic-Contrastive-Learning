# 
# ----------------------------------------------
import torch
import torch.nn.functional as F
from mmcv.runner.hooks import Hook
from clsda.utils.metrics import runningMetric
from clsda.utils import get_root_logger, get_root_writer
import os
import pickle

class CLSAnalysis(Hook):
    def __init__(self, dataset_name, pred_key):
        self.dataset_name = dataset_name
        self.img_list = []
        self.gt_list = []
        self.pred_list = []
        self.max_prob_list = []
        self.ent_list = []
        self.pred_key = pred_key
        self.correct_list = []

    def after_val_iter(self, runner):
        batch_output = runner.batch_output
        dataset_name = batch_output['dataset_name']
        if dataset_name == self.dataset_name:
            img_metas = batch_output['img_metas']
            output_prob = torch.softmax(batch_output[self.pred_key], dim=1)
            output_ent = - torch.sum(output_prob * torch.log(output_prob + 1e-5), dim=1)
            for i in range(len(img_metas.data)):
                tmp_img_name = img_metas.data[i][0]['ori_filename']  # TODO:这里为什么还要取0
                self.img_list.append(tmp_img_name)
            prob, pred = torch.max(output_prob, dim=1)
            gt = batch_output['gt']
            correct = gt.eq(pred).cpu().numpy().tolist()
            pred = pred.cpu().numpy().tolist()
            prob = prob.cpu().numpy().tolist()
            self.gt_list.extend(gt.cpu().tolist())
            self.pred_list.extend(pred)
            self.max_prob_list.extend(prob)
            self.correct_list.extend(correct)
            self.ent_list.extend(output_ent.cpu().numpy().tolist())

    def after_val_epoch(self, runner):
        logger = get_root_logger()
        # class_num = max(self.pred_list) + 1
        # class_sample_list = []  # 每个类别的真实index
        # class_pred_list = []  # 每个类别的预测index
        # class_correct_pred_list = []  # 每个类别预测正确的样本index
        # class_self_wrong_pred_list = []  # 每个类别预测错误的样本index
        # class_assign_wrong_pred_list = []  # 每个类别从其他类别错分的样本index
        # class_max_prob_list_by_gt = []  # 按真实标注的类别存储该类别的最大概率
        # class_ent_list_by_gt = []  # 按真实标注的类别存储该类别的熵
        # class_max_prob_list_by_pred = []  # 按预测结果的类别存储该类别的最大概率
        # class_ent_list_by_pred = []  # 按预测结果的类别存储该类别的熵
        # for i in range(class_num):
        #     class_sample_list.append([])
        #     class_pred_list.append([])
        #     class_correct_pred_list.append([])
        #     class_self_wrong_pred_list.append([])
        #     class_assign_wrong_pred_list.append([])
        #     class_max_prob_list_by_gt.append([])
        #     class_ent_list_by_gt.append([])
        #     class_max_prob_list_by_pred.append([])
        #     class_ent_list_by_pred.append([])
        # # 获取每个类别的概率
        # for tmp_ind, tmp_gt, tmp_pred, tmp_prob, tmp_ent in enumerate(
        #         zip(self.gt_list, self.pred_list, self.max_prob_list, self.ent_list)):
        #     class_sample_list[tmp_gt] = tmp_ind
        #     class_pred_list[tmp_pred] = tmp_ind
        #     class_max_prob_list_by_gt[tmp_gt] = tmp_prob
        #     class_max_prob_list_by_pred[tmp_pred] = tmp_prob
        #     class_ent_list_by_gt[tmp_gt] = tmp_ent
        #     class_ent_list_by_pred[tmp_pred] = tmp_pred
        #     if tmp_pred == tmp_gt:
        #         class_correct_pred_list[tmp_gt] = tmp_ind
        #     else:
        #         class_self_wrong_pred_list[tmp_gt] = tmp_ind
        #         class_assign_wrong_pred_list[tmp_pred] = tmp_ind
        #
        res = {
            'gt_list':self.gt_list,
            'pred_list':self.pred_list,
            'max_prob_list':self.max_prob_list,
            'ent_list':self.ent_list,
        }
        save_path = os.path.join(runner.logdir,'res.pkl')
        with open(save_path,'wb') as f:
            pickle.dump(res,f)
        # class_correct_ratio = []
        # class_gt_count = []
        # # 各类别的正确率
        # for i in range(class_num):
        #     class_correct_ratio.append(len(class_correct_pred_list[i]) / len(class_sample_list))
        #     class_gt_count.append(len(class_sample_list))

