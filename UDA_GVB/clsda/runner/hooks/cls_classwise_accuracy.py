# 
# ----------------------------------------------
import torch
import torch.nn.functional as F
from mmcv.runner.hooks import Hook
from clsda.utils.metrics import runningMetric
from clsda.utils import get_root_logger, get_root_writer
from mmcv.runner import get_dist_info
import pickle


class ClsClassWiseAccuracy(Hook):
    def __init__(self, runner, dataset_name, major_comparison=False, pred_key='pred'):
        rank, _ = get_dist_info()
        self.local_rank = rank
        self.dataset_name = dataset_name
        if rank == 0:
            self.running_metrics = runningMetric()  #
            log_interval = max(len(runner.test_loaders[dataset_name]) - 1, 1)
            self.running_metrics.add_metrics('{}_cls'.format(pred_key), group_name='val_loss', metric_type='avgmeter',
                                             log_interval=log_interval)
        self.correct_count = 0.0
        self.total_count = 0.0
        self.major_comparison = major_comparison
        self.best_acc = 0.0
        self.current_acc = 0.0
        self.pred_key = pred_key

    def after_val_iter(self, runner):
        batch_output = runner.batch_output
        dataset_name = batch_output['dataset_name']
        if dataset_name == self.dataset_name:
            gt = batch_output['gt']
            pred = batch_output[self.pred_key]
            pred_max = torch.argmax(pred, dim=1)
            self.correct_count += gt.eq(pred_max).cpu().sum()
            self.total_count += gt.shape[0]
            #
            loss = F.cross_entropy(pred, gt)
            if self.local_rank == 0:
                batch_metrics = {'val_loss': {'{}_cls'.format(self.pred_key): loss.item()}}
                self.running_metrics.update_metrics(batch_metrics)

    def after_val_epoch(self, runner):
        #
        local_correct_tensor = torch.tensor([self.correct_count, ],
                                            device='cuda:{}'.format(self.local_rank))
        local_total_tensor = torch.tensor([self.total_count, ], device='cuda:{}'.format(self.local_rank))
        torch.distributed.reduce(local_correct_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)
        # torch.distributed.barrier()
        torch.distributed.reduce(local_total_tensor,dst=0, op=torch.distributed.ReduceOp.SUM)
        # torch.distributed.barrier()
        #
        self.correct_count = local_correct_tensor.item()
        self.total_count = local_total_tensor.item()
        if self.local_rank == 0:
            acc = self.correct_count / self.total_count
            self.current_acc = acc
            if acc > self.best_acc:
                self.best_acc = acc
                if self.major_comparison:
                    runner.save_flag = True
            #
            logger = get_root_logger()
            writer = get_root_writer()
            # 这里输入的是训练集的迭代次数,是为了tensorboard记录方便
            self.running_metrics.log_metrics(runner.iteration, force_log=True)
            #

            writer.add_scalar('{}_acc_{}'.format(self.pred_key, self.dataset_name), acc,
                              global_step=runner.iteration)
            logger.info('Iteration {}: {} {} acc {}'.format(runner.iteration, self.pred_key, self.dataset_name, acc))

            logger.info(
                'total img of {} is {}, right {}'.format(self.dataset_name, self.total_count, self.correct_count))
        # 这里的重置不能少！！！！
        self.correct_count = 0.0
        self.total_count = 0.0
