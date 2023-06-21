# 
# ----------------------------------------------
import torch
import torch.nn.functional as F
from mmcv.runner.hooks import Hook
from clsda.utils.metrics import runningMetric
import time
import os
import glob
from clsda.utils import get_root_writer, get_root_logger


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = torch.tensor(0.0).to('cuda:0')
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.detach().norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm + 1e-10).item()

    norm = (clip_norm / max(totalnorm, clip_norm))

    for p_name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            param.grad.mul_(norm)


class LossMetrics(Hook):
    def __init__(self, log_names, group_name, log_interval):
        self.log_interval = log_interval
        self.running_metrics = runningMetric()
        self.running_metrics.add_metrics(log_names, group_name=group_name, metric_type='avgmeter',
                                         log_interval=log_interval)

    def after_train_iter(self, runner):
        batch_output = runner.train_batch_output
        self.running_metrics.update_metrics(batch_output)
        self.running_metrics.log_metrics(runner.iteration + 1)


class LrRecorder(Hook):
    def __init__(self, log_interval):
        self.log_interval = log_interval
        self.writer = get_root_writer()
        self.logger = get_root_logger()

    def after_train_iter(self, runner):
        if (runner.iteration + 1) % self.log_interval == 0:
            log_str = 'iter:{}---'.format(runner.iteration + 1)
            for name in runner.scheduler_dict:
                temp_lr = runner.scheduler_dict[name].get_last_lr()[0]
                # temp_lr = runner.optimizer_dict[name].param_groups[0]['lr']
                self.writer.add_scalar('{}/{}'.format('lr', name), temp_lr, (runner.iteration + 1))
                log_str += '{}_lr: {:.2e}\t'.format(name, temp_lr)
            self.logger.info(log_str)


class BackwardUpdate(Hook):
    def __init__(self, update_iter=1):
        self.update_iter = 1

    def before_train_iter(self, runner):
        # optimizer zero grad
        if runner.iteration % self.update_iter == 0:
            for name in runner.optimizer_dict.keys():
                runner.optimizer_dict[name].zero_grad()

    def after_train_iter(self, runner):
        # optimizer step
        if (runner.iteration + 1) % self.update_iter == 0:
            for name in runner.optimizer_dict.keys():
                # print('{} step'.format(name))
                runner.optimizer_dict[name].step()


class BackwardUpdatewithAMP(Hook):
    def __init__(self, update_iter=1):
        self.update_iter = update_iter

    def before_train_iter(self, runner):
        # optimizer zero grad
        if runner.iteration % self.update_iter == 0:
            for name in runner.optimizer_dict.keys():
                runner.optimizer_dict[name].zero_grad()

    def after_train_iter(self, runner):
        # optimizer step
        if (runner.iteration + 1) % self.update_iter == 0:
            for name in runner.optimizer_dict.keys():
                runner.scaler.step(runner.optimizer_dict[name])
        # scaler update
        runner.scaler.update()


class SchedulerStep(Hook):
    def __init__(self, update_iter=1):
        self.update_iter = update_iter

    def after_train_iter(self, runner):
        # scheduler_step
        if (runner.iteration + 1) % self.update_iter == 0:
            for name in runner.scheduler_dict.keys():
                runner.scheduler_dict[name].step()


class TrainTimeRecoder(Hook):
    def __init__(self, log_interval):
        self.start_time = time.time()
        self.forward_start_time = time.time()
        self.running_metrics = runningMetric()  #
        self.running_metrics.add_metrics('speed', group_name='other', metric_type='avgmeter',
                                         log_interval=log_interval)
        self.running_metrics.add_metrics('forward_speed', group_name='other', metric_type='avgmeter',
                                         log_interval=log_interval)

    def before_train_iter(self, runner):
        self.forward_start_time = time.time()

    def after_train_iter(self, runner):
        self.running_metrics.update_metrics({'other': {'speed': time.time() - self.start_time}})
        self.running_metrics.update_metrics({'other': {'forward_speed': time.time() - self.forward_start_time}})
        self.start_time = time.time()
        self.running_metrics.log_metrics(runner.iteration + 1)


class GradientClipper(Hook):
    def __init__(self, max_num=None):
        self.max_num = max_num

    def after_train_iter(self, runner):
        if runner.iteration % runner.update_iter == 0:
            for name in runner.model_dict.keys():
                clip_gradient(runner.model_dict[name], self.max_num)


class SaveModel(Hook):
    def __init__(self, max_save_num=0, save_interval=100000000, max_iters=1000000000):
        self.max_save_num = max_save_num
        self.save_interval = save_interval
        self.max_iters = max_iters

    def after_train_iter(self, runner):
        if (runner.iteration + 1) % self.save_interval == 0 or (runner.iteration + 1) == self.max_iters:
            save_path = os.path.join(runner.logdir, "iter_{}_model.pth".format(runner.iteration + 1))
            #
            search_template = runner.logdir + '/' + 'iter_*_model.pth'
            saved_files = glob.glob(search_template)
            if len(saved_files) >= self.max_save_num:
                sorted_files_by_ctime = sorted(saved_files, key=lambda x: os.path.getctime(x))
                os.remove(sorted_files_by_ctime[0])
            torch.save(runner.state_dict(), save_path)
