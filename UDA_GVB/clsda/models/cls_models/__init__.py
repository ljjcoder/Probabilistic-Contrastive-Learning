# 
# ----------------------------------------------
import torch
import copy
import os
from .builder import build_cls_models
from clsda.optimizers import build_model_defined_optimizer
from clsda.schedulers import build_scheduler
from clsda.utils.utils import move_models_to_gpu
from .gvb_network import GVBAdversarialNetwork, GVBResNetFc


def parse_args_for_one_cls_model(model_args, scheduler_args, n_classes=None, cuda=True, find_unused_parameters=False,
                                 max_card=1, sync_bn=False):
    """
    输入带名字的字典，
    :param model_args: 类型是字典，名字就是model，optimizer，scheduler的名字的前者
    :param scheduler_args:
    :param logger:
    :return:
    """
    assert 'optimizer' in model_args.keys(), 'model args should have optimizer args'
    model_args = copy.deepcopy(model_args)
    # 获取参数
    optimizer_params = model_args['optimizer']
    model_args.pop('optimizer')
    scheduler_params = model_args.get('scheduler', None)
    if scheduler_params is not None:
        model_args.pop('scheduler')
    else:
        scheduler_params = scheduler_args
    #
    device_params = model_args.get('device', 0)
    if 'device' in model_args.keys():
        model_args.pop(device_params)
    # 构造模型
    temp_model = build_cls_models(model_args)
    if sync_bn:
        print('Using sync_bn mode, Please pay attention for the shufflebn case!!!!!!!!!!!!!!!!!')
        temp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(temp_model)
    # move model to gpu
    if cuda:
        temp_model = move_models_to_gpu(temp_model, device_params, max_card=max_card,
                                        find_unused_parameters=find_unused_parameters)
    if optimizer_params is not None:
        temp_optimizer = build_model_defined_optimizer(temp_model, optimizer_params)
        temp_scheduler = build_scheduler(temp_optimizer, scheduler_params)
    else:
        temp_optimizer = None
        temp_scheduler = None

    return temp_model, temp_optimizer, temp_scheduler
