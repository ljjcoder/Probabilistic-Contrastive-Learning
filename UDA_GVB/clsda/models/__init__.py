import torch.nn as nn
from clsda.models.cls_models import parse_args_for_one_cls_model


def parse_args_for_models(model_args, n_classes=None, task_type=None, cuda=True, find_unused_parameters=False,
                          sync_bn=False):
    # setup task type
    if task_type == 'cls':
        parse_args_for_one_model = parse_args_for_one_cls_model
    else:
        raise RuntimeError('wrong dataset task name {}'.format(task_type))

    shared_lr_scheduler_param = model_args['lr_scheduler']
    model_args.pop('lr_scheduler')
    model_dict = nn.ModuleDict()
    optimizer_dict = {}
    scheduler_dict = {}
    # 获得一个进行最多需要多少块卡（model parallel）
    max_need_card = 0
    for key in model_args:
        tmp_device = model_args[key].get('device', 0)
        if tmp_device > max_need_card:
            max_need_card = tmp_device
    max_need_card += 1
    #
    for key in model_args:
        temp_res = parse_args_for_one_model(model_args[key], shared_lr_scheduler_param, cuda=cuda,
                                            find_unused_parameters=find_unused_parameters, max_card=max_need_card,
                                            sync_bn=sync_bn)
        model_dict[key] = temp_res[0]
        if temp_res[1] is not None:
            optimizer_dict[key] = temp_res[1]
            scheduler_dict[key] = temp_res[2]
    return model_dict, optimizer_dict, scheduler_dict
