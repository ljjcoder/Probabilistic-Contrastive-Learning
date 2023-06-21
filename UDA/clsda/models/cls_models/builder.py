# 
# ----------------------------------------------
import torch.nn as nn
from mmcv.utils import Registry, build_from_cfg

CLS_MODELS = Registry('cls_models')


def build_cls_models(cfg, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, CLS_MODELS, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, CLS_MODELS, default_args)


if __name__ == "__main__":
    pass
