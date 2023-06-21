# 
# ----------------------------------------------
from mmcv.utils import Registry, build_from_cfg
from mmcls.datasets import PIPELINES as CLS_PIPELINES

CLS_DATASETS = Registry('cls_datasets')


def build_dataset(cfg, default_args=None):
    dataset = build_from_cfg(cfg, CLS_DATASETS, default_args)
    return dataset
