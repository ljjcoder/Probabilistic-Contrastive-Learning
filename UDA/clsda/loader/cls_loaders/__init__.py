# 
# ----------------------------------------------
from copy import deepcopy
from torch.utils import data
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from mmcv.utils import build_from_cfg
from .builder import CLS_DATASETS
from mmcv.parallel import collate
from .cls_loaders import SSDA_CLS_Datasets, SSDA_CLS_Double_Datasets, SSDA_CLS_Triple_Datasets

def process_one_cls_dataset(args, pipelines, batch_size, n_workers, shuffle, debug=False,
                            sample_num=10, drop_last=True, data_root=None, random_seed=1234):
    dataset_params = deepcopy(args)
    #
    if 'pipeline' not in args:
        dataset_params['pipeline'] = pipelines
    else:
        dataset_params['pipeline'] = args['pipeline']
    #
    if 'root' not in dataset_params:
        dataset_params['root'] = data_root
    #
    if 'batch_size' in dataset_params:
        temp_batch_size = dataset_params['batch_size']
        dataset_params.pop('batch_size')
    else:
        temp_batch_size = batch_size
    #
    dataset = build_from_cfg(dataset_params, CLS_DATASETS)
    #
    if debug:
        print('dataset has {} images'.format(len(dataset)))
        random_sampler = RandomSampler(dataset, replacement=True, num_samples=sample_num)
        loader = data.DataLoader(dataset, batch_size=temp_batch_size, num_workers=n_workers, pin_memory=True,
                                 sampler=random_sampler, collate_fn=collate)
    else:
        loader = data.DataLoader(dataset, batch_size=temp_batch_size, num_workers=n_workers, shuffle=False,
                                 sampler=DistributedSampler(dataset, shuffle=shuffle), pin_memory=True,
                                 drop_last=drop_last, collate_fn=collate)
    return loader
