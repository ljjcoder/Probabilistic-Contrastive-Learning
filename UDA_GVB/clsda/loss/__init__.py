import copy
import logging
import functools

from clsda.loss.loss import cross_entropy2d, conservative_loss, Diff2d, Symkl2d, FocalLoss_Mine, EFocalLoss, \
    FocalLoss
from clsda.loss.loss import bootstrapped_cross_entropy2d, contrastive_loss, contrastive_loss_for_euclidean
from clsda.loss.loss import multi_scale_cross_entropy2d
from clsda.utils import get_root_logger

key2loss = {'cross_entropy': cross_entropy2d,
            'bootstrapped_cross_entropy': bootstrapped_cross_entropy2d,
            'multi_scale_cross_entropy': multi_scale_cross_entropy2d, }


def get_loss_function(cfg):
    logger = get_root_logger()
    if cfg['training']['loss'] is None:
        logger.info("Using default cross entropy loss")
        return cross_entropy2d

    else:
        loss_dict = cfg['training']['loss']
        loss_name = loss_dict['name']
        loss_params = {k: v for k, v in loss_dict.items() if k != 'name'}

        if loss_name not in key2loss:
            raise NotImplementedError('Loss {} not implemented'.format(loss_name))

        logger.info('Using {} with {} params'.format(loss_name,
                                                     loss_params))
        return functools.partial(key2loss[loss_name], **loss_params)


def prob_distance_criterion(criterion_name, n_class=None):
    if criterion_name == 'diff':
        criterion = Diff2d()
    elif criterion_name == "symkl":
        criterion = Symkl2d(n_target_ch=n_class)
    elif criterion_name == "nmlsymkl":
        criterion = Symkl2d(n_target_ch=n_class, size_average=True)
    else:
        raise NotImplementedError()

    return criterion
