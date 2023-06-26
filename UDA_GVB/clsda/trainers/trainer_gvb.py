import torch
import torch.nn.functional as F
from clsda.runner.validator import BaseValidator
from clsda.runner.trainer import BaseTrainer
from clsda.runner.hooks import LossMetrics, ClsAccuracy,  CLSAnalysis, ClsBestAccuracyByVal
from .builder import TRAINER, VALIDATOR
from clsda.utils import get_root_logger
import numpy as np
from clsda.models.cls_models.gvb_network import GVBLoss, calc_coeff


@VALIDATOR.register_module(name='gvb')
class ValidatorGVB(BaseValidator):
    def __init__(self, cuda, local_rank, logdir, test_loaders, model_dict, trainer=None,
                 patience=100, ):
        super(ValidatorGVB, self).__init__(cuda=cuda, local_rank=local_rank, logdir=logdir,
                                           test_loaders=test_loaders,
                                           model_dict=model_dict, trainer=trainer)
        for ind, (key, _) in enumerate(self.test_loaders.items()):
            cls_acc = ClsAccuracy(self, dataset_name=key, pred_key='pred')
            self.register_hook(cls_acc)
            #
        #
        if local_rank == 0:
            best_acc_logger = ClsBestAccuracyByVal(self, patience=patience)
            self.register_hook(best_acc_logger, priority="LOWEST")

    def eval_iter(self, val_batch_data):
        # val_img, val_label, val_name = val_batch_data
        val_img = val_batch_data['img']
        val_label = val_batch_data['gt_label'].squeeze(1)
        val_metas = val_batch_data['img_metas']
        with torch.no_grad():
            _, logits, _ = self.model_dict['base_model'](val_img, True)
            return {'gt': val_label,
                    'img_metas': val_metas,
                    'pred': logits,
                    'feat': logits}


@TRAINER.register_module('gvb')
class TrainerGVB(BaseTrainer):
    def __init__(self, cuda, local_rank, model_dict, optimizer_dict, scheduler_dict,
                 train_loaders=None,
                 logdir=None,
                 log_interval=50,
                 # new parameters for adaptsegnet
                 use_gvbg=True, use_gvbd=True,
                 lambda_adv=1.0, lambda_gvbg=1.0, lambda_gvgd=1.0,
                 using_NCE=False, num_NCE_instance=36, NCE_scale=7.0, NCE_weight=0.2,
                 using_fixmatch=False, lambda_fixmatch=1.0, prob_threshold=0.98,lambda_kld=0.1
                 ):
        super(TrainerGVB, self).__init__(cuda=cuda, local_rank=local_rank, model_dict=model_dict,
                                         optimizer_dict=optimizer_dict,
                                         scheduler_dict=scheduler_dict, train_loaders=train_loaders,
                                         logdir=logdir,
                                         )
        #
        self.use_gvbd = use_gvbd
        self.use_gvbg = use_gvbg
        self.using_NCE = using_NCE
        self.num_NCE_instance = num_NCE_instance
        self.NCE_scale = NCE_scale
        self.NCE_weight = NCE_weight
        self.lambda_adv = lambda_adv
        self.lambda_gvgd = lambda_gvgd
        self.lambda_gvbg = lambda_gvbg
        self.using_fixmatch = using_fixmatch
        self.lambda_fixmatch = lambda_fixmatch
        self.lambda_kld = lambda_kld
        self.prob_threshold = prob_threshold
        #
        self.num_class = self.train_loaders[0].dataset.n_classes
        # 增加记录
        # self.running_metrics.add_metrics([
        #                                   ], group_name='loss', metric_type='avgmeter', log_interval=self.log_interval)
        #
        if local_rank == 0:
            log_names = ['cls', 'adv', 'gvbd', 'gvbg', 'nce','fixmatch','ratio']
            loss_metrics = LossMetrics(log_names=log_names, group_name='loss', log_interval=log_interval)
            self.register_hook(loss_metrics)
        #

    def train_iter(self, *args):
        # src_img_metas ,src_img, src_label= args[0]
        # tgt_img_metas,tgt_labeled_img, tgt_labeled_gt= args[1]
        # tgt_unlabeled_img, _, _ = args[2]
        src_img = args[0]['img']
        src_label = args[0]['gt_label'].squeeze(1)
        tgt_unlabeled_img_weak = args[1][0]['img'].squeeze(0)
        # tgt_unlabeled_gt_weak = args[1][0]['gt_label'].squeeze(1)
        tgt_unlabeled_img_strong = args[1][1]['img'].squeeze(0)
        src_batchsize = src_img.shape[0]
        tgt_batchsize = tgt_unlabeled_img_weak.shape[0]
        all_batchsize = src_batchsize + tgt_batchsize
        # tgt_unlabeled_size = tgt_unlabeled_img_weak.shape[0]
        batch_metrics = {}
        batch_metrics['loss'] = {}
        #
        base_model = self.model_dict['base_model']
        ad_net = self.model_dict['discriminator']
        #
        self.zero_grad_all()
        #
        if self.using_NCE or self.using_fixmatch:
            all_input = torch.cat((src_img, tgt_unlabeled_img_weak, tgt_unlabeled_img_strong), dim=0)
        else:
            all_input = torch.cat((src_img, tgt_unlabeled_img_weak), dim=0)
        #
        features_all, outputs_all, focal_all = base_model(all_input, gvbg=self.use_gvbg)
        outputs_source = outputs_all[0:src_batchsize]
        focal_source = focal_all[0:src_batchsize]
        outputs_target = outputs_all[src_batchsize:(src_batchsize + tgt_batchsize)]
        focal_target = focal_all[src_batchsize:(src_batchsize + tgt_batchsize)]
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        focals = torch.cat((focal_source, focal_target), dim=0)
        softmax_out = F.softmax(outputs, dim=1)
        #
        transfer_loss, mean_entropy, gvbg, gvbd = GVBLoss([softmax_out, focals], ad_net, calc_coeff(self.iteration),
                                                          GVBD=self.use_gvbd, iteration=self.iteration)
        # 监督损失
        loss_supervised = F.cross_entropy(outputs_source, src_label)
        loss = loss_supervised + self.lambda_adv * transfer_loss + gvbg * self.lambda_gvbg + gvbd * self.lambda_gvgd
        #
        # fixmatch
        if self.using_fixmatch:
            strong_aug_pred = outputs_all[(src_batchsize + tgt_batchsize):]
            pseudo_label = torch.softmax(outputs_target.detach(), dim=-1)
            max_probs, tgt_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.prob_threshold).float()
            loss_consistency = (F.cross_entropy(strong_aug_pred, tgt_u, reduction='none') * mask).mean()
            logsoftmax = F.log_softmax(strong_aug_pred, dim=1)
            kld = torch.sum(-logsoftmax / self.num_class, dim=1)
            loss_consistency += (self.lambda_kld * kld * mask).mean()
            loss_consistency_val = loss_consistency.item()
            loss += self.lambda_fixmatch * loss_consistency
            ratio = mask.mean().item()
        else:
            loss_consistency_val = 0
            ratio = 0
        #
        if self.using_NCE:
            outputs_target_strong = outputs_all[(src_batchsize + tgt_batchsize):]
            w_soft = F.softmax(outputs_target[0:self.num_NCE_instance], dim=1)
            s_soft = F.softmax(outputs_target_strong[0:self.num_NCE_instance], dim=1)
            w_s_soft = torch.cat([w_soft, s_soft])
            # print(w_s_soft.shape)
            # exit()
            targets_NCE = torch.from_numpy(np.array([x for x in range(self.num_NCE_instance)]))
            out1_x = w_s_soft
            NCE_2 = torch.mm(out1_x, out1_x.transpose(0, 1).contiguous())
            unit_1 = torch.eye(self.num_NCE_instance * 2).cuda()
            #
            NCE_2 = NCE_2 * (1 - unit_1) + (-100000) * unit_1
            gt_labels_cls_cross = torch.cat([targets_NCE + self.num_NCE_instance, targets_NCE]).cuda()
            nce_loss = F.cross_entropy(self.NCE_scale * NCE_2, gt_labels_cls_cross)
            loss += self.NCE_weight * nce_loss
            nce_loss_val = nce_loss.item()
        else:
            nce_loss_val = 0
        loss.backward()
        #
        self.step_grad_all()
        #
        batch_metrics['loss']['cls'] = loss_supervised.item()
        batch_metrics['loss']['adv'] = transfer_loss.item()
        batch_metrics['loss']['gvbg'] = gvbg.item()
        batch_metrics['loss']['gvbd'] = gvbd.item()
        batch_metrics['loss']['nce'] = nce_loss_val
        batch_metrics['loss']['fixmatch'] = loss_consistency_val
        batch_metrics['loss']['ratio'] = ratio


        return batch_metrics

    def load_pretrained_model(self, weights_path):
        logger = get_root_logger()
        weights = torch.load(weights_path)
        weights = weights['base_model']
        self.model_dict['base_model'].load_state_dict(weights)
        logger.info('load pretrained model {}'.format(weights_path))
