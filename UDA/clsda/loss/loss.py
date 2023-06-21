import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def cross_entropy2d(input, target, weight=None, reduction='elementwise_mean', size_average=True, reduce=True,
                    align_corners=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # print('{},{},{},{}'.format(h,w,ht,wt))
    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.unsqueeze(1)
        target = F.interpolate(target.to(torch.float), size=(h, w), mode="nearest")
        target = target.squeeze(1).to(torch.long)
    elif h < ht and w < wt:  # upsample images
        # print('upsample input feature')
        # print('align corners {}'.format(align_corners))
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=align_corners)  # 针对0.4.0
    elif h != ht and w != wt:
        raise Exception("Only support upsampling, h,w is {}{}, target h,w is {}{}".format(h, w, ht, wt))

    # input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    # target = target.view(-1)  # 拉成一列
    # loss = F.cross_entropy(input, target, weight=weight, reduction=reduction, ignore_index=255)
    loss = F.cross_entropy(input, target, weight=weight, size_average=size_average, ignore_index=255, reduce=reduce)
    return loss


def conservative_loss(input, target):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.unsqueeze(1)
        target = F.interpolate(target, size=(h, w), mode="nearest")
        target = target.squeeze(1)
    elif h < ht and w < wt:  # upsample images
        # print('upsample input feature')
        input = F.interpolate(input, size=(ht, wt), mode="bilinear")  # 针对0.4.0
    elif h != ht and w != wt:
        raise Exception("Only support upsampling, h,w is {}{}, target h,w is {}{}".format(h, w, ht, wt))

    # input and target are both flattened
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)

    # use built-in cross_entropy loss to calculate -log(p)
    ce_loss = F.cross_entropy(input, target, size_average=False, ignore_index=255, reduce=False)

    # as there are ignored pixel(index255), which return ce_loss=0,
    # when sent them into torch.log() below, change ce_loss from 0 to 1.0, so cl_loss=0(log(1.0)=0)
    with torch.no_grad():
        small_val_indices = (ce_loss < 0.0001).nonzero()
        ce_loss[small_val_indices] = 1.0

    # use detach() means grad doesn't back propagete through modulating_factor
    # TODO: 这里暂时不detach()
    modulating_factor = torch.pow(1 - ce_loss, 2)
    cl_loss = modulating_factor * torch.log(ce_loss)
    cl_loss = torch.mean(cl_loss)
    return cl_loss


def multi_scale_cross_entropy2d(
        input, target, weight=None, reduction='sum', scale_weight=None
):
    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight == None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp))

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(input=inp, target=target, weight=weight, reduction=reduction)
    return loss


def bootstrapped_cross_entropy2d(input,
                                 target,
                                 K,
                                 weight=None,
                                 reduction='sum'):
    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input,
                                   target,
                                   K,
                                   weight=None,
                                   reduction='none'):
        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(input,
                               target,
                               weight=weight,
                               reduction='none',
                               ignore_index=255)

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            reduction=reduction,
        )
    return loss / float(batch_size)


class Diff2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Diff2d, self).__init__()
        self.weight = weight

    def forward(self, inputs1, inputs2):
        return torch.mean(torch.abs(F.softmax(inputs1, dim=1) - F.softmax(inputs2, dim=1)))


class Symkl2d(nn.Module):
    def __init__(self, weight=None, n_target_ch=21, size_average=True):
        super(Symkl2d, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.n_target_ch = n_target_ch

    def forward(self, inputs1, inputs2):
        self.prob1 = F.softmax(inputs1, dim=1)
        self.prob2 = F.softmax(inputs2, dim=1)
        self.log_prob1 = F.log_softmax(inputs1, dim=1)
        self.log_prob2 = F.log_softmax(inputs2, dim=1)

        loss = 0.5 * (F.kl_div(self.log_prob1, self.prob2, size_average=self.size_average)
                      + F.kl_div(self.log_prob2, self.prob1, size_average=self.size_average))

        return loss


class FocalLoss_Mine(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, ignore_index=255, cuda=False):
        super(FocalLoss_Mine, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = torch.tensor(alpha).unsqueeze(1)

        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.ignore_index = torch.tensor(255)
        self.zero_tensor = torch.tensor(0.0)
        self.one_tensor = torch.tensor(1.0)
        if cuda:
            self.ignore_index = self.ignore_index.cuda()
            self.zero_tensor = self.zero_tensor.cuda()
            self.one_tensor = self.one_tensor.cuda()
            self.alpha = self.alpha.cuda()

    def forward(self, inputs, targets):
        # print('input shape {}, targets shape {}'.format(inputs.shape, targets.shape))
        N = inputs.size(0)
        C = inputs.size(1)
        if C == 1:
            inputs = inputs.view(-1)
            probs = torch.sigmoid(inputs)
            targets = targets.view(-1)
            ignore_mask = torch.where(targets == self.ignore_index, self.zero_tensor, self.one_tensor)
            targets[targets == self.ignore_index] = 0
            # print('max val of targets is {}'.format(torch.max(targets)))
            alpha = self.alpha
            # print('alpha {}'.format(alpha.item()))
            # print('ignore mask sum {}, targets sum {}, prob sum {}'.format(torch.sum(ignore_mask), torch.sum(targets),
            #                                                                torch.sum(probs)))
        else:
            P = F.softmax(inputs, dim=1)
            inputs = inputs.view(N, -1)
            C = inputs.size(1)
            # print('C is {}, numel {}'.format(C,torch.numel(inputs)))
            class_mask = torch.zeros((N, C)).to(inputs.device)
            ids = targets.view(-1, 1)
            ignore_mask = torch.where(ids == self.ignore_index, self.zero_tensor, self.one_tensor)
            ids[ids == self.ignore_index] = 0
            # print('class mask shape {}, ids shape {}'.format(class_mask.shape, ids.shape))
            class_mask.scatter_(1, ids, 1.)
            # print(class_mask)

            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.to(inputs.device)
            alpha = self.alpha[ids.view(-1)]  # alpha的维度还是(ids的元素个数 x 1)

            probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = torch.log(probs)
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p * ignore_mask
        # print('batch loss {}'.format(batch_loss))
        # print('self gamma is {}'.format(self.gamma))
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class EFocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(EFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        # inputs = torch.sigmoid(inputs)
        P = F.softmax(inputs)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)
        batch_loss = -alpha * torch.exp(-self.gamma * probs) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, sigmoid=False, reduce=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.sigmoid = sigmoid
        self.reduce = reduce

    def forward(self, inputs, targets):
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        if self.sigmoid:
            P = torch.sigmoid(inputs)
            # print('shape {}, shape {}'.format(inputs.shape,P.shape))
            # print('target shape {}'.format(targets.shape))
            # F.softmax(inputs)
            # if targets == 0:
            #     probs = 1 - P  # (P * class_mask).sum(1).view(-1, 1)
            #     log_p = probs.log()
            #     batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
            # if targets == 1:
            #     probs = P  # (P * class_mask).sum(1).view(-1, 1)
            #     log_p = probs.log()
            #     batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
            probs = P * targets + (1 - P) * (1 - targets) + 1e-8
            log_p = probs.log()
            batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
        else:
            # inputs = torch.sigmoid(inputs)
            P = F.softmax(inputs, dim=1)

            class_mask = inputs.data.new(N, C).fill_(0)
            class_mask = Variable(class_mask)
            ids = targets.view(-1, 1)
            class_mask.scatter_(1, ids.data, 1.)
            # print(class_mask)

            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.cuda()
            alpha = self.alpha[ids.data.view(-1)]

            probs = (P * class_mask).sum(1).view(-1, 1)

            log_p = probs.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)

            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
            # print('-----bacth_loss------')
            # print(batch_loss)

        if not self.reduce:
            return batch_loss
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalPseudo(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, threshold=0.8):
        super(FocalPseudo, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.threshold = threshold

    def forward(self, inputs):
        N = inputs.size(0)
        C = inputs.size(1)
        inputs = inputs[0, :, :]
        # print(inputs)
        # pdb.set_trace()
        inputs, ind = torch.max(inputs, 1)
        ones = torch.ones(inputs.size()).cuda()
        value = torch.where(inputs > self.threshold, inputs, ones)
        #
        # pdb.set_trace()
        # ind
        # print(value)
        try:
            ind = value.ne(1)
            indexes = torch.nonzero(ind, as_tuple=False)
            # value2 = inputs[indexes]
            inputs = inputs[indexes]
            log_p = inputs.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)
            if not self.gamma == 0:
                batch_loss = - (torch.pow((1 - inputs), self.gamma)) * log_p
            else:
                batch_loss = - log_p
        except:
            # inputs = inputs#[indexes]
            log_p = value.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)
            if not self.gamma == 0:
                batch_loss = - (torch.pow((1 - inputs), self.gamma)) * log_p
            else:
                batch_loss = - log_p
        # print('-----bacth_loss------')
        # print(batch_loss)
        # batch_loss = batch_loss #* weight
        if self.size_average:
            try:
                loss = batch_loss.mean()  # + 0.1*balance
            except:
                pdb.set_trace()
        else:
            loss = batch_loss.sum()
        return loss


def contrastive_loss(rois_label, cluster_label, sim_matrix, class_num, lambda_other_positive=0.5, lambda_negative=1.0,
                     margin=0.0, instance_weight=None, negative_weight_type='normal', weights_temperature=1.0,
                     cls_negative_weights=None, element_weights=None):
    if instance_weight is None:
        instance_weight = torch.ones(rois_label.shape[0]).to(rois_label.device) / rois_label.shape[0]
    if element_weights is None:
        element_weights = torch.ones_like(sim_matrix)
    cluster_num_per_class = int(cluster_label.shape[0] / class_num)
    lambda_nearest_positive = 1
    if cluster_num_per_class > 1:
        lambda_other_positive = 1.0 / (cluster_num_per_class - 1) * lambda_other_positive
    else:
        lambda_other_positive = 0
    # rois_label indicate mask
    rois_ind_mask = rois_label.unsqueeze(1).expand(rois_label.shape[0], cluster_label.shape[0]).to(torch.long)
    # cluster_label indicate mask
    cluster_ind_mask = cluster_label.unsqueeze(0).expand(rois_label.shape[0], cluster_label.shape[0]).to(torch.long)
    # 最终的形式表示为max{a, b+c*similarity}
    # 对于最接近的正样本，a=-2, b=1, c=-lambda_nearest_positive
    # 对于其他正样本， a=-2, b=lambda_other_positive, c=-lambda_other_positive
    # 对于负样本，a=0, b=-margin * lambda_negative, c=lambda_negative
    #
    # 首先设置负样本的系数, 对负样本的处理选择不同的模式
    mask_a = torch.zeros((rois_label.shape[0], cluster_label.shape[0])).to(rois_label.device)
    if negative_weight_type == 'normal':
        #
        lambda_negative = lambda_negative / (cluster_label.shape[0] - cluster_num_per_class)
        if isinstance(margin, (int, float)):
            mask_b = torch.ones((rois_label.shape[0], cluster_label.shape[0]), device=rois_label.device) * (
                    -margin * lambda_negative)
        elif isinstance(margin, torch.Tensor):
            mask_b = -margin[rois_label, :] * lambda_negative
        else:
            raise RuntimeError('wrong type of margin {}'.format(type(margin)))
        mask_c = torch.ones((rois_label.shape[0], cluster_label.shape[0]), device=rois_label.device) * lambda_negative
    elif negative_weight_type == 'only_bg':
        # 只考虑背景类别作为负样本
        mask_b = torch.zeros((rois_label.shape[0], cluster_label.shape[0]), device=rois_label.device)
        mask_c = torch.zeros((rois_label.shape[0], cluster_label.shape[0]), device=rois_label.device)
        #
        mask_b[:, cluster_num_per_class] = -margin * lambda_negative
        mask_c[:, cluster_num_per_class] = lambda_negative
    elif negative_weight_type == 'sim_related':
        # 根据相似程度加权，相似性越大，权重越高
        normalized_sim = torch.softmax(sim_matrix.detach() * weights_temperature, dim=1)
        mask_b = normalized_sim * (-margin * lambda_negative)
        mask_c = normalized_sim * lambda_negative
    else:
        raise RuntimeError('wrong negative weight type {}'.format(negative_weight_type))
    # 为每一个单独设置负样本对的系数，如果该类样本数较少，则负样本对的系数较低
    # mask_b, mask_c都要乘以相应的系数
    if cls_negative_weights is not None:
        tmp_weights = cls_negative_weights[rois_label, :]
        mask_b = mask_b * tmp_weights
        mask_c = mask_c * tmp_weights
    # 设置所有正样本的a,b,和其它正样本的c
    same_cls_pos_ind_1, same_cls_pos_ind_2 = torch.nonzero(rois_ind_mask == cluster_ind_mask, as_tuple=True)
    mask_a[same_cls_pos_ind_1, same_cls_pos_ind_2] = -2
    mask_b[same_cls_pos_ind_1, same_cls_pos_ind_2] = lambda_other_positive
    mask_c[same_cls_pos_ind_1, same_cls_pos_ind_2] = -lambda_other_positive
    # 设置最接近的正样本的系数
    row_ind = torch.arange(rois_label.shape[0]).to(rois_label.device)
    same_cls_ind = same_cls_pos_ind_2.view(rois_label.shape[0], cluster_num_per_class)
    in_class_sim_matrix = sim_matrix[same_cls_pos_ind_1, same_cls_pos_ind_2].view(rois_label.shape[0],
                                                                                  cluster_num_per_class)
    nearest_pos_ind = torch.argmax(in_class_sim_matrix, dim=1)
    nearest_orig_ind = same_cls_ind[row_ind, nearest_pos_ind]
    mask_b[row_ind, nearest_orig_ind] = 1
    mask_c[row_ind, nearest_orig_ind] = -lambda_nearest_positive
    # contrastive loss
    contrastive_loss = torch.max(mask_a, mask_b + mask_c * sim_matrix)
    contrastive_loss = torch.sum(torch.sum(contrastive_loss * element_weights, dim=1) * instance_weight)
    return contrastive_loss


def contrastive_loss_for_euclidean(rois_label, cluster_label, dist_matrix, class_num, lambda_other_positive=0.5,
                                   lambda_negative=1.0,
                                   margin=1.0, instance_weight=None,
                                   element_weights=None):
    if instance_weight is None:
        instance_weight = torch.ones(rois_label.shape[0]).to(rois_label.device) / rois_label.shape[0]
    if element_weights is None:
        element_weights = torch.ones_like(dist_matrix)
    cluster_num_per_class = int(cluster_label.shape[0] / class_num)
    lambda_nearest_positive = 1.0
    if cluster_num_per_class > 1:
        lambda_other_positive = 1.0 / (cluster_num_per_class - 1) * lambda_other_positive
    else:
        lambda_other_positive = 0
    # rois_label indicate mask
    rois_ind_mask = rois_label.unsqueeze(1).expand(rois_label.shape[0], cluster_label.shape[0]).to(torch.long)
    # cluster_label indicate mask
    cluster_ind_mask = cluster_label.unsqueeze(0).expand(rois_label.shape[0], cluster_label.shape[0]).to(torch.long)
    # 最终的形式表示为max{a, b+c*similarity}
    # 对于最接近的正样本，a=-2, b=1, c=-lambda_nearest_positive
    # 对于其他正样本， a=-2, b=lambda_other_positive, c=-lambda_other_positive
    # 对于负样本，a=0, b=-margin * lambda_negative, c=lambda_negative
    #
    # 首先设置负样本的系数
    mask_a = torch.zeros((rois_label.shape[0], cluster_label.shape[0])).to(rois_label.device)
    lambda_negative = lambda_negative / (cluster_label.shape[0] - cluster_num_per_class)
    if isinstance(margin, (int, float)):
        mask_b = torch.ones((rois_label.shape[0], cluster_label.shape[0]), device=rois_label.device) * (
                margin * lambda_negative)
    elif isinstance(margin, torch.Tensor):
        mask_b = margin[rois_label, :] * lambda_negative
    else:
        raise RuntimeError('wrong type of margin {}'.format(type(margin)))
    mask_c = torch.ones((rois_label.shape[0], cluster_label.shape[0]), device=rois_label.device) * (-lambda_negative)
    # 为每一个单独设置负样本对的系数，如果该类样本数较少，则负样本对的系数较低
    # mask_b, mask_c都要乘以相应的系数
    # 设置所有正样本的a,b,和其它正样本的c
    same_cls_pos_ind_1, same_cls_pos_ind_2 = torch.nonzero(rois_ind_mask == cluster_ind_mask, as_tuple=True)
    mask_b[same_cls_pos_ind_1, same_cls_pos_ind_2] = 0
    mask_c[same_cls_pos_ind_1, same_cls_pos_ind_2] = lambda_other_positive
    # 设置最接近的正样本的系数
    row_ind = torch.arange(rois_label.shape[0]).to(rois_label.device)
    same_cls_ind = same_cls_pos_ind_2.view(rois_label.shape[0], cluster_num_per_class)
    in_class_sim_matrix = dist_matrix[same_cls_pos_ind_1, same_cls_pos_ind_2].view(rois_label.shape[0],
                                                                                   cluster_num_per_class)
    nearest_pos_ind = torch.argmax(in_class_sim_matrix, dim=1)
    nearest_orig_ind = same_cls_ind[row_ind, nearest_pos_ind]
    mask_b[row_ind, nearest_orig_ind] = 0
    mask_c[row_ind, nearest_orig_ind] = lambda_nearest_positive
    # contrastive loss
    contrastive_loss = torch.max(mask_a, mask_b + mask_c * dist_matrix)
    contrastive_loss = torch.sum(torch.sum(contrastive_loss * element_weights, dim=1) * instance_weight)
    return contrastive_loss


def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0) // 2:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0) // 2] = 0
        target_weight = entropy * target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(
            weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)
