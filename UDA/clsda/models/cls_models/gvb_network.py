import numpy as np
import torch.nn as nn
from torchvision import models
import torch
from clsda.models.cls_models.builder import CLS_MODELS
from .basenet import grad_reverse


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


@CLS_MODELS.register_module()
class GVBResNetFc(nn.Module):
    def __init__(self, resnet_name, use_bottleneck=False, bottleneck_dim=256, new_cls=True, class_num=1000):
        super(GVBResNetFc, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        # self.select_layers = nn.Sequential(self.layer3, self.layer4, self.avgpool)

        self.use_bottleneck = use_bottleneck
        self.sigmoid = nn.Sigmoid()
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.gvbg = nn.Linear(bottleneck_dim, class_num)
                self.focal1 = nn.Linear(class_num, class_num)
                self.focal2 = nn.Linear(class_num, 1)
                self.bottleneck.apply(init_weights)
                self.fc.apply(init_weights)
                self.gvbg.apply(init_weights)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
                self.fc.apply(init_weights)
                self.gvbg = nn.Linear(model_resnet.fc.in_features, class_num)
                self.gvbg.apply(init_weights)
                self.__in_features = model_resnet.fc.in_features
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features

    def forward(self, x, gvbg=True):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
        bridge = self.gvbg(x)
        y = self.fc(x)
        if gvbg:
            y = y - bridge
        return x, y, bridge

    def output_num(self):
        return self.__in_features

    def optim_parameters(self, lr):
        if self.new_cls:
            if self.use_bottleneck:
                parameter_list = [{"params": self.feature_layers.parameters(), "lr": lr},
                                  {"params": self.bottleneck.parameters(), "lr": lr * 10},
                                  {"params": self.fc.parameters(), "lr": lr * 10}]
            else:
                parameter_list = [{"params": self.feature_layers.parameters(), "lr": lr},
                                  {"params": self.fc.parameters(), "lr": lr * 10},
                                  {"params": self.gvbg.parameters(), "lr": lr * 10}]
        else:
            parameter_list = [{"params": self.parameters(), "lr": lr}]
        return parameter_list


@CLS_MODELS.register_module()
class GVBAdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size=1024):
        super(GVBAdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.gvbd = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

    def forward(self, x, iteration):
        coeff = calc_coeff(iteration)
        x = grad_reverse(x, lambd=coeff)
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        z = self.gvbd(x)
        return y, z

    def output_num(self):
        return 1

    def optim_parameters(self, lr):
        # TODO: 训练参数
        return [{"params": self.parameters(), "lr": 10 * lr, 'weight_decay': 0.001}]


class Myloss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(Myloss, self).__init__()
        self.epsilon = epsilon
        return

    def forward(self, input_, label, weight):
        entropy = - label * torch.log(input_ + self.epsilon) - (1 - label) * torch.log(1 - input_ + self.epsilon)
        return torch.sum(entropy * weight) / 2


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def GVBLoss(input_list, ad_net, coeff=None, myloss=Myloss(), GVBD=False, iteration=None):
    softmax_output = input_list[0]
    focals = input_list[1].reshape(-1)
    ad_out, fc_out = ad_net(softmax_output, iteration=iteration)
    if GVBD:
        ad_out = nn.Sigmoid()(ad_out - fc_out)
    else:
        ad_out = nn.Sigmoid()(ad_out)
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(softmax_output.device)

    x = softmax_output
    entropy = Entropy(x)
    entropy = grad_reverse(entropy, lambd=coeff)
    entropy = torch.exp(-entropy)
    mean_entropy = torch.mean(entropy)
    gvbg = torch.mean(torch.abs(focals))
    gvbd = torch.mean(torch.abs(fc_out))

    source_mask = torch.ones_like(entropy)
    source_mask[softmax_output.size(0) // 2:] = 0
    source_weight = entropy * source_mask
    target_mask = torch.ones_like(entropy)
    target_mask[0:softmax_output.size(0) // 2] = 0
    target_weight = entropy * target_mask
    weight = source_weight / torch.sum(source_weight).detach().item() + \
             target_weight / torch.sum(target_weight).detach().item()
    return myloss(ad_out, dc_target, weight.view(-1, 1)), mean_entropy, gvbg, gvbd
