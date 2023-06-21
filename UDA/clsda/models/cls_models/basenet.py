from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function
import math
from .builder import CLS_MODELS
import torch.distributions as dist
import torch.nn.init as init


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
        print('1')
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
        print('2')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)
        print('3')


class GradientReverse(torch.autograd.Function):
    scale = 1.0

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()


def grad_reverse(x, lambd=1.0):
    GradientReverse.scale = lambd
    return GradientReverse.apply(x)


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


@CLS_MODELS.register_module()
class AlexNetBase(nn.Module):
    def __init__(self, pretrained=True):
        super(AlexNetBase, self).__init__()
        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = nn.Sequential(*list(model_alexnet.features._modules.values())[:])
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features

    def optim_parameters(self, lr):
        optim_param = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'classifier' in name:
                    optim_param.append({'params': param, 'lr': lr * 10, 'weight_decay': 0.0005})
                    print('{} will be optimized, lr {}'.format(name, lr * 10))
                else:
                    optim_param.append({'params': param, 'lr': lr, 'weight_decay': 0.0005})
                    print('{} will be optimized, lr {}'.format(name, lr))
            else:
                print('{} will be ignored'.format(name))

        return optim_param


@CLS_MODELS.register_module()
class VGGBase(nn.Module):
    def __init__(self, pretrained=True, no_pool=False):
        super(VGGBase, self).__init__()
        vgg16 = models.vgg16(pretrained=pretrained)
        self.classifier = nn.Sequential(*list(vgg16.classifier._modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features._modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        x = self.classifier(x)
        return x

    def optim_parameters(self, lr):
        optim_param = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'classifier' in name:
                    optim_param.append({'params': param, 'lr': lr * 10, 'weight_decay': 0.0005})
                    print('{} will be optimized, lr {}'.format(name, lr * 10))
                else:
                    optim_param.append({'params': param, 'lr': lr, 'weight_decay': 0.0005})
                    print('{} will be optimized, lr {}'.format(name, lr))
            else:
                print('{} will be ignored'.format(name))

        return optim_param


@CLS_MODELS.register_module()
class Classifier_shallow(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Classifier_shallow, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp
        weights_init(self)

    def forward(self, x, reverse=False, eta=0.1):
        # x = self.bn(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x, x_out


@CLS_MODELS.register_module()
class Classifier_deep(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Classifier_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp
        weights_init(self)

    def forward(self, x, gt=None, reverse=False, eta=0.1, normalize=True):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        if normalize:
            x = F.normalize(x)  # reverse 和 normalize可以交换
        x_out = self.fc2(x) / self.temp
        return x_out




