import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model):
        self.model = model
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        self.gradients = []
        x.register_hook(self.save_gradient)
        output = self.model(x)
        return output


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, feature_module):
        self.feature_extractor = FeatureExtractor(feature_module)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        x = self.feature_extractor(x)
        return x


class GradCam:
    def __init__(self, feature_module, feat_shape=(512, 7, 7), prob_type='soft_fg'):
        self.feature_module = feature_module
        self.feat_shape = feat_shape
        self.extractor = ModelOutputs(self.feature_module)
        self.prob_type = prob_type

    # def forward(self, input):
    #     return self.feature_module(input)

    def __call__(self, input, index=None):
        output = self.extractor(input)

        if index == None:
            if self.prob_type == 'hard':
                index = torch.argmax(output.detach(), dim=1)
                one_hot = torch.zeros_like(output)
                one_hot.scatter_(1, index.view(index.shape[0], 1), 1)
            elif self.prob_type == 'soft_fg':
                one_hot = torch.softmax(output.detach(), dim=1)
                one_hot[:, 0] = 0
            elif self.prob_type == 'soft_all':
                one_hot = torch.softmax(output.detach(), dim=1)
            else:
                raise RuntimeError('wrong type of prob type')
        else:
            one_hot = torch.zeros_like(output)
            one_hot.scatter_(1, index.view(index.shape[0], 1), 1)

        one_hot.requires_grad = True
        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()  # BxCxHxW
        grads_val = grads_val[-1].view(input.shape[0], *self.feat_shape)
        target = input.view(input.shape[0], *self.feat_shape)

        weights = torch.mean(torch.mean(grads_val, dim=2, keepdim=True), dim=3, keepdim=True)  # BxCx1x1

        cam = torch.sum(target * weights, dim=1)
        cam = torch.max(cam, torch.zeros_like(cam))
        cam_min = torch.min(cam.view(input.shape[0], -1), dim=1)[0]
        cam_max = torch.max(cam.view(input.shape[0], -1), dim=1)[0] + 1e-8
        cam = cam - cam_min.view(cam_min.shape[0], 1, 1)
        cam = cam / cam_max.view(cam_max.shape[0], 1, 1)
        self.feature_module.zero_grad()
        return cam


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))
