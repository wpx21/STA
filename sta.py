import torch
from transferattack.utils import *
import random
import numpy as np
from transferattack.gradient.mifgsm import MIFGSM
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms import v2

class STA(MIFGSM):
    """
    STA Attack
    """

    def __init__(self, model_name, epsilon=16 / 255, num_block=4,alpha=1.6 / 255, epoch=10, feature_layer=2, decay=1., num_scale=5, alp_scale=120., gamma=0.3, targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='STA', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_scale = num_scale
        self.beta = epsilon
        self.gamma = gamma
        self.alp_scale = alp_scale
        self.num_block = num_block

    def get_length(self, length):
        rand = np.random.uniform(size=self.num_block)
        rand_norm = np.round(rand/rand.sum()*length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def shuffle(self, x):
        dims = [2,3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])
        return torch.cat([torch.cat(self.shuffle_single_dim(x_strip, dim=dims[1]), dim=dims[1]) for x_strip in x_strips], dim=dims[0])

    def get_mask(self, x1, x2):
        mask = torch.clamp(torch.sign(x1 - x2), 0, 1)
        return mask

    def get_st(self, x):
        elastic_transformer = v2.ElasticTransform(alpha=self.alp_scale, sigma=6.5)
        x = elastic_transformer(x)
        x = self.shuffle(x)
        return self.beta * x

# STA

    def transform(self, x, **kwargs):
        delta = self.init_delta(x)
        x_adv = x.clone().detach()
        x_st = self.get_st(x_adv.clone().detach())
        x_adv = x + delta + x_st
        mask = self.get_mask(x_adv, x)
        x_adv = (self.gamma) * x_adv * mask + x * (1 - mask)
        delta = x_adv - x
        return torch.cat([x_adv / (2 ** i) for i in range(self.num_scale)])##### integrate SIM
        # return x_adv##### withou SIM

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return -self.loss(logits, label.repeat(self.num_scale)) if self.targeted else self.loss(logits, label.repeat(self.num_scale))##### integrate SIM
