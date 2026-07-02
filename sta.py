import random

import numpy as np
import torch
from torchvision.transforms import v2

from ..gradient.mifgsm import MIFGSM


class STA(MIFGSM):
    """
    Structural Transformation Attack (STA).

    This attack extends MI-FGSM with structural image transformations:
    - random block-wise shuffling,
    - elastic transformation,

    Args:
        model_name (str): Name of the surrogate model.
        epsilon (float): Perturbation budget.
        alpha (float): Step size.
        epoch (int): Number of attack iterations.
        num_scale (int): Number of scaled copies used.
        num_block (int): Number of blocks used when splitting each image dimension.
        elastic_factor (float): Strength of the elastic transformation.
        gamma (float): Scaling factor applied to transformed adversarial regions.
        targeted (bool): Whether to perform a targeted attack.
        random_start (bool): Whether to use random initialization for perturbation.
        norm (str): Perturbation norm, either ``"l2"`` or ``"linfty"``.
        loss (str): Loss function name.
        device (torch.device | None): Device used for attack data.
        attack (str): Attack name.

    Official settings:
        epsilon = 10 / 255
        alpha = epsilon / epoch = 1.0 / 255
        epoch = 10
        num_scale = 5
        num_block = 9
    """

    def __init__(
        self,
        model_name,
        epsilon=10 / 255,
        alpha=1 / 255,
        epoch=10,
        num_scale=5,
        num_block=9,
        elastic_factor=150.0,
        gamma=0.3,
        targeted=False,
        random_start=False,
        norm="linfty",
        loss="crossentropy",
        device=None,
        attack="STA",
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            epsilon=epsilon,
            alpha=alpha,
            epoch=epoch,
            targeted=targeted,
            random_start=random_start,
            norm=norm,
            loss=loss,
            device=device,
            attack=attack,
        )

        self.num_scale = num_scale
        self.num_block = num_block
        self.elastic_factor = elastic_factor
        self.gamma = gamma
        self.transform_scale = epsilon

    def _random_split_lengths(self, total_length):

        random_weights = np.random.uniform(size=self.num_block)
        split_lengths = np.round(
            random_weights / random_weights.sum() * total_length
        ).astype(np.int32)

        split_lengths[split_lengths.argmax()] += total_length - split_lengths.sum()

        return tuple(split_lengths)


    def _shuffle_along_dim(self, x, dim):

        split_lengths = self._random_split_lengths(x.size(dim))
        strips = list(x.split(split_lengths, dim=dim))
        random.shuffle(strips)
        return strips


    def _shuffle_spatial_blocks(self, x):

        spatial_dims = [2, 3]  # height, width
        random.shuffle(spatial_dims)

        first_dim, second_dim = spatial_dims

        strips = self._shuffle_along_dim(x, dim=first_dim)

        shuffled_strips = [
            torch.cat(
                self._shuffle_along_dim(strip, dim=second_dim),
                dim=second_dim,
            )
            for strip in strips
        ]

        return torch.cat(shuffled_strips, dim=first_dim)


    @staticmethod
    def _positive_difference_mask(x_adv, x_clean):

        return torch.clamp(torch.sign(x_adv - x_clean), min=0, max=1)


    def _elastic_shuffle_transform(self, x):

        elastic_transform = v2.ElasticTransform(
            alpha=self.elastic_factor,
            sigma=6.5,
        )

        transformed = elastic_transform(x)
        transformed = self._shuffle_spatial_blocks(transformed)

        return self.transform_scale * transformed


    def transform(self, x, **kwargs):

        delta = self.init_delta(x)

        x_clean = x.clone().detach()
        elastic_shuffle_noise = self._elastic_shuffle_transform(x_clean)

        x_adv = x + delta + elastic_shuffle_noise

        mask = self._positive_difference_mask(x_adv, x)
        x_adv = self.gamma * x_adv * mask + x * (1 - mask)

        delta = x_adv - x

        scaled_inputs = [x_adv / (2**scale_idx) for scale_idx in range(self.num_scale)]

        return torch.cat(scaled_inputs, dim=0)
        

    def get_loss(self, logits, label):

        repeated_labels = label.repeat(self.num_scale)

        if self.targeted:
            return -self.loss(logits, repeated_labels)

        return self.loss(logits, repeated_labels)
