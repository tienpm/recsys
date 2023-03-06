# -*- coding: utf-8 -*-

"""
kvc_recsys_training.modeling.abstract_recommender
"""

from logging import getLogger

import numpy as np
import pylab as p
import torch
import torch.nn as nn


class AbstractRecommender(nn.Module):
    """
        Base class for all recommendation models
    """
    def __int__(self):
        self.logger = getLogger()
        super(AbstractRecommender, self).__init__()

    def __str__(self):
        """
            Model prints with number of trainable parameters
            Return:
                The model paprameters (string)
        """
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_params])

        return (
            super.__str__() + "\nTrainable parameters" + f": params"
        )

    def calculate_loss(self, interaction):
        """
            Base class for all recommendation models
        Args:
            metric_operators (operator): the operator that calculate loss for each metrics

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self):
        """

        Returns:
            torch.Tensor: The result of models which predicted for user and items, shape: [batch_size]
        """
        raise NotImplementedError


