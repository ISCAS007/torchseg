# -*- coding: utf-8 -*-


def freeze_layer(layer):
    """
    freeze layer weights
    """
    for param in layer.parameters():
        param.requires_grad = False