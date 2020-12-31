"""
https://github.com/catalyst-team/catalyst/blob/master/catalyst/metrics/focal.py
https://github.com/catalyst-team/catalyst/blob/master/catalyst/contrib/nn/criterion/focal.py
"""

from functools import partial

from torch.nn.modules.loss import _Loss  # noqa: WPS450

import torch
import torch.nn.functional as F

def sigmoid_focal_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
):
    """
    Compute binary focal loss between target and output logits.
    Args:
        outputs: tensor of arbitrary shape
        targets: tensor of the same shape as input
        gamma: gamma for focal loss
        alpha: alpha for focal loss
        reduction (string, optional):
            specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"`` | ``"batchwise_mean"``.
            ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output,
            ``"sum"``: the output will be summed.
    Returns:
        computed loss
    Source: https://github.com/BloodAxe/pytorch-toolbelt
    """
    targets = targets.type(outputs.type())

    logpt = -F.binary_cross_entropy_with_logits(
        outputs, targets, reduction="none"
    )
    pt = torch.exp(logpt)

    # compute the loss
    loss = -((1 - pt).pow(gamma)) * logpt

    if alpha is not None:
        loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


def reduced_focal_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    gamma: float = 2.0,
    reduction="mean",
) -> torch.Tensor:
    """Compute reduced focal loss between target and output logits.
    It has been proposed in `Reduced Focal Loss\: 1st Place Solution to xView
    object detection in Satellite Imagery`_ paper.
    .. note::
        ``size_average`` and ``reduce`` params are in the process of being
        deprecated, and in the meantime, specifying either of those two args
        will override ``reduction``.
    Source: https://github.com/BloodAxe/pytorch-toolbelt
    .. _Reduced Focal Loss\: 1st Place Solution to xView object detection
        in Satellite Imagery: https://arxiv.org/abs/1903.01347
    Args:
        outputs: tensor of arbitrary shape
        targets: tensor of the same shape as input
        threshold: threshold for focal reduction
        gamma: gamma for focal reduction
        reduction (string, optional):
            specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"`` | ``"batchwise_mean"``.
            ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output,
            ``"sum"``: the output will be summed.
            ``"batchwise_mean"`` computes mean loss per sample in batch.
            Default: "mean"
    Returns:  # noqa: DAR201
        torch.Tensor: computed loss
    """
    targets = targets.type(outputs.type())

    logpt = -F.binary_cross_entropy_with_logits(
        outputs, targets, reduction="none"
    )
    pt = torch.exp(logpt)

    # compute the loss
    focal_reduction = ((1.0 - pt) / threshold).pow(gamma)
    focal_reduction[pt < threshold] = 1

    loss = -focal_reduction * logpt

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss

class FocalLossBinary(_Loss):
    """Compute focal loss for binary classification problem.
    It has been proposed in `Focal Loss for Dense Object Detection`_ paper.
    @TODO: Docs (add `Example`). Contribution is welcome.
    .. _Focal Loss for Dense Object Detection:
        https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        ignore: int = None,
        reduced: bool = False,
        gamma: float = 2.0,
        alpha: float = 0.25,
        threshold: float = 0.5,
        reduction: str = "mean",
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.ignore = ignore

        if reduced:
            self.loss_fn = partial(
                reduced_focal_loss,
                gamma=gamma,
                threshold=threshold,
                reduction=reduction,
            )
        else:
            self.loss_fn = partial(
                sigmoid_focal_loss,
                gamma=gamma,
                alpha=alpha,
                reduction=reduction,
            )

    def forward(self, logits, targets):
        """
        Args:
            logits: [bs; ...]
            targets: [bs; ...]
        Returns:
            computed loss
        """
        targets = targets.view(-1)
        logits = logits.view(-1)

        if self.ignore is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = targets != self.ignore
            logits = logits[not_ignored]
            targets = targets[not_ignored]

        loss = self.loss_fn(logits, targets)

        return loss


class FocalLossMultiClass(FocalLossBinary):
    """Compute focal loss for multiclass problem.
    Ignores targets having -1 label.
    It has been proposed in `Focal Loss for Dense Object Detection`_ paper.
    @TODO: Docs (add `Example`). Contribution is welcome.
    .. _Focal Loss for Dense Object Detection:
        https://arxiv.org/abs/1708.02002
    """

    def forward(self, input, target):
        """
        Args:
            logits: [bs; num_classes; ...]
            targets: [bs; ...]
        Returns:
            computed loss
        """
        logits=input
        targets=target
        
        num_classes = logits.size(1)
        loss = 0
        targets = targets.view(-1)
        logits = logits.view(-1, num_classes)

        # Filter anchors with -1 label from loss computation
        if self.ignore is not None:
            not_ignored = targets != self.ignore

        for class_id in range(num_classes):
            cls_label_target = (
                targets == (class_id + 0)  # noqa: WPS345
            ).long()
            cls_label_input = logits[..., class_id]

            if self.ignore is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]

            loss += self.loss_fn(cls_label_input, cls_label_target)

        return loss


# @TODO: check
# class FocalLossMultiLabel(_Loss):
#     """Compute focal loss for multilabel problem.
#     Ignores targets having -1 label.
#
#     It has been proposed in `Focal Loss for Dense Object Detection`_ paper.
#
#     @TODO: Docs (add `Example`). Contribution is welcome.
#
#     .. _Focal Loss for Dense Object Detection:
#         https://arxiv.org/abs/1708.02002
#     """
#
#     def forward(self, logits, targets):
#         """
#         Args:
#             logits: [bs; num_classes]
#             targets: [bs; num_classes]
#         """
#         num_classes = logits.size(1)
#         loss = 0
#
#         for cls in range(num_classes):
#             # Filter anchors with -1 label from loss computation
#             if cls == self.ignore:
#                 continue
#
#             cls_label_target = targets[..., cls].long()
#             cls_label_input = logits[..., cls]
#
#             loss += self.loss_fn(cls_label_input, cls_label_target)
#
#         return loss

__all__ = ["FocalLossBinary", "FocalLossMultiClass"]