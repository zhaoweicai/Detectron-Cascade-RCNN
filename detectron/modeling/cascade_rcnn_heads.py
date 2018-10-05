# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Various network "heads" for classification and bounding box prediction.

The design is as follows:

... -> RoI ----\                               /-> box cls output -> cls loss
                -> RoIFeatureXform -> box head
... -> Feature /                               \-> box reg output -> reg loss
       Map

The Cascade R-CNN head produces a feature representation of the RoI for the purpose
of bounding box classification and regression. The box output module converts
the feature representation into classification and regression predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.utils.blob as blob_utils

# ---------------------------------------------------------------------------- #
# Cascade R-CNN outputs and losses
# by Zhaowei Cai
# ---------------------------------------------------------------------------- #


def add_cascade_proposal_outputs(model, stage, pre_stage):
    """Add RoI classification and bounding box regression output ops."""
    stage_name = "_{}".format(stage)
    pre_stage_name = "_{}".format(pre_stage) if pre_stage >= 2 else ""

    # decode bboxes from previous stages
    bbox_reg_weights = cfg.CASCADE_RCNN.BBOX_REG_WEIGHTS[pre_stage - 1]
    blobs_in = ["bbox_pred" + pre_stage_name, "rois" + pre_stage_name]
    if model.train:
        blobs_in += ["mapped_gt_boxes" + pre_stage_name]
    model.DecodeBBoxes(blobs_in, "proposals" + stage_name, bbox_reg_weights)

    if cfg.FPN.FPN_ON:
        # DistributeCascadeProposals also labels proposals when in training mode
        model.DistributeCascadeProposals(stage)
    else:
        raise NotImplementedError('Only support FPN for now!')


def add_cascade_rcnn_outputs(model, blob_in, dim, stage):
    """Add RoI classification and bounding box regression output ops."""
    stage_name = "_{}".format(stage)
    model.FC(
        blob_in,
        "cls_score" + stage_name,
        dim,
        model.num_classes,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0),
    )
    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        model.Softmax("cls_score" + stage_name, "cls_prob" + stage_name, engine="CUDNN")

    num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes
    model.FC(
        blob_in,
        "bbox_pred" + stage_name,
        dim,
        num_bbox_reg_classes * 4,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0),
    )
    # add stage parameters to list
    if str(stage) not in model.stage_params:
        model.stage_params[str(stage)] = []
    for idx in range(-2, 0):
        model.stage_params[str(stage)].append(model.weights[idx])
        model.stage_params[str(stage)].append(model.biases[idx])
    return "cls_prob" + stage_name, "bbox_pred" + stage_name


def add_cascade_rcnn_losses(model, stage):
    """Add losses for RoI classification and bounding box regression."""
    stage_name = "_{}".format(stage)
    if cfg.CASCADE_RCNN.SCALE_LOSS:
        loss_scalar = cfg.CASCADE_RCNN.STAGE_WEIGHTS[stage - 1]
    else:
        loss_scalar = 1.0
    cls_prob, loss_cls = model.net.SoftmaxWithLoss(
        ["cls_score" + stage_name, "labels_int32" + stage_name],
        ["cls_prob" + stage_name, "loss_cls" + stage_name],
        scale=model.GetLossScale() * loss_scalar,
    )
    loss_bbox = model.net.SmoothL1Loss(
        [
            "bbox_pred" + stage_name,
            "bbox_targets" + stage_name,
            "bbox_inside_weights" + stage_name,
            "bbox_outside_weights" + stage_name,
        ],
        "loss_bbox" + stage_name,
        scale=model.GetLossScale() * loss_scalar,
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, loss_bbox])
    model.Accuracy(
        ["cls_prob" + stage_name, "labels_int32" + stage_name],
        "accuracy_cls" + stage_name,
    )
    model.AddLosses(["loss_cls" + stage_name, "loss_bbox" + stage_name])
    model.AddMetrics("accuracy_cls" + stage_name)
    bbox_reg_weights = cfg.CASCADE_RCNN.BBOX_REG_WEIGHTS[stage - 1]
    model.AddBBoxAccuracy(
        [
            "bbox_pred" + stage_name,
            "rois" + stage_name,
            "labels_int32" + stage_name,
            "mapped_gt_boxes" + stage_name,
        ],
        ["bbox_iou" + stage_name, "bbox_iou" + stage_name + "_pre"],
        bbox_reg_weights,
    )
    model.AddMetrics(["bbox_iou" + stage_name, "bbox_iou" + stage_name + "_pre"])
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #


def add_roi_2mlp_head(model, blob_in, dim_in, spatial_scale, stage):
    """Add a ReLU MLP with two hidden layers."""
    stage_name = "_{}".format(stage)
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        "roi" + stage_name + "_feat",
        blob_rois="rois" + stage_name,
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale,
    )

    if cfg.CASCADE_RCNN.SCALE_GRAD:
        grad_scalar = cfg.CASCADE_RCNN.STAGE_WEIGHTS[stage - 1]
        model.net.Scale(
            roi_feat, roi_feat, scale=1.0, scale_grad=grad_scalar
        )

    model.FC(
        roi_feat, "fc6" + stage_name, dim_in * roi_size * roi_size, hidden_dim
    )
    model.Relu("fc6" + stage_name, "fc6" + stage_name)
    model.FC("fc6" + stage_name, "fc7" + stage_name, hidden_dim, hidden_dim)
    model.Relu("fc7" + stage_name, "fc7" + stage_name)
    # add stage parameters to list
    if str(stage) not in model.stage_params:
        model.stage_params[str(stage)] = []
    for idx in range(-2, 0):
        model.stage_params[str(stage)].append(model.weights[idx])
        model.stage_params[str(stage)].append(model.biases[idx])
    return "fc7" + stage_name, hidden_dim


def add_roi_2mlp_head_shared(model, head_stage, roi_stage, dim_in=None):
    """Add a 2MLP head and output, sharing weights with another head."""
    add_stage_name = "_{}_{}".format(head_stage, roi_stage)
    shared_stage_name = ""
    if head_stage > 1:
        shared_stage_name = "_{}".format(head_stage)
    roi_feat = "roi_{}_feat".format(roi_stage)
    model.FCShared(
        roi_feat,
        "fc6" + add_stage_name,
        weight="fc6" + shared_stage_name + "_w",
        bias="fc6" + shared_stage_name + "_b",
    )
    model.Relu("fc6" + add_stage_name, "fc6" + add_stage_name)
    model.FCShared(
        "fc6" + add_stage_name,
        "fc7" + add_stage_name,
        weight="fc7" + shared_stage_name + "_w",
        bias="fc7" + shared_stage_name + "_b",
    )
    model.Relu("fc7" + add_stage_name, "fc7" + add_stage_name)

    model.FCShared(
        "fc7" + add_stage_name,
        "cls_score" + add_stage_name,
        weight="cls_score" + shared_stage_name + "_w",
        bias="cls_score" + shared_stage_name + "_b",
    )
    model.Softmax(
        "cls_score" + add_stage_name, "cls_prob" + add_stage_name, engine="CUDNN"
    )
    return "cls_prob" + add_stage_name


def add_roi_Xconv1fc_gn_head(model, blob_in, dim_in, spatial_scale, stage):
    """Add a X conv + 1fc head, with GroupNorm"""
    stage_name = "_{}".format(stage)
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        "roi" + stage_name + "_feat",
        blob_rois='rois' + stage_name,
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    if cfg.CASCADE_RCNN.SCALE_GRAD:
        grad_scalar = cfg.CASCADE_RCNN.STAGE_WEIGHTS[stage - 1]
        model.net.Scale(
            roi_feat, roi_feat, scale=1.0, scale_grad=grad_scalar
        )

    current = roi_feat
    num_convs = cfg.FAST_RCNN.NUM_STACKED_CONVS
    for i in range(num_convs):
        current = model.ConvGN(
            current, 'head_conv' + str(i + 1) + stage_name,
            dim_in, hidden_dim, 3,
            group_gn=get_group_gn(hidden_dim),
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6' + stage_name, dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6' + stage_name, 'fc6' + stage_name)
    # add stage parameters to list
    if str(stage) not in model.stage_params:
        model.stage_params[str(stage)] = []
    num_params = 2 * num_convs + 1
    for idx in range(-num_params, 0):
        model.stage_params[str(stage)].append(model.weights[idx])
    # head convs don't have bias
    num_params = num_convs + 1
    for idx in range(-num_params, 0):
        model.stage_params[str(stage)].append(model.biases[idx])
    return 'fc6' + stage_name, fc_dim


def add_roi_Xconv1fc_gn_head_shared(model, head_stage, roi_stage, dim_in=None):
    """Add a X conv + 1fc head and output, with GroupNorm, sharing weights
    with another head."""
    add_stage_name = "_{}_{}".format(head_stage, roi_stage)
    shared_stage_name = ""
    if head_stage > 1:
        shared_stage_name = "_{}".format(head_stage)
    roi_feat = "roi_{}_feat".format(roi_stage)
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        add_name_prefix = 'head_conv' + str(i + 1) + add_stage_name
        shared_name_prefix = 'head_conv' + str(i + 1) + shared_stage_name
        current = model.ConvShared(
            current, add_name_prefix, dim_in, hidden_dim, 3,
            stride=1, pad=1,
            weight=shared_name_prefix + "_w",
            no_bias=1,
        )
        current = model.SpatialGNShared(
            add_name_prefix, add_name_prefix + "_gn",
            group_gn=get_group_gn(hidden_dim),
            scale=shared_name_prefix + "_gn_s",
            bias=shared_name_prefix + "_gn_b",
        )
        model.Relu(current, current)
        dim_in = hidden_dim

    model.FCShared(
        current,
        "fc6" + add_stage_name,
        weight="fc6" + shared_stage_name + "_w",
        bias="fc6" + shared_stage_name + "_b",
    )
    model.Relu("fc6" + add_stage_name, "fc6" + add_stage_name)

    model.FCShared(
        "fc6" + add_stage_name,
        "cls_score" + add_stage_name,
        weight="cls_score" + shared_stage_name + "_w",
        bias="cls_score" + shared_stage_name + "_b",
    )
    model.Softmax(
        "cls_score" + add_stage_name, "cls_prob" + add_stage_name, engine="CUDNN"
    )
    return "cls_prob" + add_stage_name


def add_ensemble_output(model, blobs_in, stage):
    """Add ops that ensembles detection heads."""
    output_name = "cls_prob_{}_sum".format(stage)
    model.net.Sum(blobs_in, output_name)
