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

"""Construct minibatches for Cascade R-CNN training. Handles the minibatch blobs
that are specific to Cascade R-CNN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np

from detectron.core.config import cfg
import detectron.modeling.FPN as fpn
import detectron.roi_data.keypoint_rcnn as keypoint_rcnn_roi_data
import detectron.roi_data.mask_rcnn as mask_rcnn_roi_data
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils

logger = logging.getLogger(__name__)


def get_cascade_rcnn_blob_names(stage, is_training=True):
    """Cascade R-CNN blob names."""
    # rois blob: holds R regions of interest, each is a 5-tuple
    # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
    # rectangle (x1, y1, x2, y2)
    stage_name = "_{}".format(stage)
    blob_names = ["rois" + stage_name]
    if is_training:
        blob_names += ["labels_int32" + stage_name]
        blob_names += ["bbox_targets" + stage_name]
        blob_names += ["bbox_inside_weights" + stage_name]
        blob_names += ["bbox_outside_weights" + stage_name]
        blob_names += ["mapped_gt_boxes" + stage_name]
    if is_training and cfg.MODEL.MASK_ON and cfg.MRCNN.AT_STAGE == stage:
        # 'mask_rois': RoIs sampled for training the mask prediction branch.
        # Shape is (#masks, 5) in format (batch_idx, x1, y1, x2, y2).
        blob_names += ["mask_rois"]
        # 'roi_has_mask': binary labels for the RoIs specified in 'rois'
        # indicating if each RoI has a mask or not. Note that in some cases
        # a *bg* RoI will have an all -1 (ignore) mask associated with it in
        # the case that no fg RoIs can be sampled. Shape is (batchsize).
        blob_names += ["roi_has_mask_int32"]
        # 'masks_int32' holds binary masks for the RoIs specified in
        # 'mask_rois'. Shape is (#fg, M * M) where M is the ground truth
        # mask size.
        blob_names += ["masks_int32"]
    if is_training and cfg.MODEL.KEYPOINTS_ON and cfg.KRCNN.AT_STAGE == stage:
        # 'keypoint_rois': RoIs sampled for training the keypoint prediction
        # branch. Shape is (#instances, 5) in format (batch_idx, x1, y1, x2,
        # y2).
        blob_names += ['keypoint_rois']
        # 'keypoint_locations_int32': index of keypoint in
        # KRCNN.HEATMAP_SIZE**2 sized array. Shape is (#instances). Used in
        # SoftmaxWithLoss.
        blob_names += ['keypoint_locations_int32']
        # 'keypoint_weights': weight assigned to each target in
        # 'keypoint_locations_int32'. Shape is (#instances). Used in
        # SoftmaxWithLoss.
        blob_names += ['keypoint_weights']
        # 'keypoint_loss_normalizer': optional normalization factor to use if
        # cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS is False.
        blob_names += ['keypoint_loss_normalizer']
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        # Support for FPN multi-level rois without bbox reg isn't
        # implemented (... and may never be implemented)
        k_max = cfg.FPN.ROI_MAX_LEVEL
        k_min = cfg.FPN.ROI_MIN_LEVEL
        # Same format as rois blob, but one per FPN level
        for lvl in range(k_min, k_max + 1):
            blob_names += ["rois" + stage_name + "_fpn" + str(lvl)]
        blob_names += ["rois" + stage_name + "_idx_restore_int32"]
        if is_training:
            if cfg.MODEL.MASK_ON and cfg.MRCNN.AT_STAGE == stage:
                for lvl in range(k_min, k_max + 1):
                    blob_names += ["mask_rois_fpn" + str(lvl)]
                blob_names += ["mask_rois_idx_restore_int32"]
            if cfg.MODEL.KEYPOINTS_ON and cfg.KRCNN.AT_STAGE == stage:
                for lvl in range(k_min, k_max + 1):
                    blob_names += ['keypoint_rois_fpn' + str(lvl)]
                blob_names += ['keypoint_rois_idx_restore_int32']
    return blob_names


def add_cascade_rcnn_blobs(blobs, im_scales, roidb, stage):
    """Add blobs needed for training Cascade R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    for im_i, entry in enumerate(roidb):
        frcn_blobs = _sample_rois(entry, im_scales[im_i], im_i, stage)
        for k, v in frcn_blobs.items():
            if 'mask' not in k and 'keypoint' not in k:
                k += "_{}".format(stage)
            blobs[k].append(v)
    # Concat the training blob lists into tensors
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)
    # Add FPN multilevel training RoIs, if configured
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois(blobs, stage)

    # Perform any final work and validity checks after the collating blobs for
    # all minibatch images
    valid = True
    if cfg.MODEL.KEYPOINTS_ON and cfg.KRCNN.AT_STAGE == stage:
        valid = keypoint_rcnn_roi_data.finalize_keypoint_minibatch(blobs, valid)

    return valid


def _sample_rois(roidb, im_scale, batch_idx, stage):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    fg_thresh = cfg.CASCADE_RCNN.FG_THRESHS[stage - 1]
    bg_thresh_hi = cfg.CASCADE_RCNN.BG_THRESHS_HI[stage - 1]
    bg_thresh_lo = cfg.CASCADE_RCNN.BG_THRESHS_LO[stage - 1]

    max_overlaps = roidb["max_overlaps"]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= fg_thresh)[0]
    fg_rois_per_this_image = fg_inds.size

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where(
        (max_overlaps < bg_thresh_hi) & (max_overlaps >= bg_thresh_lo)
    )[0]

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Label is the class each RoI has max overlap with
    sampled_labels = roidb["max_classes"][keep_inds]
    sampled_labels[fg_rois_per_this_image:] = 0  # Label bg RoIs with class 0
    sampled_boxes = roidb["boxes"][keep_inds]

    gt_inds = np.where(roidb["gt_classes"] > 0)[0]
    gt_boxes = roidb["boxes"][gt_inds, :]
    gt_assignments = gt_inds[roidb["box_to_gt_ind_map"][keep_inds]]

    # [mapped_gt_boxes, max_overlaps]
    mapped_gt_boxes = blob_utils.zeros((keep_inds.size, 5))
    mapped_gt_boxes[:, :4] = gt_boxes[gt_assignments, :] * im_scale
    mapped_gt_boxes[:, 4] = max_overlaps[keep_inds]
    mapped_gt_boxes[fg_rois_per_this_image:, :] = 0

    if "bbox_targets" not in roidb:
        bbox_targets = _compute_targets(
            sampled_boxes, gt_boxes[gt_assignments, :], sampled_labels, stage
        )
    else:
        bbox_targets = roidb['bbox_targets'][keep_inds, :]

    bbox_targets, bbox_inside_weights = _expand_bbox_targets(bbox_targets)
    bbox_outside_weights = np.array(
        bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype
    )

    # Scale rois and format as (batch_idx, x1, y1, x2, y2)
    sampled_rois = sampled_boxes * im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((sampled_rois.shape[0], 1))
    sampled_rois = np.hstack((repeated_batch_idx, sampled_rois))

    # Base Cascade R-CNN blobs
    blob_dict = dict(
        labels_int32=sampled_labels.astype(np.int32, copy=False),
        rois=sampled_rois,
        bbox_targets=bbox_targets,
        bbox_inside_weights=bbox_inside_weights,
        bbox_outside_weights=bbox_outside_weights,
        mapped_gt_boxes=mapped_gt_boxes,
    )

    # Optionally add Mask R-CNN blobs
    if cfg.MODEL.MASK_ON and cfg.MRCNN.AT_STAGE == stage:
        mask_rcnn_roi_data.add_mask_rcnn_blobs(
            blob_dict, sampled_boxes, roidb, im_scale, batch_idx
        )

    # Optionally add Keypoint R-CNN blobs
    if cfg.MODEL.KEYPOINTS_ON and cfg.KRCNN.AT_STAGE == stage:
        keypoint_rcnn_roi_data.add_keypoint_rcnn_blobs(
            blob_dict, roidb, fg_rois_per_this_image,
            fg_inds, im_scale, batch_idx, fg_thresh
        )

    return blob_dict


def _compute_targets(ex_rois, gt_rois, labels, stage):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    bbox_reg_weights = cfg.CASCADE_RCNN.BBOX_REG_WEIGHTS[stage - 1]
    targets = box_utils.bbox_transform_inv(ex_rois, gt_rois, bbox_reg_weights)
    return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _expand_bbox_targets(bbox_target_data):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    num_bbox_reg_classes = cfg.MODEL.NUM_CLASSES
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        num_bbox_reg_classes = 2  # bg and fg

    clss = bbox_target_data[:, 0]
    bbox_targets = blob_utils.zeros((clss.size, 4 * num_bbox_reg_classes))
    bbox_inside_weights = blob_utils.zeros(bbox_targets.shape)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind]) if not cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else 1
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights


def _add_multilevel_rois(blobs, stage):
    """By default training RoIs are added for a single feature map level only.
    When using FPN, the RoIs must be distributed over different FPN levels
    according the level assignment heuristic (see: modeling.FPN.
    map_rois_to_fpn_levels).
    """
    stage_name = "_{}".format(stage)
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL

    def _distribute_rois_over_fpn_levels(rois_blob_name):
        """Distribute rois over the different FPN levels."""
        # Get target level for each roi
        # Recall blob rois are in (batch_idx, x1, y1, x2, y2) format, hence take
        # the box coordinates from columns 1:5
        target_lvls = fpn.map_rois_to_fpn_levels(
            blobs[rois_blob_name][:, 1:5], lvl_min, lvl_max
        )
        # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
        fpn.add_multilevel_roi_blobs(
            blobs, rois_blob_name, blobs[rois_blob_name], target_lvls, lvl_min, lvl_max
        )

    _distribute_rois_over_fpn_levels("rois" + stage_name)
    if cfg.MODEL.MASK_ON and cfg.MRCNN.AT_STAGE == stage:
        _distribute_rois_over_fpn_levels("mask_rois")
    if cfg.MODEL.KEYPOINTS_ON and cfg.KRCNN.AT_STAGE == stage:
        _distribute_rois_over_fpn_levels('keypoint_rois')
