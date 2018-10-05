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

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

from detectron.core.config import cfg
import detectron.utils.boxes as box_utils


class BBoxAccuracyOp(object):
    """Output bbox prediction IoU accuracy, by Zhaowei Cai.
    """

    def __init__(self, bbox_reg_weights):
        self._bbox_reg_weights = bbox_reg_weights

    def forward(self, inputs, outputs):
        """See modeling.detector.AddBBoxAccuracy for inputs/outputs
        documentation.
        """

        # predicted bbox deltas
        bbox_deltas = inputs[0].data
        # proposals
        bbox_data = inputs[1].data
        assert bbox_data.shape[1] == 5
        bbox_prior = bbox_data[:, 1:]
        # labels
        labels = inputs[2].data
        # mapped gt boxes
        mapped_gt_boxes = inputs[3].data
        gt_boxes = mapped_gt_boxes[:, :4]
        max_overlap = mapped_gt_boxes[:, 4]

        # bbox iou only for fg and non-gt boxes
        keep_inds = np.where((labels > 0) & (max_overlap < 1.0))[0]
        num_boxes = keep_inds.size
        bbox_deltas = bbox_deltas[keep_inds, :]
        bbox_prior = bbox_prior[keep_inds, :]
        labels = labels[keep_inds]
        gt_boxes = gt_boxes[keep_inds, :]
        max_overlap = max_overlap[keep_inds]

        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG or num_boxes == 0:
            bbox_deltas = bbox_deltas[:, -4:]
        else:
            bbox_deltas = np.vstack(
                [
                    bbox_deltas[i, labels[i] * 4: labels[i] * 4 + 4]
                    for i in range(num_boxes)
                ]
            )
        pred_boxes = box_utils.bbox_transform(
            bbox_prior, bbox_deltas, self._bbox_reg_weights
        )

        avg_iou = 0.
        pre_avg_iou = sum(max_overlap)
        for i in range(num_boxes):
            gt_box = gt_boxes[i, :]
            pred_box = pred_boxes[i, :]
            tmp_iou = box_utils.bbox_overlaps(
                gt_box[np.newaxis, :].astype(dtype=np.float32, copy=False),
                pred_box[np.newaxis, :].astype(dtype=np.float32, copy=False),
            )
            avg_iou += tmp_iou[0]
        if num_boxes > 0:
            avg_iou /= num_boxes
            pre_avg_iou /= num_boxes
        outputs[0].reshape([1])
        outputs[0].data[...] = avg_iou
        outputs[1].reshape([1])
        outputs[1].data[...] = pre_avg_iou
