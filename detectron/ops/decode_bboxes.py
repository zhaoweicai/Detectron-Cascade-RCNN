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


class DecodeBBoxesOp(object):
    """Output predicted bbox, by Zhaowei Cai for Cascade R-CNN.
    """

    def __init__(self, bbox_reg_weights):
        self._bbox_reg_weights = bbox_reg_weights

    def forward(self, inputs, outputs):
        """See modeling.detector.DecodeBBoxes for inputs/outputs
        documentation.
        """

        bbox_deltas = inputs[0].data
        assert cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
        assert bbox_deltas.shape[1] == 8
        bbox_deltas = bbox_deltas[:, -4:]
        bbox_data = inputs[1].data
        assert bbox_data.shape[1] == 5
        batch_inds = bbox_data[:, :1]
        bbox_prior = bbox_data[:, 1:]

        # Transform bbox priors into proposals via bbox transformations
        bbox_decode = box_utils.bbox_transform(
            bbox_prior, bbox_deltas, self._bbox_reg_weights
        )

        # remove mal-boxes with non-positive width or height and ground
        # truth boxes during training
        if len(inputs) > 2:
            mapped_gt_boxes = inputs[2].data
            max_overlap = mapped_gt_boxes[:, 4]
            keep = _filter_boxes(bbox_decode, max_overlap)
            bbox_decode = bbox_decode[keep, :]
            batch_inds = batch_inds[keep, :]

        bbox_decode = np.hstack((batch_inds, bbox_decode))
        outputs[0].reshape(bbox_decode.shape)
        outputs[0].data[...] = bbox_decode


def _filter_boxes(boxes, max_overlap):
    """Only keep boxes with positive height and width, and not-gt.
    """
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws > 0) & (hs > 0) & (max_overlap < 1.0))[0]
    return keep
