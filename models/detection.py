import torch

import os
import sys

import timer
from config import cfg
from box_utils import decode, jaccard

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../data"))
sys.path.append(os.path.join(BASE_DIR, "../utils"))


class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """

    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError("nms_threshold must be non negative.")
        self.conf_thresh = conf_thresh

        self.use_cross_class_nms = False
        self.use_fast_nms = False

    def __call__(self, predictions):
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            mask_data: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_data:
                (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order:
            class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        loc_data = predictions["loc"]
        conf_data = predictions["conf"]
        mask_data = predictions["mask"]
        prior_data = predictions["priors"]

        proto_data = predictions["proto"] if "proto" in predictions else None
        inst_data = predictions["inst"] if "inst" in predictions else None

        out = []

        with timer.env("Detect"):
            batch_size = loc_data.size(0)
            num_priors = prior_data.size(0)

            conf_preds = (
                conf_data.view(batch_size, num_priors, self.num_classes)
                .transpose(2, 1)
                .contiguous()
            )

            for batch_idx in range(batch_size):
                decoded_boxes = decode(loc_data[batch_idx], prior_data)
                result = self.detect(
                    batch_idx, conf_preds, decoded_boxes, mask_data, inst_data
                )

                if result is not None and proto_data is not None:
                    result["proto"] = proto_data[batch_idx]

                out.append(result)

        return out

    def detect(
        self, batch_idx, conf_preds, decoded_boxes, mask_data, inst_data
    ):
        """ Perform nms for only the max scoring class
        that isn't background (class 0) """
        cur_scores = conf_preds[batch_idx, 1:, :]
        conf_scores, _ = torch.max(cur_scores, dim=0)

        keep = conf_scores > self.conf_thresh
        scores = cur_scores[:, keep]
        boxes = decoded_boxes[keep, :]
        masks = mask_data[batch_idx, keep, :]

        if scores.size(1) == 0:
            return None

        if self.use_fast_nms:
            if self.use_cross_class_nms:
                boxes, masks, classes, scores = self.cc_fast_nms(
                    boxes, masks, scores, self.nms_thresh, self.top_k
                )
            else:
                boxes, masks, classes, scores = self.fast_nms(
                    boxes, masks, scores, self.nms_thresh, self.top_k
                )
        else:
            boxes, masks, classes, scores = self.fast_nms(
                boxes, masks, scores, self.nms_thresh, self.top_k
            )

            if self.use_cross_class_nms:
                print(
                    "Warning: Cross Class Traditional NMS is not implemented."
                )

        return {"box": boxes, "mask": masks, "class": classes, "score": scores}

    def cc_fast_nms(
        self,
        boxes,
        masks,
        scores,
        iou_threshold: float = 0.5,
        top_k: int = 200,
    ):
        # Collapse all the classes into 1
        scores, classes = scores.max(dim=0)

        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]

        boxes_idx = boxes[idx]

        iou = jaccard(boxes_idx, boxes_idx)

        iou.triu_(diagonal=1)

        iou_max, _ = torch.max(iou, dim=0)

        idx_out = idx[iou_max <= iou_threshold]

        return (
            boxes[idx_out],
            masks[idx_out],
            classes[idx_out],
            scores[idx_out],
        )

    def fast_nms(
        self,
        boxes,
        masks,
        scores,
        iou_threshold: float = 0.5,
        top_k: int = 200,
        second_threshold: bool = False,
    ):
        scores, idx = scores.sort(1, descending=True)

        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]

        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou = jaccard(boxes, boxes)
        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)

        keep = iou_max <= iou_threshold

        if second_threshold:
            keep *= scores > self.conf_thresh

        classes = torch.arange(num_classes)[:, None].cuda().expand_as(keep)
        classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]

        scores, idx = scores.sort(0, descending=True)
        idx = idx[: cfg.max_num_detections]
        scores = scores[: cfg.max_num_detections]

        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        return boxes, masks, classes, scores
