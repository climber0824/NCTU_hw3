from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from itertools import groupby
from pycocotools import mask as maskutil

from models.model import Yolact
from utils.augmentations import FastBaseTransform
from utils import timer
from models.output_utils import postprocess
from data.config import cfg

import torch
import torch.backends.cudnn as cudnn
import argparse
import json
from collections import defaultdict
import yaml

import cv2


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {}  # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})


def pred_sub(
    args,
    dets_out,
    img,
    h,
    w,
    undo_transform=True,
    class_color=False,
    mask_alpha=0.45,
    fps_str="",
):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    h, w, _ = img.shape

    with timer.env("Postprocess"):
        t = postprocess(
            dets_out,
            w,
            h,
            visualize_lincomb=args.display_lincomb,
            crop_masks=args.crop,
            score_threshold=args.score_threshold,
        )
        torch.cuda.synchronize()

    with timer.env("Copy"):
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][: args.top_k]
        classes, scores, boxes = [
            x[: args.top_k].cpu().detach().numpy() for x in t[:3]
        ]

    return classes, scores, masks


def binary_mask_to_rle(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")
    for i, (value, elements) in enumerate(
        groupby(binary_mask.ravel(order="F"))
    ):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(
        rle, rle.get("size")[0], rle.get("size")[1]
    )
    compressed_rle["counts"] = str(compressed_rle["counts"], encoding="utf-8")
    return compressed_rle


def evaluate(args, net, train_mode=False):
    net.eval()
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    coco_dt = []
    coco_test = COCO(args.test_json_name)
    for imgid in coco_test.imgs:
        image = cv2.imread(
            f"{args.data_path}" + coco_test.loadImgs(ids=imgid)[0]["file_name"]
        )
        frame = torch.from_numpy(image).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = net(batch)
        classes, scores, masks = pred_sub(
            args, preds, frame, None, None, undo_transform=False
        )
        num_dets_to_consider = min(args.top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < args.score_threshold:
                num_dets_to_consider = j
                break
        if num_dets_to_consider == 0:
            continue
        masks = masks[:num_dets_to_consider, :, :, None].squeeze(3)
        for pred_id in range(num_dets_to_consider):
            mask = masks[pred_id].cpu().detach().numpy()
            pred = {}
            pred["image_id"] = int(
                imgid
            )  # this imgid must be same as the key of test.json
            pred["category_id"] = int(classes[pred_id]) + 1
            pred["segmentation"] = binary_mask_to_rle(
                mask
            )  # save binary mask to RLE, e.g. 512x512 -> rle
            pred["score"] = float(scores[pred_id])
            coco_dt.append(pred)
    with open(f"{args.save_json_name}", "w") as f:
        json.dump(coco_dt, f)

    if args.calc_mAP:
        cocoGt = COCO(f"{args.gt_json_name}")
        cocoDt = cocoGt.loadRes(f"{args.save_json_name}")
        cocoEval = COCOeval(cocoGt, cocoDt, "segm")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=str)

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    for key, value in config["predict"].items():
        setattr(args, key, value)

    cudnn.fastest = True
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    print("Loading model...", end="")
    net = Yolact().cuda()
    net.load_weights(f"{args.weights_path}/{args.epoch}.pth")
    net.eval()

    evaluate(args, net)
