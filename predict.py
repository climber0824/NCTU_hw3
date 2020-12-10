from models.model import Yolact
from utils.augmentations import FastBaseTransform
from utils import timer
from models.output_utils import postprocess, undo_image_transformation


from data.config import cfg, COLORS

import torch
import torch.backends.cudnn as cudnn
import argparse
import os
from collections import defaultdict
from pathlib import Path
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


def prep_display(
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
    Note: If undo_transform=False then
    im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
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

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.0
                color_cache[on_gpu][color_idx] = color
            return color

    if (
        args.display_masks
        and cfg.eval_mask_branch
        and num_dets_to_consider > 0
    ):
        masks = masks[:num_dets_to_consider, :, :, None]

        colors = torch.cat(
            [
                get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3)
                for j in range(num_dets_to_consider)
            ],
            dim=0,
        )
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        inv_alph_masks = masks * (-mask_alpha) + 1

        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[
                : (num_dets_to_consider - 1)
            ].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    if args.display_fps:
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(
            fps_str, font_face, font_scale, font_thickness
        )[0]

        img_gpu[0: text_h + 8, 0: text_w + 8] *= 0.6

    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if args.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(
            img_numpy,
            fps_str,
            text_pt,
            font_face,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

    if num_dets_to_consider == 0:
        return img_numpy

    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = (
                    "%s: %.2f" % (_class, score)
                    if args.display_scores
                    else _class
                )

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(
                    text_str, font_face, font_scale, font_thickness
                )[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(
                    img_numpy,
                    (x1, y1),
                    (x1 + text_w, y1 - text_h - 4),
                    color,
                    -1,
                )
                cv2.putText(
                    img_numpy,
                    text_str,
                    text_pt,
                    font_face,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA,
                )

    return img_numpy


def evalimage(args, net: Yolact, path: str, save_path: str = None):
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    img_numpy = prep_display(
        args, preds, frame, None, None, undo_transform=False
    )
    cv2.imwrite(save_path, img_numpy)


def evalimages(net: Yolact, input_folder: str, output_folder: str):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for p in Path(input_folder).glob("*"):
        path = str(p)
        name = os.path.basename(path)
        name = ".".join(name.split(".")[:-1]) + ".png"
        out_path = os.path.join(output_folder, name)

        evalimage(args, net, path, out_path)
        print(path + " -> " + out_path)
    print("Done.")


def evaluate(args, net, train_mode=False):
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    evalimages(net, args.data_path, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=str)

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    for key, value in config["predict"].items():
        setattr(args, key, value)

    os.makedirs(args.save_path, exist_ok=True)

    cudnn.fastest = True
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    print("Loading model...", end="")
    net = Yolact().cuda()
    net.load_weights(f"{args.weights_path}/{args.epoch}.pth")
    net.eval()

    evaluate(args, net)
