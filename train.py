import torch
import torch.optim as optim

import argparse
import yaml
import os
import lera
import random

from data.dataloader import Detection, detection_collate, enforce_size
#from data.config import cfg, MEANS
from data.config import cfg, MEANS
from utils.augmentations import SSDAugmentation
from models.model import Yolact
from models.multibox_loss import MultiBoxLoss


def train(args):
    lera.log_hyperparams({"title": "hw4", "batch size": args.batch_size})
    # ----------------------------dataset--------------------------------
    dataset = Detection(
        image_path=cfg.dataset.train_images,
        info_file=cfg.dataset.train_info,
        transform=SSDAugmentation(MEANS),
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=detection_collate,
        pin_memory=True,
    )

    print(f"> load {len(dataset)} images to train")
    # ----------------------------model--------------------------------
    net = Yolact().cuda()
    if args.pre_epoch != 0:
        print(f"loading {args.pre_epoch} epoch model...")
        net.load_weights(f"{args.weights_path}/{args.pre_epoch}.pth")

    optimizer = optim.SGD(
        net.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.decay,
    )

    criterion = MultiBoxLoss(
        num_classes=cfg.num_classes,
        pos_threshold=cfg.positive_iou_threshold,
        neg_threshold=cfg.negative_iou_threshold,
        negpos_ratio=cfg.ohem_negpos_ratio,
    )

    # ----------------------------training--------------------------------
    for epoch in range(args.pre_epoch + 1, args.epochs + 1):
        for _iter, data in enumerate(data_loader, 0):
            net.train()
            optimizer.zero_grad()
            images, (targets, masks, num_crowds) = data
            for i in range(len(targets)):
                images[i].requires_grad = False
                images[i] = images[i].cuda()
                targets[i].requires_grad = False
                targets[i] = targets[i].cuda()
                masks[i].requires_grad = False
                masks[i] = masks[i].cuda()

            if cfg.preserve_aspect_ratio:
                # Choose a random size from the batch
                _, h, w = images[random.randint(0, len(images) - 1)].size()

                for idx, (image, target, mask, num_crowd) in enumerate(
                    zip(images, targets, masks, num_crowds)
                ):
                    images[idx], targets[idx], masks[idx], num_crowds[
                        idx
                    ] = enforce_size(image, target, mask, num_crowd, w, h)

            images = torch.stack(images, dim=0)

            preds = net(images)

            losses = criterion(preds, targets, masks, num_crowds)
            loss = (
                sum([losses[k] for k in losses])
                + 3 * losses["M"]
                + losses["C"]
            )
            loss.backward()
            optimizer.step()
            loss_b = losses["B"].item()
            loss_c = losses["C"].item()
            loss_m = losses["M"].item()
            loss_s = losses["S"].item()
            if _iter % args.log_interval == 0:
                lera.log(
                    {
                        "total loss": loss.item(),
                        "loss_b": loss_b,
                        "loss_c": loss_c,
                        "loss_m": loss_m,
                        "loss_s": loss_s,
                    }
                )
                print(
                    f"epoch:{epoch:04}/{args.epochs:04}",
                    f"|| iter:{_iter:03}/{len(data_loader):03}",
                    f"|| B: {loss_b:04f} || C: {loss_c:04f}",
                    f"|| M: {loss_m:04f} || S: {loss_s:04f}",
                    f"|| total loss: {loss.item():04f}",
                )
        if epoch % args.save_interval == 0:
            print("Saving state, epoch:", epoch)
            net.save_weights(f"{args.weights_path}/{epoch}.pth")
        # if epoch % args.mAP_interval == 0:
        #     evaluate(args, net)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=str)

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    for key, value in config["train"].items():
        setattr(args, key, value)

    os.makedirs(args.weights_path, exist_ok=True)

    train(args)
