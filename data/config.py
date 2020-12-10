import torch
#from backbone import ResNetBackbone, ResNetBackboneGN

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../models"))

from backbone import ResNetBackbone, ResNetBackboneGN

# for making bounding boxes pretty
COLORS = (
    (244, 67, 54),
    (233, 30, 99),
    (156, 39, 176),
    (103, 58, 183),
    (63, 81, 181),
    (33, 150, 243),
    (3, 169, 244),
    (0, 188, 212),
    (0, 150, 136),
    (76, 175, 80),
    (139, 195, 74),
    (205, 220, 57),
    (255, 235, 59),
    (255, 193, 7),
    (255, 152, 0),
    (255, 87, 34),
    (121, 85, 72),
    (158, 158, 158),
    (96, 125, 139),
)


# These are in BGR and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)

VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

# ----------------------- CONFIG CLASS ----------------------- #


class Config(object):
    """
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = Config(vars(self))

        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def print(self):
        for k, v in vars(self).items():
            print(k, " = ", v)


# ----------------------- DATASETS ----------------------- #

dataset_base = Config(
    {
        "name": "Base Dataset",
        "train_images": "./data/coco/images/",
        "train_info": "path_to_annotation_file",
        "valid_images": "./data/coco/images/",
        "valid_info": "path_to_annotation_file",
        "has_gt": True,
        "class_names": VOC_CLASSES,
        "label_map": None,
    }
)

training_dataset = dataset_base.copy(
    {
        "name": "train",
        "train_images": "./dataset/train_images",
        "valid_images": "./dataset/test_images",
        "train_info": "./dataset/pascal_train.json",
        "valid_info": "./dataset/test.json",
    }
)

testing_dataset = dataset_base.copy(
    {
        "name": "test",
        "valid_images": "./dataset/test_images",
        "valid_info": "./dataset/test.json",
    }
)


# ----------------------- TRANSFORMS ----------------------- #

resnet_transform = Config(
    {
        "channel_order": "RGB",
        "normalize": True,
        "subtract_means": False,
        "to_float": False,
    }
)

# ----------------------- BACKBONES ----------------------- #

backbone_base = Config(
    {
        "name": "Base Backbone",
        "path": "path/to/pretrained/weights",
        "type": object,
        "args": tuple(),
        "transform": resnet_transform,
        "selected_layers": list(),
        "pred_scales": list(),
        "pred_aspect_ratios": list(),
        "use_pixel_scales": False,
        "preapply_sqrt": True,
        "use_square_anchors": False,
    }
)

resnet101_backbone = backbone_base.copy(
    {
        "name": "ResNet101",
        "path": "resnet101_reducedfc.pth",
        "type": ResNetBackbone,
        "args": ([3, 4, 23, 3],),
        "transform": resnet_transform,
        "selected_layers": list(range(2, 8)),
        "pred_scales": [[1]] * 6,
        "pred_aspect_ratios": [
            [[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]]
        ]
        * 6,
    }
)

resnet101_gn_backbone = backbone_base.copy(
    {
        "name": "ResNet101_GN",
        "path": "R-101-GN.pkl",
        "type": ResNetBackboneGN,
        "args": ([3, 4, 23, 3],),
        "transform": resnet_transform,
        "selected_layers": list(range(2, 8)),
        "pred_scales": [[1]] * 6,
        "pred_aspect_ratios": [
            [[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]]
        ]
        * 6,
    }
)


# ----------------------- MASK BRANCH TYPES ----------------------- #

mask_type = Config({"direct": 0, "lincomb": 1})

# ----------------------- ACTIVATION FUNCTIONS ----------------------- #

activation_func = Config(
    {
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "softmax": lambda x: torch.nn.functional.softmax(x, dim=-1),
        "relu": lambda x: torch.nn.functional.relu(x, inplace=True),
        "none": lambda x: x,
    }
)


# ----------------------- FPN DEFAULTS ----------------------- #

fpn_base = Config(
    {
        "num_features": 256,
        "interpolation_mode": "bilinear",
        "num_downsample": 1,
        "use_conv_downsample": False,
        "pad": True,
        "relu_downsample_layers": False,
        "relu_pred_layers": True,
    }
)


# ----------------------- CONFIG DEFAULTS ----------------------- #

base_config = Config(
    {
        "dataset": training_dataset,
        "num_classes": 21,  # This should include the background class
        "max_iter": 400000,
        "max_num_detections": 100,
        "lr": 0.0001,
        "momentum": 0.9,
        "decay": 5e-4,
        "gamma": 0.1,
        "lr_steps": (280000, 360000, 400000),
        "lr_warmup_init": 1e-4,
        "lr_warmup_until": 500,
        "conf_alpha": 1,
        "bbox_alpha": 1.5,
        "mask_alpha": 0.4 / 256 * 140 * 140,
        "eval_mask_branch": True,
        "nms_top_k": 200,
        "nms_conf_thresh": 0.05,
        "nms_thresh": 0.5,
        "mask_type": mask_type.direct,
        "mask_size": 16,
        "masks_to_train": 100,
        "mask_proto_src": None,
        "mask_proto_net": [(256, 3, {}), (256, 3, {})],
        "mask_proto_bias": False,
        "mask_proto_prototype_activation": activation_func.relu,
        "mask_proto_mask_activation": activation_func.sigmoid,
        "mask_proto_coeff_activation": activation_func.tanh,
        "mask_proto_crop": True,
        "mask_proto_crop_expand": 0,
        "mask_proto_loss": None,
        "mask_proto_binarize_downsampled_gt": True,
        "mask_proto_normalize_mask_loss_by_sqrt_area": False,
        "mask_proto_reweight_mask_loss": False,
        "mask_proto_grid_file": "data/grid.npy",
        "mask_proto_use_grid": False,
        "mask_proto_coeff_gate": False,
        "mask_proto_prototypes_as_features": False,
        "mask_proto_prototypes_as_features_no_grad": False,
        "mask_proto_remove_empty_masks": False,
        "mask_proto_reweight_coeff": 1,
        "mask_proto_coeff_diversity_loss": False,
        "mask_proto_coeff_diversity_alpha": 1,
        "mask_proto_normalize_emulate_roi_pooling": False,
        "mask_proto_double_loss": False,
        "mask_proto_double_loss_alpha": 1,
        "mask_proto_split_prototypes_by_head": False,
        "mask_proto_crop_with_pred_box": False,
        "augment_photometric_distort": True,
        "augment_expand": True,
        "augment_random_sample_crop": True,
        "augment_random_mirror": True,
        "augment_random_flip": False,
        "augment_random_rot90": False,
        "discard_box_width": 4 / 550,
        "discard_box_height": 4 / 550,
        "freeze_bn": False,
        "fpn": None,
        "share_prediction_module": False,
        "ohem_use_most_confident": False,
        "use_focal_loss": False,
        "focal_loss_alpha": 0.25,
        "focal_loss_gamma": 2,
        "focal_loss_init_pi": 0.01,
        "use_class_balanced_conf": False,
        "use_sigmoid_focal_loss": False,
        "use_objectness_score": False,
        "use_class_existence_loss": False,
        "class_existence_alpha": 1,
        "use_semantic_segmentation_loss": False,
        "semantic_segmentation_alpha": 1,
        "use_mask_scoring": False,
        "mask_scoring_alpha": 1,
        "use_change_matching": False,
        "extra_head_net": None,
        "head_layer_params": {"kernel_size": 3, "padding": 1},
        "extra_layers": (0, 0, 0),
        "positive_iou_threshold": 0.5,
        "negative_iou_threshold": 0.5,
        "ohem_negpos_ratio": 3,
        "crowd_iou_threshold": 1,
        "mask_dim": None,
        "max_size": 300,
        "force_cpu_nms": True,
        "use_coeff_nms": False,
        "use_instance_coeff": False,
        "num_instance_coeffs": 64,
        "train_masks": True,
        "train_boxes": True,
        "use_gt_bboxes": False,
        "preserve_aspect_ratio": False,
        "use_prediction_module": False,
        "use_yolo_regressors": False,
        "use_prediction_matching": False,
        "delayed_settings": [],
        "no_jit": False,
        "backbone": None,
        "name": "base_config",
    }
)


# ----------------------- YOLACT v1.0 CONFIGS ----------------------- #

train_base_config = base_config.copy(
    {
        "name": "train_base",
        # Dataset stuff
        "dataset": training_dataset,
        "num_classes": len(training_dataset.class_names) + 1,
        # Image Size
        "max_size": 550,
        # Training params
        "lr_steps": (280000, 600000, 700000, 750000),
        "max_iter": 800000,
        # Backbone Settings
        "backbone": resnet101_backbone.copy(
            {
                "selected_layers": list(range(1, 4)),
                "use_pixel_scales": True,
                "preapply_sqrt": False,
                "use_square_anchors": True,
                "pred_aspect_ratios": [[[1, 1 / 2, 2]]] * 5,
                "pred_scales": [[24], [48], [96], [192], [384]],
            }
        ),
        # FPN Settings
        "fpn": fpn_base.copy(
            {"use_conv_downsample": True, "num_downsample": 2}
        ),
        # Mask Settings
        "mask_type": mask_type.lincomb,
        "mask_alpha": 6.125,
        "mask_proto_src": 0,
        "mask_proto_net": [(256, 3, {"padding": 1})] * 3
        + [(None, -2, {}), (256, 3, {"padding": 1})]
        + [(32, 1, {})],
        "mask_proto_normalize_emulate_roi_pooling": True,
        # Other stuff
        "share_prediction_module": True,
        "extra_head_net": [(256, 3, {"padding": 1})],
        "positive_iou_threshold": 0.5,
        "negative_iou_threshold": 0.4,
        "crowd_iou_threshold": 0.7,
        "use_semantic_segmentation_loss": True,
    }
)


# Default config
cfg = train_base_config.copy()


def set_cfg(config_name: str):
    """ Sets the active config. Works even if cfg is already imported! """
    global cfg

    cfg.replace(eval(config_name))

    if cfg.name is None:
        cfg.name = config_name.split("_config")[0]


def set_dataset(dataset_name: str):
    """ Sets the dataset of the current config. """
    cfg.dataset = eval(dataset_name)
