_base_ = [
    "../../mmdetection/configs/_base_/datasets/coco_detection.py",
    "../../mmdetection/configs/_base_/schedules/schedule_1x.py",
    "../../mmdetection/configs/_base_/default_runtime.py",
]

model = dict(
    type="DDOD",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_output",
        num_outs=5,
    ),
    bbox_head=dict(
        type="DDODHead",
        num_classes=4,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
        loss_iou=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
    ),
    train_cfg=dict(
        # assigner is mean cls_assigner
        assigner=dict(type="ATSSAssigner", topk=9, alpha=0.8),
        reg_assigner=dict(type="ATSSAssigner", topk=9, alpha=0.5),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.6),
        max_per_img=100,
    ),
)

# Data
albu_train_transforms = [
    dict(type="RandomRotate90", p=0.25),
    dict(
        type="ShiftScaleRotate",
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=15,
        interpolation=1,
        p=0.5,
    ),
    dict(type="CLAHE", clip_limit=2, p=0.1),
    dict(type="HueSaturationValue", p=0.1),
    dict(type="RandomBrightnessContrast", p=0.2),
    dict(type="GaussNoise", p=0.1),
    dict(type="MotionBlur", blur_limit=3, p=0.2),
    dict(type="ISONoise", p=0.2),
    dict(type="ImageCompression", quality_lower=20, quality_upper=40, p=0.25),
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
img_scale = (1333, 800)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", img_scale=img_scale, keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5, direction=["horizontal", "vertical"]),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_labels"],
            min_visibility=0.0,
            filter_lost_elements=True,
        ),
        keymap={"img": "image", "gt_masks": "masks", "gt_bboxes": "bboxes"},
        update_pad_shape=False,
        skip_img_without_anno=True,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
classes = ("bottle_blue", "bottle_white", "bottle_brown", "plastic")
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        ann_file="data/annotation_train.json",
        img_prefix="data/images/",
        classes=classes,
        pipeline=train_pipeline,
    ),
    val=dict(
        img_prefix="data/images/",
        classes=classes,
        ann_file="data/annotation_val.json",
        pipeline=test_pipeline,
    ),
    test=dict(
        img_prefix="data/images/",
        classes=classes,
        ann_file="data/annotation_test.json",
        pipeline=test_pipeline,
    ),
    persistent_workers=True,
)
optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=250,
    warmup_ratio=0.001,
    step=[10, 20, 25],
)
runner = dict(type="EpochBasedRunner", max_epochs=30)
load_from = (
    "../mmdetection/checkpoints/ddod_r50_fpn_1x_coco_20220523_223737-29b2fc67.pth"
)
