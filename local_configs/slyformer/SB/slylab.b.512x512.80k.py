_base_ = [
    # '../../_base_/models/slyformer.py',
    '../../_base_/datasets/pascal_voc12_real.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_80k_adamw.py'
]


# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    # pretrained='open-mmlab://resnet50_v1c',
    # pretrained='open-mmlab://contrib/mobilenet_v3_large',
    pretrained='open-mmlab://msra/hrnetv2_w18',
    backbone=dict(
        type='shunted_b',
    ),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
