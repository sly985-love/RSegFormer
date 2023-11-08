_base_ = [
    '../_base_/models/pspnet_r50-d8.py',
    '../../_base_/datasets/pascal_voc12_cp.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_80k_adamw.py'
]

model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
        num_classes=5
    ),
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=5))

