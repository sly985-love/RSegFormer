_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py',
    '../../_base_/datasets/pascal_voc12.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_80k_adamw.py'
]
model = dict(test_cfg=dict(crop_size=(512, 512), stride=(170, 170)), decode_head=dict(num_classes=5))
evaluation = dict(interval=500, metric='mIoU')
