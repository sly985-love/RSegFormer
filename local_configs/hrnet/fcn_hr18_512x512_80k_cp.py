_base_ = [
    '../_base_/models/fcn_hr18.py',
    '../_base_/datasets/pascal_voc12_cp.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k_adamw.py'
]
model = dict(decode_head=dict(num_classes=5))
