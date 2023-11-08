_base_ = [
    '../slyformer/SB/slylab.b.512x512.80k.py',
    '../_base_/datasets/pascal_voc12.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k_adamw.py'
]
model = dict(
    decode_head=dict(num_classes=5), auxiliary_head=dict(num_classes=5))
