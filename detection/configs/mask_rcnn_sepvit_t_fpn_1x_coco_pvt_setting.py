_base_ = [
    '../configs/_base_/models/mask_rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_instance.py',
    # '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]
model = dict(
    pretrained='pretrained/SepViT_Tiny.pth',
    backbone=dict(
        type='SepViT_Tiny',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5))
# optimizer
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12
