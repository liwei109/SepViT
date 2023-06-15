_base_ = './mask_rcnn_sepvit_t_fpn_1x_coco_pvt_setting.py'

model = dict(
    pretrained='pretrained/SepViT_Small.pth',
    backbone=dict(
        type='SepViT_Small',
        style='pytorch'),
    neck=dict(
        in_channels=[96, 192, 384, 768],
        out_channels=256,))
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.0001)
