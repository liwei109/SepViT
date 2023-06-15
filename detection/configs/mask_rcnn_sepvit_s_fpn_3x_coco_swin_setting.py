_base_ = './mask_rcnn_sepvit_t_fpn_3x_coco_swin_setting.py'

model = dict(
    pretrained='pretrained/SepViT_Small.pth',
    backbone=dict(
        _delete_=True,
        type='SepViT_Small',
        style='pytorch'),
    neck=dict(
        in_channels=[96, 192, 384, 768],
        out_channels=256))
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.1,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
