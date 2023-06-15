# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import torch
from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmseg.models import build_segmentor

import sepvit

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', default='configs/fpn_sepvit_t_512x512_80k_ade20k.py', help='train config file path')
    parser.add_argument(
        '--shape', 
        type=int,
        nargs='+',
        default=[2048, 512],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')

    # model_name = 'fpn_80k_tiny'
    # onnx_path = "../onnx/segmentation/%s_%sG_%sM.onnx" % (model_name, flops[:3], params[:4])
    # input_shape = (8, 3, 512, 512)
    # input = torch.ones(input_shape, dtype=torch.float32).cuda()
    # model.eval()
    # torch.onnx.export(model, input, onnx_path, opset_version=11)


if __name__ == '__main__':
    main()