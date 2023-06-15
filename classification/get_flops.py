import torch
from timm.models import create_model
from fvcore.nn import FlopCountAnalysis, parameter_count

import sepvit

model_name = 'SepViT_Base'
model = create_model(model_name, num_classes=1000)
model = model.cuda()

input = torch.randn(1, 3, 224, 224)
input = input.cuda()
flops = FlopCountAnalysis(model, input)
params = parameter_count(model)
flops, params = flops.total(), params['']
print(flops / 1e9,  params / 1e6)


# onnx_path = "onnx/%s_%.1fG_%.1fM.onnx" % (model_name, flops/1e9, params/1e6)
# print(onnx_path)
# input_shape = (8, 3, 224, 224)
# input = torch.ones(input_shape, dtype=torch.float32).cuda()
# model.eval()
# torch.onnx.export(model, input, onnx_path, opset_version=11)
