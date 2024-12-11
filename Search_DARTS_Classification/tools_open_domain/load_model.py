import torch
from ptflops import get_model_complexity_info

model = torch.load('model.pth.tar')
# print(model)
with torch.cuda.device(0):
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))