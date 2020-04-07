from timm.models import create_model
from timm.data import resolve_data_config
import torch
import torch.nn as nn
import argparse
from external.utils_pruning import compute_flops, load_module_from_ckpt, measure_time
from external.hyperml import HypermlDownloader
from timm.utils import setup_default_logging
from timm.models import load_checkpoint

import logging

setup_default_logging()
parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--model', '-m', metavar='MODEL', default='efficientnet_b2_pruned',
                    help='model architecture (default: efficientnet_b2_pruned)')

parser.add_argument('--batch_size', default=512, type=int,
                    help='batch size)')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use half precision (fp16)')
parser.add_argument('--use_eca', action='store_true', default=False,
                    help='Use eca bn for efficientNet')
parser.add_argument("--local_rank", default=0, type=int)

args = parser.parse_args()

model_name = args.model
batch_size = args.batch_size
model = create_model(
    model_name,
    num_classes=1000,
    in_chans=3,
    pretrained=False)


data_config = resolve_data_config(vars(args), model=model)

flops, flops_per_conv = compute_flops(model, data_config['input_size'], flops_conv=True)
size = data_config['input_size'][1]
print(f"Model {model_name} has input size of {size} and {flops/1e9} GFlops")

total_time, dict_time = measure_time(model, input_size=data_config['input_size'][1], batch_size=batch_size,
                                     iterations=20, fp16=args.fp16)
print(f"Inference speed is {batch_size / total_time} images/second")
print(f"Inference took: {total_time}")

dict_layer_macs = {}
for n, p in model.named_modules():
    if (len(n.split('.')) == 2 and n[:5] != 'conv1' and n.split('.')[0] != 'global_pool') or n == 'conv1':
        dict_layer_macs[n] = 0

total_time2 = 0
total_GF = 0
for n, p in model.named_modules():
    if isinstance(p, nn.Conv2d) or isinstance(p,
                                              nn.Conv1d):  # or isinstance(p, nn.BatchNorm2d) or isinstance(p, nn.ReLU) or isinstance(p,
        # nn.AdaptiveAvgPool2d) or isinstance(p, nn.Linear) or isinstance(p, nn.AdaptiveMaxPool2d):
        v = dict_time[n]
        if n.split('.')[0] in dict_layer_macs:
            dict_layer_macs[n.split('.')[0]] += flops_per_conv[n] / 1e9

        if '.'.join(n.split('.')[:2]) in dict_layer_macs:
            dict_layer_macs['.'.join(n.split('.')[:2])] += flops_per_conv[n] / 1e9
        if n in flops_per_conv.keys():
            Gf = flops_per_conv[n] / 1e9
            total_GF += Gf
            print(f'Module {n} has {Gf} GFLOPS and took {1e3*v} ms to run. Ratio time/GFLOPS is {v/Gf}')

        else:
            print(f'Module {n} took {v} seconds to run')
        total_time2 += v

print(f"time accumulated: {total_time2}")

for n, p in model.named_modules():
    if n in dict_layer_macs:
        v = dict_time[n]
        print(f'Module {n} took {1e3*v} ms to run and has {dict_layer_macs[n]} GFLOPS')
