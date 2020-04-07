import re
import collections
import numpy as np
from operator import itemgetter
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(message)s')
# logging.basicConfig(filename='/mnt/workspace/yonathan/log_infer.txt', level=logging.INFO, format='%(message)s')
# logging.getLogger().addHandler(logging.StreamHandler())
# import torch
# import torch.nn as nn
# import time

#
# list_batch_size = [128, 256, 512, 1024, 2048]
# list_channels = [128, 256, 512, 1024, 2048]
# list_kernel = [1, 3, 5, 7]
# list_w = [7, 14, 28, 56, 112, 224]
# matrix = np.zeros((len(list_batch_size), len(list_channels), len(list_kernel), len(list_w)))
# logging.info(np.prod(matrix.size))
#
# synchronize = True
#
# if synchronize:
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#
# for i1, bs in enumerate(list_batch_size):
#     for i2, ch in enumerate(list_channels):
#         for i3, k in enumerate(list_kernel):
#             for i4, W in enumerate(list_w):
#                 mod = nn.Conv2d(ch, ch, k, bias=False)
#                 mod = mod.cuda(device=0)
#                 mod.eval()
#                 with torch.no_grad():
#                     try:
#                         l = torch.randn(bs, ch, W, W, requires_grad=False).cuda(device=0)
#                         temp = mod(l)
#                     except:
#                         logging.info(
#                             f"Out of memory for batch size: {bs}, number of channels {ch}, kernel size {k} and input dim {W}")
#                         torch.cuda.empty_cache()
#                         continue
#                     if synchronize:
#                         start.record()
#                     else:
#                         start = time.time()
#                     try:
#                         temp2 = mod(l)
#                     except:
#                         logging.info(
#                             f"Out of memory for batch size: {bs}, number of channels {ch}, kernel size {k} and input dim {W}")
#                         logging.info("*************************************************************")
#                         torch.cuda.empty_cache()
#                         continue
#                     if synchronize:
#                         end.record()
#                         torch.cuda.synchronize()
#                         elapsed = start.elapsed_time(end)
#                     else:
#                         elapsed = time.time() - start
#                     matrix[i1, i2, i3, i4] = elapsed
#                     logging.info(f"batch size: {bs}, number of channels {ch}, kernel size {k} and input dim {W}")
#                     logging.info(
#                         f"MACS:{(W*k*ch)**2} time: elapsed: {elapsed} ms. Ratio = {(W*k*ch)**2/(1e6*elapsed)} Gmacs per second")
#                     logging.info("*************************************************************")
#                     torch.cuda.empty_cache()

# file = open("/mnt/workspace/yonathan/infer.txt", "r")
file = open("/Users/yonathan/Desktop/infer.txt", "r")

lst = file.read().split('\n')
file.close()


def parse_file(lst):
    i = 1
    dct = {}
    while i < len(lst):
        o1 = lst[i]
        if o1.startswith('batch'):
            o2 = lst[i + 1]
            i += 1
            mtch1 = 'batch size: (\d+), number of channels (\d+), kernel size (\d+) and input dim (\d+)'
            mtch2 = 'MACS:(\d+) time: elapsed: (\d+\.\d+) ms. Ratio = (\d+\.\d+) (\D+)'
            x1 = re.match(mtch1, o1)
            x2 = re.match(mtch2, o2)
            data1 = list(x1.groups())
            data2 = list(x2.groups())[:-1]
            batch_size = int(data1[0])
            ch = int(data1[1])
            k = int(data1[2])
            dim = int(data1[3])
            total_macs = int(data2[0])*batch_size
            time = float(data2[1])
            r = batch_size*(ch*dim*k)**2/(1e6*time)
            key = total_macs
            if key in dct.keys():
                dct[key].append((batch_size, ch, k, dim, time, r))
            else:
                dct[key] = [(batch_size, ch, k, dim, time, r)]

        i += 1
    return dct


dct = parse_file(lst)
dct = collections.OrderedDict(sorted(dct.items()))

fig, ax = plt.subplots()
index_x=[]
index_y=[]
for k, v in dct.items():
    lst_ratio = [x[-1] for x in v]
    if len(v) > 1:
        sorted_index = list(np.argsort(lst_ratio)[::-1])
        dct[k] = list(itemgetter(*sorted_index)(v))
    v = dct[k]
    logging.info(f'{k} : {[x for x in v]}')
    for x in v:
        index_x.append(k)
        index_y.append(x[-2])

x=np.array(index_x)
y=np.array(index_y)


ax.scatter(x, y, marker='.', s=1)
# fig = plt.gcf()
# fig.set_size_inches(20, 20)
# plt.show()
plt.savefig('./temp.png', dpi=200)