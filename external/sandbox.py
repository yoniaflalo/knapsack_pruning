from external.utils_pruning import *
from timm.models import create_model

# torch.backends.cudnn.enabled = False
# bs = 512
# mod1 = nn.Conv2d(3, 64, 7, stride=2)
# mod1 = mod1.cuda(device=0).eval()
# mod2 = nn.Conv2d(64, 128, 3)
# mod2 = mod2.cuda(device=0).eval()
# num_try = 100
# total_time = 0
# torch.cuda.empty_cache()
# total_time1 = 0
# with torch.no_grad():
#     l = torch.randn(bs, 3, 224, 224, requires_grad=False).cuda(device=0)
#     mod1(l)
#     for i in range(num_try):
#         start = time.time()
#         a = mod1(l)
#         torch.cuda.synchronize()
#         end = time.time()
#         total_time1 += end - start
#     total_time1 /= num_try
#
# total_time2 = 0
#
# torch.cuda.empty_cache()
#
# with torch.no_grad():
#     mod2(a)
#     for i in range(num_try):
#         start = time.time()
#         b = mod2(a)
#         torch.cuda.synchronize()
#         end = time.time()
#         total_time2 += end - start
#     total_time2 /= num_try
# start_all = time.time()
# end_all = time.time()
#
# print(total_time1 + total_time2)
#
# torch.cuda.empty_cache()
#
# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)
#
# with torch.no_grad():
#     l = torch.randn(bs, 3, 224, 224, requires_grad=False).cuda(device=0)
#     mod2(mod1(l))
#     start_t = time.time()
#     for i in range(num_try):
#         start.record()
#         a = mod2(mod1(l))
#         end.record()
#         torch.cuda.synchronize()
#         total_time += start.elapsed_time(end) / 1000.0
#     total_time /= num_try
# end_t = time.time()
# torch.cuda.synchronize()
# print(total_time)
# print(f'***{(end_t-start_t)/num_try}')
# #
# # torch.cuda.empty_cache()
# # bs = 256
# # mod1 = nn.Conv2d(3, 128, 7, stride=2)
# # mod1 = mod1.cuda(device=0).eval()
# # mod2 = nn.Conv2d(128, 64, 3)
# # mod2 = mod2.cuda(device=0).eval()
# num_try = 100
# total_time = 0
# with torch.no_grad():
#     l = torch.randn(bs, 3, 224, 224, requires_grad=False).cuda(device=0)
#     mod2(mod1(l))
#     for i in range(num_try):
#         start = time.time()
#         a = mod1(l)
#         b = mod2(a)
#         torch.cuda.synchronize()
#         end = time.time()
#         total_time += end - start
#     total_time /= num_try
#
# print(total_time)
#
# torch.cuda.empty_cache()
#
# num_try = 100
# total_time = 0
# mod3 = nn.Sequential(mod1, mod2)
# with torch.no_grad():
#     l = torch.randn(bs, 3, 224, 224, requires_grad=False).cuda(device=0)
#     mod3(l)
#     for i in range(num_try):
#         start = time.time()
#         a = mod3(l)
#         torch.cuda.synchronize()
#         end = time.time()
#         total_time += end - start
#     total_time /= num_try
#
# print(total_time)

setup_default_logging()
data_loader0 = []
bs0 = 5
for i in range(2):
    with torch.no_grad():
        l = torch.randn(bs0, 3, 224, 224, requires_grad=False).cuda(device=0)
        data_loader0.append((l, torch.LongTensor(bs0).random_(0, 999).cuda(device=0)))

bs1 = 100
bs2 = bs1

effnet = create_model(
    'efficientnet_b0',
    num_classes=1000,
    in_chans=3,
    pretrained=False)

effnet = effnet.cuda()
effnet(data_loader0[0][0])
loss = nn.CrossEntropyLoss().cuda()
effnet.eval()

# list_channel_to_prune = compute_num_channels_per_layer_taylor(resnet50, [3, 224, 224], data_loader0, pruning_ratio=0.1,
#                                                               last_layer=True)

macs = compute_macs_per_layer(effnet, 224, take_in_channel_saving=False)
time = compute_time_per_layer(effnet, 224, batch_size=512)


list_channel_to_prune = compute_num_channels_per_layer_taylor(effnet, [3, 224, 224], data_loader0,
                                                              pruning_ratio=0.5, prune_skip=True)

# print(list_channel_to_prune)

new_net = redesign_module_resnet(effnet, list_channel_to_prune, input_size=224)
new_net.eval()
effnet.eval()

effnet = effnet.cuda(0)

with torch.no_grad():
    t = torch.randn(bs1, 3, 224, 224, requires_grad=False)
    a = new_net(t.cuda(device=0))

co_mod = build_co_train_model(effnet.cuda(0), new_net.cuda(0))

for n, m in resnet50.named_modules():
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = False

for n, m in new_net.named_modules():
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = False

torch.cuda.empty_cache()
new_net = new_net.cuda(device='cuda:1')
torch.cuda.empty_cache()
for n, p in resnet50.named_parameters():
    if p.grad is not None:
        p.grad = None
prune_network(resnet50, list_channel_to_prune)
torch.cuda.empty_cache()
with torch.no_grad():
    t = torch.randn(bs1, 3, 224, 224, requires_grad=False)
    a = resnet50(t.cuda(device=0))
    b = new_net(t.cuda(device=1))
    a = a.cpu()
    b = b.cpu()
    print(torch.max(torch.abs(a - b)) / torch.mean(torch.abs(a)))

dict_new_net = {}
dict_resnet = {}

# measure_time(new_net, dict_new_net, True)
# measure_time(resnet50, dict_resnet, True)

import time

torch.cuda.empty_cache()

with torch.no_grad():
    t = torch.randn(bs1, 3, 224, 224, requires_grad=False).cuda(device=0)
    start = time.time()
    for i in range(5):
        a = resnet50(t)
    torch.cuda.synchronize(device=0)
    end = time.time()
    torch.cuda.empty_cache()
    print(f'******************************************************resnet50 :{1e4*(end-start)/5}')
    t = torch.randn(bs2, 3, 224, 224, requires_grad=False).cuda(device=1)
    start = time.time()
    for i in range(5):
        b = new_net(t)
    torch.cuda.synchronize(device=1)
    end = time.time()
    print(f'******************************************************new_net :{1e4*(end-start)/5}')

co_mod = build_co_train_model(resnet50.cuda(0), new_net.cuda(0))

o, l = co_mod(torch.randn(30, 3, 224, 224, requires_grad=False).cuda(device=0))

print(l)

total_sum_resnet = 0
total_sum_new_net = 0
for k, v in dict_resnet.items():
    l1 = extract_layer(resnet50, k)
    l2 = extract_layer(new_net, k)
    if isinstance(l1, nn.Conv2d):
        mac1 = torch.prod(torch.tensor(l1.weight.size())).item() * 1.0
        mac2 = torch.prod(torch.tensor(l2.weight.size())).item() * 1.0
        print(
            f'{k} took {v} for resnet and {dict_new_net[k]} for new_net ({dict_new_net[k]/v} time ratio and {mac2/mac1} macs ratio)')
    else:
        print(f'{k} took {v} for resnet and {dict_new_net[k]} for new_net')
        if isinstance(l1, nn.BatchNorm2d) or isinstance(l1, nn.Conv2d) or isinstance(l1, nn.ReLU) or isinstance(l1,
                                                                                                                nn.Linear):
            total_sum_resnet += v
            total_sum_new_net += dict_new_net[k]
print(1e4 * total_sum_new_net / bs2)
print(1e4 * total_sum_resnet / bs1)
