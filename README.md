# Knapsack Pruning with Inner Distillation

This code has been adapted from the excellent [Ross Wightman repository](https://github.com/rwightman/pytorch-image-models)
that we used to train our models. We have used several features from this repository, such as 
* Data manager
* SGD Optimizer
* Cosine Scheduler
* Distributed and APEX support
* Several other cool features

It is the implementation of our paper, available on [Arxiv](https://arxiv.org/abs/2002.08258), with several improvements.

For now, our code supports the pruning of the following networks
* [Resnet](https://arxiv.org/abs/1512.03385)
* [Gluon Resnet version D](https://arxiv.org/abs/1812.01187)
* [EfficientNet](https://arxiv.org/abs/1905.11946)

We will see how to use this repository.
Examples are given with hyperML, our internal training platform.

## Train the base model

To train the base model, you can use the `train.py` file, and all the instructions can be found on the main page of 
[Ross Wightman repository](https://github.com/rwightman/pytorch-image-models). So we will skip this part.

## Pruning the model

The code to prune the model can be found in the file `train_pruning.py`. We will go over every of the parameters. 
Let start with a command that can reproduce pruning of 41% of the FLOPS of ResNet-50 and get 78.54% final accuracy.
For this, we should start from the model in the
[latest checkpoint of Ross Wightman](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth)
that achieves 79% accuracy on the test set. 
**For now, our code supports only distributed data parallel training and not Pytorch data parallel.**


```
python -u -m torch.distributed.launch --nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
./train_pruning.py \
/data/imagenet/ \
-b=192 \
--amp \
--model=resnet50 \
--lr=0.02 \
--sched=cosine \
-bp=128 \
--pruning_ratio=0.27 \
--prune \
--prune_skip \
--gamma_knowledge=20 \
--epochs=50 \
```

Let's go over the parameters:

* The first lines indicates that we run the code in distributed mode on 8 GPU's. 
```
python -u -m torch.distributed.launch --nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
./train_pruning.py \
```

* The first parameter `/data/imagenet/` is the location of the dataset. 

* The second parameter `-b` represents the batch size.

* The third parameter `-amp` indicates that we want to train with mixed precision (NVIDIA APEX). This speeds the training
by a factor 3 when training on Tesla V100 GPU's.

* The parameter `-j` indicates the number of worker per GPU.

* The parameter `-model` selects the model to prune. You can select any of the ResNet's, or the EfficientNet's

* The learning rate for fine tuning can be set with `--lr`.

* The training scheduler is chosen with `--sched`. We recommend to use cosine scheduler.

* In order to choose the batch size of the Taylor Computation for the pruning, you can use the parameter `-bp`

* The pruning ratio can be set with `pruning_ratio`. Note that when pruning a skip connection, this ratio is not very 
accurate so you need to fine tune to observe the actual pruning ratio that is computed by dynamic programming or greedy
algorithm. The default algorithm is greedy which is faster and provides a comparable accuracy.

* The option `--prune` indicates that you want to prune the network. TODO: Remove this parameter and set it always to TRUE

* The option `--prune_skip` is important only for ResNets and force pruning the skip connection via grouping as described
in the paper

* The option `--gamma_knowledge` represents the Inner Knowledge Distillation penalty constant.
* To set the number of epochs, use `--epochs`. 

* In order to reproduce the results from the paper (inferior results), you can use the option 
`--taylor_var`. It uses the Taylor scheme of [Molchanov et.al](https://arxiv.org/abs/1611.06440). It is not recommended to use this option since this led to
lower accuracy.   

* `--initial-checkpoint` represents the local path of the unpruned checkpoint.
* `--output` is the path for output folder


## Loading a pruned model
Suppose that you have trained and pruned a model, and would like to fine-tune it or load it in another repository.
The function `load_module_from_ckpt` located in `external.utils_pruning` is able to adapt an unpruned model to a pruned 
checkpoint. You need to provide the original model as first parameter of the function, and the path of the pruned
checkpoint as second parameter. The function will analyse and compare the number of channels of the convolutions, 
batch-norm layers and fully connected layers of the unpruned model and compare them with the one in the pruned checkpoint.
You can also prune a pruned model by using the parameter `--initial-checkpoint-pruned` 
in the `train_pruning.py` script.

## Pretrained checkpoint 
All of the pretrained checkpoint for efficientNet and ResNet are located in:
`TODO add checkpoints`

In particular, for ResNet-50 you have four checkpoints:
* `resnet50-19c8e357.pth`, representing the official Pytorch pretrained model (accuracy: 76.15%)
* `rw_resnet50-86acaeed.pth` representing the training of Ross Wightman  (accuracy: 78.47%)
* `ya_resnet50-e18cda54.pth` representing my own training similar to Ross Wightman (accuracy: 78.45%)
* `resnet50_ram-a26f946b.pth` representing the best ResNet model trained with JSD loss and AugMix augmentation scheme
 (accuracy: 79.0%). **We strongly recommend to use this checkpoint to get a pruned model**.

## Benchmark
To reproduce the results we got, we provide here some paths to hyperML runs, as well as final accuracy:

Model to pruned | Pruning ratio | Unpruned accuracy |Pruned accuracy | Link
---|---|---|---|---|
ResNet-50| 40.74% | 79.00% | 78.54% | [add link](https://hyperml.alibaba-inc.com/job/38458)
ResNet-50| 50.80% | 79.00% | 77.74% | [add link](https://hyperml.alibaba-inc.com/job/38551)
ResNet-50| 41.79% | 78.45% | 78.25% | [add link](https://hyperml.alibaba-inc.com/job/38555)
ResNet-50| 50.80% | 78.45% | 77.90% | [add link](https://hyperml.alibaba-inc.com/job/38557)
EfficientNet B0 | 46.00% | 77.30% | 75.50% | [add link](https://hyperml.alibaba-inc.com/job/34240)
EfficientNet B1 | 44.28% | 79.20% | 78.30% | [add link](https://hyperml.alibaba-inc.com/job/33970)
EfficientNet B2 | 30.00% | 80.30% | 79.90% | [add link](https://hyperml.alibaba-inc.com/job/33987)
EfficientNet B3 | 44.00% | 81.7% | 80.80% | [add link](https://hyperml.alibaba-inc.com/job/32497)



