def add_prune_to_parser(parser):
    parser.add_argument('-bp', '--batch-size-prune', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--prune', action='store_true', default=False,
                        help='Prune the network')
    parser.add_argument('--pruning_ratio', type=float, default=0.2,
                        help='pruning ratio')
    parser.add_argument('--gamma_knowledge', type=float, default=0.2,
                        help='factor for distillation penalty')
    parser.add_argument('--taylor_file', default=None, type=str, help='File where to save the taylor computation')
    parser.add_argument('--prune_test', action='store_true', default=False,
                        help='Prune on test set')
    parser.add_argument('--no_pwl', action='store_true', default=False,
                        help='Do not prune pwl conv')
    parser.add_argument('--taylor_abs', action='store_true', default=False,
                        help='Alternative taylor')
    parser.add_argument('--prune_skip', action='store_true', default=False,
                        help='Prune skip connection in resnet')
    parser.add_argument('--only_last', action='store_true', default=False,
                        help='IKD only on last layer of the block')
    parser.add_argument('--prune_conv1', action='store_true', default=False,
                        help='Prune stem')
    parser.add_argument('--progressive_IKD_factor', action='store_true', default=False,
                        help='The IKD factor increase linearily along the depth of the network')
    parser.add_argument('--use_time', action='store_true', default=False,
                        help='Use time measurement instead of FLOPS')
