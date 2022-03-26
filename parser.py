import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run KPCR models.")

    parser.add_argument('--seed', type=int, default=3,
                        help='Random seed.')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu id')

    parser.add_argument('--dataset', nargs='?', default='xue',
                        help='dataset name')

    parser.add_argument('--kpcr_mode', nargs='?', default='s',
                        help='[ls, s, si]')

    parser.add_argument('--data_dir', nargs='?', default='./',
                        help='Input data path.')

    parser.add_argument('--embed_dim', type=int, default=64,
                        help='User / entity / relation Embedding size.')
    # ConvE part
    parser.add_argument('--dropout', nargs='?', default='[0.2, 0.1, 0.3]',
                        help='Dropout probability of ConvE(emb,feat,hid)')

    parser.add_argument('--embed_shape', type=int, default=16,
                        help='reshaping shape of ConvE.')

    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='label smoothing for ConvE.')

    parser.add_argument('--out_channels', type=int, default=16,
                        help='out channels of ConvE.')

    parser.add_argument('--kernel_size', type=int, default=3,
                        help='kernel_size of ConvE.')

    # Train part

    parser.add_argument('--cf_batch_size', type=int, default=1024,
                        help='cf batch size.')

    parser.add_argument('--kg_batch_size', type=int, default=1024,
                        help='kg batch size.')

    parser.add_argument('--kg_l2_loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')

    parser.add_argument('--cf_l2_loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating CF l2 loss.')

    parser.add_argument('--test_batch_size', type=int, default=256,
                        help='Test batch size.')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    parser.add_argument('--n_epoch', type=int, default=1000,
                        help='Number of epoch.')

    parser.add_argument('--stopping_steps', type=int, default=25,
                        help='Number of epoch for early stopping')

    parser.add_argument('--print_every', type=int, default=1,
                        help='Iter interval of printing loss.')

    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Epoch interval of evaluating CF.')

    parser.add_argument('--K', type=int, default=10,
                        help='Calculate metric@K when evaluating.')
    args = parser.parse_args()
    print(args)

    return args
