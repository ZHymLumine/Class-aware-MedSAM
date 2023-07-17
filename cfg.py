import argparse


# %% set up parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', '--tr_npy_path', type=str,
                        default='data/npy/CT_Abd/train',
                        help='path to training npy files; two subfolders: gts and imgs')
    parser.add_argument('-test', '--ts_npy_path', type=str,
                        default='data/npy/CT_Abd/test',
                        help='path to test npy files; two subfolders: gts and imgs')
    parser.add_argument("-tolerance", default=5, type=int)
    parser.add_argument('-task_name', type=str, default='Class-aware-MedSAM-ViT-B')
    parser.add_argument('-model_type', type=str, default='vit_b')
    parser.add_argument('-checkpoint', type=str, default='work_dir/SAM/medsam_vit_b.pth')
    parser.add_argument('-vis', type=int, default=None, help='visualization')
    parser.add_argument('-thd', type=bool, default=False, help='3d or not')
    # parser.add_argument('-device', type=str, default='cuda:0')
    parser.add_argument('--load_pretrain', type=bool, default=True,
                        help='use wandb to monitor training')
    parser.add_argument('-pretrain_model_path', type=str, default='')
    parser.add_argument('-work_dir', type=str, default='./work_dir')
    # train
    parser.add_argument('-num_epochs', type=int, default=1000)
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-num_workers', type=int, default=0)
    # Optimizer parameters
    parser.add_argument('-weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    parser.add_argument('-lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('-use_wandb', type=bool, default=False,
                        help='use wandb to monitor training')
    parser.add_argument('-use_amp', action='store_true', default=False,
                        help='use amp')
    parser.add_argument('--resume', type = str, default = '',
                        help="Resuming training from checkpoint")
    parser.add_argument('-image_size', type=int, default=1024, help='image_size')
    parser.add_argument('-out_size', type=int, default=256, help='output_size')
    #parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument(
        '-data_path',
        type=str,
        default='./data/BTCV',
        help='The path of segmentation data')
    parser.add_argument('-roi_size', type=int, default=96 , help='resolution of roi')
    parser.add_argument('-chunk', type=int, default=96, help='crop volume depth')
    parser.add_argument('-num_sample', type=int, default=4, help='sample pos and neg')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-fast', type=bool, default=False, help='whether fast dataloader')
    # seed
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    opt = parser.parse_args()

    return opt