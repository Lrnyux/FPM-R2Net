import torch
import argparse
from PIL import Image
import os
from tqdm import tqdm
from utils import AverageMeter,ConsoleLogger,warp_image
from loss_func_regis import *
from metrics import *
from skimage import transform
from model.networks import ModelHieDense
from model import losses

def create_code_snapshot(root, dst_path, extensions=(".py", ".json"), exclude=()):
    """Creates tarball with the source code"""
    import tarfile
    from pathlib import Path

    with tarfile.open(str(dst_path), "w:gz") as tar:
        tar.add(root, arcname='code', recursive=True)


def get_args():
    parser = argparse.ArgumentParser()

    # =========for hyper parameters===
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--trial',type=str, default='baseline')
    parser.add_argument('--mode', type=str, default='train',choices=['train','test','test_offline'])
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--train_batchsize', type=int, default=2)
    parser.add_argument('--val_batchsize', type=int, default=1)
    parser.add_argument('--max_epoch', default=150, help='max training epoch', type=int)
    parser.add_argument('--lr', default=5e-5, help='learning rate', type=float)
    parser.add_argument('--weight_decay', default=0.0001, help='decay of learning rate', type=float)
    parser.add_argument('--freq_print_train', default=20, help='Printing frequency for training', type=int)
    parser.add_argument('--freq_print_val', default=20, help='Printing frequency for validation', type=int)
    parser.add_argument('--freq_print_test', default=50, help='Printing frequency for test', type=int)
    parser.add_argument('--load_model', type=str, default='')

    # ==========loss function============
    parser.add_argument('--loss_type',type=str,default='mse_reg',choices=['mse_reg'])
    parser.add_argument('--w_mse_img',type=float,default=1.00,help='loss weight for mse image loss')
    parser.add_argument('--w_grad',type=float,default=0.2, help='loss weight for regularization loss')
    parser.add_argument('--clip_min',type=float,default=1.0)
    parser.add_argument('--clip_max', type=float, default=3.0)



    # ========for model ==============
    parser.add_argument('--model_type',type=str,default='hierarchical',choices=['hierarchical'])
    parser.add_argument('--enc', type=int, nargs='+',
                        help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+',
                        help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int_steps', type=int, default=7,
                        help='number of integration steps (default: 7)')
    parser.add_argument('--int_downsize', type=int, default=2,
                        help='flow downsample factor for integration (default: 2)')
    parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--save_gray', type=int, default=0) # 0 for color images; 1 for gray images;
    parser.add_argument('--save_epoch',type=int, default=0)

    # parse configs
    args = parser.parse_args()
    return args


def train():
    args = get_args()
    LOGGER = ConsoleLogger('RegisHie_'+args.trial, 'train')
    logdir = LOGGER.getLogFolder()
    LOGGER.info(args)
    torch.manual_seed(args.seed)
    # -------save current code---------------------------
    save_code_root = os.path.join(logdir, 'code.tar')
    dst_root = os.path.abspath(__file__)
    create_code_snapshot(dst_root, save_code_root)
    # =================Add your own dataset here============================================================
    train_dataloader = None
    #==================================================================================================
    device = torch.device(f'cuda:{args.gpu}')

    bidir = True
    inshape = [32,64,128]
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]
    model = ModelHieDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )


    if args.load_model:
        model_path = args.load_model
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        LOGGER.info(f'---------------Finishing loading models----------------')

    model = model.cuda(device)

    Loss_img =  losses.MSE().loss
    Loss_regular = losses.Grad('l2', loss_mult=args.int_downsize).loss

    best_perf = 100.0


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    step_size = int(args.max_epoch * 0.4 * len(train_dataloader))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                                gamma=0.1)

    # ================train process=============================================
    for epoch in range(args.max_epoch):
        LOGGER.info(f'---------------Training epoch : {epoch}-----------------')
        batch_time = AverageMeter()
        loss_log = AverageMeter()
        loss_img_log = AverageMeter()
        loss_regu_log = AverageMeter()
        loss_val_log = AverageMeter()
        start = time.time()

        model.train()
        for it, batch in enumerate(train_dataloader, 0):

            rgb_img = batch['rgb_img'].to(torch.float32).cuda(device)
            pam_img = batch['pam_img'].to(torch.float32).cuda(device)
            batch_size = len(rgb_img)

            rgb_img_list = []
            pam_img_list = []
            for idx in range(len(inshape)):
                rgb_img_re = model.resize_layers[idx](rgb_img)
                rgb_img_list.append(rgb_img_re)
                pam_img_re = model.resize_layers[idx](pam_img)
                pam_img_list.append(pam_img_re)

            (rgb_img_warp_list, preint_transf_list, pred_transf_list) = model(pam_img_list, rgb_img_list)

            loss_term_img = 0
            loss_term_regular = 0

            for idx in range(len(inshape)):
                loss_term_img += Loss_img(rgb_img_list[idx], rgb_img_warp_list[idx])
                loss_term_regular += Loss_regular(pred_transf_list[idx],preint_transf_list[idx])

            loss = args.w_mse_img * loss_term_img + args.w_grad * loss_term_regular

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # ===============Logging the info and Save the model================
            batch_time.update(time.time() - start)
            loss_log.update(loss.detach(), batch_size)
            loss_img_log.update(loss_term_img.detach(),batch_size)
            loss_regu_log.update(loss_term_regular.detach(),batch_size)
            if it % args.freq_print_train == 0:
                message = 'Epoch : [{0}][{1}/{2}]  Learning rate  {learning_rate:.7f}\t' \
                          'Batch Time {batch_time.val:.3f}s ({batch_time.ave:.3f})\t' \
                          'Speed {speed:.1f} samples/s \t' \
                          'Loss_train {loss1.val:.5f} ({loss1.ave:.5f})\t'\
                          'Loss_img {loss2.val:.5f} ({loss2.ave:.5f})\t' \
                          'Loss_regular {loss3.val:.5f} ({loss3.ave:.5f}) \t'.format(
                    epoch, it, len(train_dataloader), learning_rate=optimizer.param_groups[0]['lr'],
                    batch_time=batch_time, speed=batch_size / batch_time.val, loss1=loss_log,
                    loss2 = loss_img_log, loss3 = loss_regu_log)
                LOGGER.info(message)
            start = time.time()

        if args.save_epoch == 1:
            checkpoint_dir = os.path.join(logdir, 'checkpoints')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            LOGGER.info('=> saving checkpoint to {}'.format(checkpoint_dir))
            states = dict()
            states['model_state_dict'] = model.state_dict()
            states['optimizer_state_dict'] = optimizer.state_dict()

            torch.save(states,os.path.join(checkpoint_dir, 'checkpoint_'+str(epoch)+'.tar'))

    checkpoint_dir = os.path.join(logdir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    LOGGER.info('=> saving checkpoint to {}'.format(checkpoint_dir))
    states = dict()
    states['model_state_dict'] = model.state_dict()
    states['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(states, os.path.join(checkpoint_dir, 'last.tar'))

    LOGGER.info('Finish Training')




if __name__ == "__main__":
    args = get_args()

    if args.mode == 'train':
        train()











