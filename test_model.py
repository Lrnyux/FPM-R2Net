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


def _test_offline(checkpoint=''):
    args = get_args()
    if checkpoint:
        args.load_model = checkpoint
    LOGGER = ConsoleLogger('RegHie_test_'+args.trial, 'test')
    logdir = LOGGER.getLogFolder()
    LOGGER.info(args)

    save_gray = 1

    # =================Add your collected data here============================================================
    rgb_img_folder = ''
    pam_img_folder = ''
    # ==================================================================================================

    img_listdir = sorted(os.listdir(rgb_img_folder))
    device = torch.device(f'cuda:{args.gpu}')

    # model

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

    model = model.to(device)
    time_list = []

    num = 0
    with torch.no_grad():
        model.eval()
        for idx in tqdm(range(len(img_listdir))):

            rgb_img_root = os.path.join(rgb_img_folder,img_listdir[idx])
            pam_img_root = os.path.join(pam_img_folder,img_listdir[idx])
            filename = img_listdir[idx].replace('.png','')

            rgb_img_info = np.array(Image.open(rgb_img_root))/255.0
            pam_img_info = np.array(Image.open(pam_img_root))/255.0
            h_ori,w_ori = rgb_img_info.shape

            rgb_img = np.copy(rgb_img_info)
            pam_img = np.copy(pam_img_info)

            rgb_img_list = []
            pam_img_list = []

            for idx in range(len(inshape)):
                rgb_img_re = transform.resize(rgb_img,(inshape[idx],inshape[idx]))
                rgb_img_list.append(torch.from_numpy(rgb_img_re).unsqueeze_(0).unsqueeze_(0).to(torch.float32).to(device))
                pam_img_re = transform.resize(pam_img,(inshape[idx],inshape[idx]))
                pam_img_list.append(torch.from_numpy(pam_img_re).unsqueeze_(0).unsqueeze_(0).to(torch.float32).to(device))

            # =========feed into the network=========================
            s_time = time.time()
            rgb_img_warp_list, pred_transf_list = model(pam_img_list, rgb_img_list, registration=True)

            time_list.append(time.time()-s_time)

            pred_transf = np.zeros((h_ori,w_ori,2))

            for idx in range(len(inshape)):
                transf_per = pred_transf_list[idx][0].cpu().numpy()
                transf_per[0,:,:] = transf_per[0,:,:]/(inshape[idx]-1)
                transf_per[1,:,:] = transf_per[1,:,:]/(inshape[idx]-1)
                transf_per = np.transpose(transf_per,[1,2,0])
                transf_per = transform.resize(transf_per,[h_ori,w_ori])
                pred_transf = pred_transf + transf_per


            f = torch.from_numpy(pred_transf).unsqueeze_(0)
            img_m = torch.from_numpy(pam_img_info).unsqueeze_(0).unsqueeze_(1)
            img_warp_big = warp_image(img_m,f).numpy()

            render_img_big = np.zeros((h_ori, w_ori, 3))
            render_img_big[:, :, 0] = abs(rgb_img_info)
            render_img_big[:, :, 1] = abs(img_warp_big)

            np.save(os.path.join(logdir,filename+'.npy'),pred_transf)

            if save_gray == 1:
                Image.fromarray(np.array(255 * (img_warp_big[0,0]/np.max(img_warp_big)),dtype=np.uint8)).save(os.path.join(logdir,filename+'.png'))

        time_list = np.array(time_list)
        time_avg = np.mean(time_list)
        LOGGER.info('Average Time: {:.5f}'.format(time_avg))
        LOGGER.info('Fininshing.')



if __name__ == "__main__":
    args = get_args()

    if args.mode == 'test_offline':
        _test_offline()











