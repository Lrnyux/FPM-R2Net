import torch
import argparse
import logging
import os
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from torchvision import transforms, utils
from functools import partial
from tqdm import tqdm
from collections import OrderedDict
import torch.nn.functional as F
import logging
import os
import time
from tqdm import tqdm
import random
from utils import AverageMeter,ConsoleLogger,calc_rmse,calc_psnr,calc_ssim,save_results,save_results_3d,save_results_reg
from model_unet_regis import unet
from loss_func import *
from loss_func_regis import *
from metrics import *
import lpips
from scipy.interpolate import interp1d
from skimage import transform
import SimpleITK as sitk
from seg_utils.func_imit_bleed import add_blood_block,simulate_blood,gen_gauss
import scipy.io
from Regis_utils.func_airlab_regis import regis_with_airlab, compute_grid, warp_image
import Regis_utils.voxelmorph as vxm
import matplotlib.pyplot  as plt


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

    # ==========define the task==============
    parser.add_argument('--task',type=str,default='Registration',choices=['Registration'])
    parser.add_argument('--dataset_type', type=str, default='brain-3', choices=['brain-1','brain-2','brain-3','syn'])
    parser.add_argument('--few_shot_root',type=str,default='')

    parser.add_argument('--if_rgb_seg',type=int,default=0,choices=[0,1],help='Whether to use mask for rgb fixed image (0 for not use 1 for use)')
    parser.add_argument('--if_mask',type=int,default=0,choices=[0,1],help='Whether to use mask for mse loss (0 for not use 1 for use)')
    parser.add_argument('--loss_mask_thres',type=float,default=0.1,help='The value for binary thresholding rgb image for mse loss')
    parser.add_argument('--if_depth_concat',type=int,default=0,choices=[0,1],help='Whether to use depth information for model [concatenation] (0 for not use 1 for use)')
    parser.add_argument('--if_depth_mults', type=int, default=0, choices=[0, 1],help='Whether to use depth information for model [multiplication] (0 for not use 1 for use)')



    # =========for Regis dataset with affine combined with non-linear [cross subject] ============
    parser.add_argument('--train_rgb_img_root', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Regis_cut/split_nopad/color_gray_pretrain_train')
    parser.add_argument('--train_pam_img_root', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Regis_cut/split_nopad/pam_gray_pretrain_train')
    parser.add_argument('--train_pam_depth_root', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Regis_cut/split_nopad/pam_depth_train')
    parser.add_argument('--train_data_root', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Regis_cut/split_nopad/regis_results_mat_train')


    parser.add_argument('--val_rgb_img_root', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Regis_cut/split_nopad/color_gray_pretrain_test')
    parser.add_argument('--val_pam_img_root', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Regis_cut/split_nopad/pam_gray_pretrain_test')
    parser.add_argument('--val_pam_depth_root', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Regis_cut/split_nopad/pam_depth_test')
    parser.add_argument('--val_data_root', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Regis_cut/split_nopad/regis_results_mat_test')

    parser.add_argument('--test_rgb_img_root', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Regis_cut/split_nopad/color_gray_pretrain_test')
    parser.add_argument('--test_pam_img_root', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Regis_cut/split_nopad/pam_gray_pretrain_test')
    parser.add_argument('--test_pam_depth_root', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Regis_cut/split_nopad/pam_depth_test')
    parser.add_argument('--test_data_root', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Regis_cut/split_nopad/regis_results_mat_test')

    # parser.add_argument('--train_rgb_img_root', type=str,
    #                     default='/home/work/Yuxuan/Code/Seg_PA_2D/in_vivo_test/regis_data/affine_nonlinear/rgb_gray_pad_train')
    # parser.add_argument('--train_pam_img_root', type=str,
    #                     default='/home/work/Yuxuan/Code/Seg_PA_2D/in_vivo_test/regis_data/affine_nonlinear/pam_gray_pad_train')
    # parser.add_argument('--train_pam_depth_root', type=str,
    #                     default='/home/work/Yuxuan/Code/Seg_PA_2D/in_vivo_test/regis_data/affine_nonlinear/pam_depth_train')
    # parser.add_argument('--train_data_root', type=str,
    #                     default='/home/work/Yuxuan/Code/Seg_PA_2D/in_vivo_test/regis_data/affine_nonlinear/regis_results_mat_train')
    #
    # parser.add_argument('--val_rgb_img_root', type=str,
    #                     default='/home/work/Yuxuan/Code/Seg_PA_2D/in_vivo_test/regis_data/affine_nonlinear/rgb_gray_pad_test')
    # parser.add_argument('--val_pam_img_root', type=str,
    #                     default='/home/work/Yuxuan/Code/Seg_PA_2D/in_vivo_test/regis_data/affine_nonlinear/pam_gray_pad_test')
    # parser.add_argument('--val_pam_depth_root', type=str,
    #                     default='/home/work/Yuxuan/Code/Seg_PA_2D/in_vivo_test/regis_data/affine_nonlinear/pam_depth_test')
    # parser.add_argument('--val_data_root', type=str,
    #                     default='/home/work/Yuxuan/Code/Seg_PA_2D/in_vivo_test/regis_data/affine_nonlinear/regis_results_mat_test')
    #
    # parser.add_argument('--test_rgb_img_root', type=str,
    #                     default='/home/work/Yuxuan/Code/Seg_PA_2D/in_vivo_test/regis_data/affine_nonlinear/rgb_gray_pad_test')
    # parser.add_argument('--test_pam_img_root', type=str,
    #                     default='/home/work/Yuxuan/Code/Seg_PA_2D/in_vivo_test/regis_data/affine_nonlinear/pam_gray_pad_test')
    # parser.add_argument('--test_pam_depth_root', type=str,
    #                     default='/home/work/Yuxuan/Code/Seg_PA_2D/in_vivo_test/regis_data/affine_nonlinear/pam_depth_test')
    # parser.add_argument('--test_data_root', type=str,
    #                     default='/home/work/Yuxuan/Code/Seg_PA_2D/in_vivo_test/regis_data/affine_nonlinear/regis_results_mat_test')

    parser.add_argument('--train_rgb_img_root_syn', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Syn_gen/Data_Syn_III/split/train/rgb_seg_pretrain')
    parser.add_argument('--train_pam_img_root_syn', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Syn_gen/Data_Syn_III/split/train/pam_seg')
    parser.add_argument('--train_pam_depth_root_syn', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Syn_gen/Data_Syn_III/split/train/pam')
    parser.add_argument('--train_data_root_syn', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Syn_gen/Data_Syn_III/split/train/regis_mat')

    parser.add_argument('--val_rgb_img_root_syn', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Syn_gen/Data_Syn_III/split/test/rgb_seg_pretrain')
    parser.add_argument('--val_pam_img_root_syn', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Syn_gen/Data_Syn_III/split/test/pam_seg')
    parser.add_argument('--val_pam_depth_root_syn', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Syn_gen/Data_Syn_III/split/test/pam')
    parser.add_argument('--val_data_root_syn', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Syn_gen/Data_Syn_III/split/test/regis_mat')

    parser.add_argument('--test_rgb_img_root_syn', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Syn_gen/Data_Syn_III/split/test/rgb_seg_pretrain')
    parser.add_argument('--test_pam_img_root_syn', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Syn_gen/Data_Syn_III/split/test/pam_seg')
    parser.add_argument('--test_pam_depth_root_syn', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Syn_gen/Data_Syn_III/split/test/pam')
    parser.add_argument('--test_data_root_syn', type=str,
                        default='/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Syn_gen/Data_Syn_III/split/test/regis_mat')



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
    parser.add_argument('--w_mse_f',type=float,default=0.000, help='loss weight for mse deformation field loss')
    parser.add_argument('--w_grad',type=float,default=0.2, help='loss weight for regularization loss')

    parser.add_argument('--clip_min',type=float,default=1.0)
    parser.add_argument('--clip_max', type=float, default=3.0)

    # ===========threshold for segmentation==================
    parser.add_argument('--thres_high',type=float,default=0.7)
    parser.add_argument('--thres_low',type=float,default=0.1)



    # ========for model ==============
    parser.add_argument('--model_type',type=str,default='hierarchical',choices=['voxelmorph','hierarchical'])
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



class PA_dataset(Dataset):

    #  baseline method for photoacoustic image super resolution
    #  input is the bicubic interpolation of the down sampling image

    def __init__(self,rgb_img_dir,pam_img_dir,pam_depth_dir,data_dir,args,stage='Train'):
        self.rgb_img_dir = rgb_img_dir
        self.pam_img_dir = pam_img_dir
        self.pam_depth_dir = pam_depth_dir
        self.data_dir = data_dir
        self.args = args
        self.stage = stage
        self.rgb_img_filelist = []
        self.pam_img_filelist = []
        self.pam_depth_filelist = []
        self.reg_filelist = []

        assert len(self.rgb_img_filelist) == len(self.pam_img_filelist)

        root_list = sorted(os.listdir(self.rgb_img_dir))

        if (self.stage == 'Train') and (self.args.few_shot_root):
            few_shot_index = np.load(self.args.few_shot_root)
            print('Few shot Learning: ' + self.args.few_shot_root)
            for idx in range(len(few_shot_index)):
                data_root = os.path.join(self.data_dir, str(few_shot_index[idx]).zfill(4) + '.mat')
                self.reg_filelist.append(data_root)
                rgb_img_root = os.path.join(self.rgb_img_dir, str(few_shot_index[idx]).zfill(4) + '.png')
                self.rgb_img_filelist.append(rgb_img_root)
                pam_img_root = os.path.join(self.pam_img_dir, str(few_shot_index[idx]).zfill(4) + '.png')
                self.pam_img_filelist.append(pam_img_root)
                pam_depth_root = os.path.join(self.pam_depth_dir, str(few_shot_index[idx]).zfill(4) + '.png')
                self.pam_depth_filelist.append(pam_depth_root)
        else:
            for idx in range(len(root_list)):
                data_root = os.path.join(self.data_dir, root_list[idx].replace('.png', '.mat'))
                self.reg_filelist.append(data_root)
                rgb_img_root = os.path.join(self.rgb_img_dir, root_list[idx])
                self.rgb_img_filelist.append(rgb_img_root)
                pam_img_root = os.path.join(self.pam_img_dir, root_list[idx])
                self.pam_img_filelist.append(pam_img_root)
                pam_depth_root = os.path.join(self.pam_depth_dir, root_list[idx])
                self.pam_depth_filelist.append(pam_depth_root)

        self.length = len(self.reg_filelist)



    def __len__(self):
        return self.length

    def __getitem__(self, index):

        rgb_img_file_root = self.rgb_img_filelist[index]
        pam_img_file_root = self.pam_img_filelist[index]
        pam_depth_file_root = self.pam_depth_filelist[index]
        reg_file_root = self.reg_filelist[index]




        if self.stage == 'Train':
            a_rgb = (np.random.rand()-0.5) * 30
            a_pam = (np.random.rand()-0.5) * 30
            # rgb_img_info = np.array(Image.open(rgb_img_file_root).rotate((np.random.rand()-0.5)*30,expand=True),dtype=np.float64)/255.0
            # pam_img_info = np.array(Image.open(pam_img_file_root).rotate((np.random.rand()-0.5)*30,expand=True))/255.0
            rgb_img_info = np.array(Image.open(rgb_img_file_root).rotate(a_rgb, expand=True),dtype=np.float64) / 255.0
            pam_img_info = np.array(Image.open(pam_img_file_root).rotate(a_pam,expand=True))/255.0
            pam_depth_info = np.array(Image.open(pam_depth_file_root).rotate(a_pam,expand=True))



            if self.args.dataset_type == 'brain':
                rgb_img_crop = np.zeros_like(rgb_img_info)
                pam_img_crop = np.zeros_like(pam_img_info)
                pam_depth_crop = np.zeros_like(pam_depth_info)

                h_crop,w_crop = rgb_img_crop.shape
                d_shift_rgb = [int((np.random.rand()-0.5)*30),int((np.random.rand()-0.5)*30)]
                rgb_img_crop[25+d_shift_rgb[0]:h_crop-25,25+d_shift_rgb[1]:w_crop-25]=rgb_img_info[25:h_crop-25-d_shift_rgb[0],25:w_crop-25-d_shift_rgb[1]]

                h_crop, w_crop = pam_img_crop.shape
                d_shift_pam = [int((np.random.rand()-0.5)*30), int((np.random.rand()-0.5)*30)]
                pam_img_crop[25 + d_shift_pam[0]:h_crop-25, 25 + d_shift_pam[1]:w_crop-25] = pam_img_info[
                                                                                       25:h_crop -25 - d_shift_pam[0],
                                                                                       25:w_crop -25 - d_shift_pam[1]]
                pam_depth_crop[25 + d_shift_pam[0]:h_crop - 25, 25 + d_shift_pam[1]:w_crop - 25] = pam_depth_info[
                                                                                        25:h_crop - 25 -d_shift_pam[0],
                                                                                        25:w_crop - 25 -d_shift_pam[1]]



                rgb_img_info = np.copy(rgb_img_crop)
                pam_img_info = np.copy(pam_img_crop)
                pam_depth_info = np.copy(pam_depth_crop)







        else:
            rgb_img_info = np.array(Image.open(rgb_img_file_root),dtype=np.float64)/255.0
            pam_img_info = np.array(Image.open(pam_img_file_root),dtype=np.float64)/255.0
            pam_depth_info = np.array(Image.open(pam_depth_file_root))


        # ================================        process for the depth image    =======================================
        pam_img_mask = np.zeros_like(pam_img_info)
        pam_img_mask[pam_img_info>0.5]=1.0
        pam_depth_info = pam_depth_info * pam_img_mask
        depth_list = pam_depth_info[pam_depth_info != 0]
        depth_mean = np.mean(depth_list)
        pam_depth_info[pam_depth_info == 0] = depth_mean
        pam_depth_info = pam_depth_info - depth_mean
        pam_depth_info = pam_depth_info / np.max(abs(pam_depth_info))
        pam_depth_info = 1 - abs(pam_depth_info)

        pam_depth_info = (pam_depth_info - np.min(pam_depth_info)) / (np.max(abs(pam_depth_info)) - np.min(abs(pam_depth_info)))

        h_ori, w_ori = rgb_img_info.shape
        reg_info = scipy.io.loadmat(reg_file_root)

        try:
            f_affine = reg_info['displacement_affine']
            f_nonlinear = reg_info['displacement_nonlinear']
        except:
            f_affine = np.zeros((h_ori,w_ori,2))
            f_nonlinear = np.zeros((h_ori,w_ori,2))
        f_total = f_affine + f_nonlinear




        img_size = args.image_size
        rgb_img_info = transform.resize(rgb_img_info,(img_size,img_size))
        pam_img_info = transform.resize(pam_img_info,(img_size,img_size))
        pam_depth_info = transform.resize(pam_depth_info,(img_size,img_size))
        f_affine = transform.resize(f_affine,(img_size,img_size))
        f_nonlinear = transform.resize(f_nonlinear,(img_size,img_size))
        f_total = transform.resize(f_total,(img_size,img_size))




        h,w = rgb_img_info.shape



        # ==============augmentation===================================
        if self.stage == 'Train':

            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            rot90 = random.random() < 0.5

            if hflip:
                rgb_img_info = rgb_img_info[::-1,:]
                pam_img_info = pam_img_info[::-1,:]
                pam_depth_info = pam_depth_info[::-1,:]
                # f_total = -f_total
            #
            if vflip:
                rgb_img_info = rgb_img_info[:,::-1]
                pam_img_info = pam_img_info[:,::-1]
                pam_depth_info = pam_depth_info[:,::-1]
                # f_total = -f_total
            #
            if rot90:
                # pass
                rgb_img_info = np.rot90(rgb_img_info,1)
                pam_img_info = np.rot90(pam_img_info,1)
                pam_depth_info = np.rot90(pam_depth_info,1)

        #     add small jiters for moving image

        rgb_img_mask = np.zeros_like(rgb_img_info)
        rgb_img_mask[rgb_img_info > self.args.loss_mask_thres] = 1.0

        if args.if_rgb_seg == 1:
            rgb_img_info[rgb_img_info>0.1]=1.0


        if self.stage == 'Train':
            rgb_img_info = rgb_img_info ** (np.random.rand() + 0.5)
            pam_img_info = pam_img_info ** (np.random.rand() + 0.5)


        rgb_img_info = rgb_img_info.reshape((1,h,w))
        pam_img_info = pam_img_info.reshape((1,h,w))
        pam_depth_info = pam_depth_info.reshape((1,h,w))
        rgb_img_mask = rgb_img_mask.reshape((1, h, w))
        # concat_img_info = np.concatenate([rgb_img_info,pam_img_info],axis=0)
        f_total = f_total.reshape((h,w,2))

        sample={}
        sample['gt']= f_total.copy()
        sample['f_affine'] = f_affine.copy()
        sample['f_nonlinear'] = f_nonlinear.copy()
        sample['rgb_img'] = rgb_img_info.copy()
        sample['pam_img'] = pam_img_info.copy()
        sample['pam_depth'] = pam_depth_info.copy()
        sample['root_rgb_img'] = rgb_img_file_root
        sample['root_pam_img'] = pam_img_file_root
        sample['root_pam_depth'] = pam_depth_file_root
        sample['root_reg'] = reg_file_root
        sample['ori_size'] = np.array([h_ori,w_ori])
        sample['mask'] = rgb_img_mask.copy()
        return sample



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
    # dataset============================================================
    if (args.dataset_type == 'brain-1') or (args.dataset_type == 'brain-2') or (args.dataset_type == 'brain-3'):
        fold_idx = args.dataset_type.replace('brain-', '')
        args.train_rgb_img_root = args.train_rgb_img_root.replace('split_nopad', 'split_nopad_' + fold_idx)
        args.train_pam_img_root = args.train_pam_img_root.replace('split_nopad', 'split_nopad_' + fold_idx)
        args.train_pam_depth_root = args.train_pam_depth_root.replace('split_nopad', 'split_nopad_' + fold_idx)
        args.train_data_root = args.train_data_root.replace('split_nopad', 'split_nopad_' + fold_idx)
        args.val_rgb_img_root = args.val_rgb_img_root.replace('split_nopad', 'split_nopad_' + fold_idx)
        args.val_pam_img_root = args.val_pam_img_root.replace('split_nopad', 'split_nopad_' + fold_idx)
        args.val_pam_depth_root = args.val_pam_depth_root.replace('split_nopad', 'split_nopad_' + fold_idx)
        args.val_data_root = args.val_data_root.replace('split_nopad', 'split_nopad_' + fold_idx)
        args.test_rgb_img_root = args.test_rgb_img_root.replace('split_nopad', 'split_nopad_' + fold_idx)
        args.test_pam_img_root = args.test_pam_img_root.replace('split_nopad', 'split_nopad_' + fold_idx)
        args.test_pam_depth_root = args.test_pam_depth_root.replace('split_nopad', 'split_nopad_' + fold_idx)
        args.test_data_root = args.test_data_root.replace('split_nopad', 'split_nopad_' + fold_idx)
        file_name_list = [args.train_rgb_img_root, args.train_pam_img_root, args.train_pam_depth_root,
                          args.train_data_root,
                          args.val_rgb_img_root, args.val_pam_img_root, args.val_pam_depth_root,
                          args.val_data_root,
                          args.test_rgb_img_root, args.test_pam_img_root, args.test_pam_depth_root,
                          args.test_data_root, ]
        LOGGER.info(file_name_list)
        train_set = PA_dataset(args.train_rgb_img_root,args.train_pam_img_root,args.train_pam_depth_root,args.train_data_root,args,stage='Train')
        train_dataloader = DataLoader(train_set, batch_size=args.train_batchsize, shuffle=True, num_workers=16,drop_last=True)
        val_set = PA_dataset(args.val_rgb_img_root,args.val_pam_img_root,args.val_pam_depth_root, args.val_data_root, args, stage='Val')
        val_dataloader = DataLoader(val_set, batch_size=args.val_batchsize, shuffle=False, num_workers=16, drop_last=False)
        test_set = PA_dataset(args.test_rgb_img_root,args.test_pam_img_root,args.test_pam_depth_root, args.test_data_root, args, stage='Test')
        test_dataloader = DataLoader(test_set, batch_size=args.val_batchsize, shuffle=False, num_workers=16, drop_last=False)

    if args.dataset_type == 'syn':
        train_set = PA_dataset(args.train_rgb_img_root_syn, args.train_pam_img_root_syn, args.train_pam_depth_root_syn,
                               args.train_data_root_syn, args, stage='Train')
        train_dataloader = DataLoader(train_set, batch_size=args.train_batchsize, shuffle=True, num_workers=16,
                                      drop_last=True)
        val_set = PA_dataset(args.val_rgb_img_root_syn, args.val_pam_img_root_syn, args.val_pam_depth_root_syn, args.val_data_root_syn,
                             args, stage='Val')
        val_dataloader = DataLoader(val_set, batch_size=args.val_batchsize, shuffle=False, num_workers=16,
                                    drop_last=False)
        test_set = PA_dataset(args.test_rgb_img_root_syn, args.test_pam_img_root_syn, args.test_pam_depth_root_syn,
                              args.test_data_root_syn, args, stage='Test')
        test_dataloader = DataLoader(test_set, batch_size=args.val_batchsize, shuffle=False, num_workers=16,
                                     drop_last=False)

    LOGGER.info('Initial Dataset Finished|| dataset type: '+args.dataset_type)
    #====================================================================
    device = torch.device(f'cuda:{args.gpu}')
    # model
    if args.model_type == 'hierarchical':
        bidir = args.bidir
        inshape = [32,64,128]
        enc_nf = args.enc if args.enc else [16, 32, 32, 32]
        dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]
        model = vxm.networks.VxmHieDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=bidir,
            int_steps=args.int_steps,
            int_downsize=args.int_downsize
        )

        LOGGER.info('Initial Model Finished : '+args.model_type+ ' Image size : '+str(args.image_size)+' Downsampling factor : '+ str(args.int_downsize)+' If use depth concat : '+ str(args.if_depth_concat))

    if args.load_model:
        model_path = args.load_model
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        LOGGER.info(f'---------------Finishing loading models----------------')

    model = model.cuda(device)

    Loss_img =  vxm.losses.MSE().loss
    Loss_f = vxm.losses.MSE_F().loss
    Loss_regular = vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss

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
        loss_f_log = AverageMeter()
        loss_regu_log = AverageMeter()
        loss_val_log = AverageMeter()
        start = time.time()

        model.train()
        for it, batch in enumerate(train_dataloader, 0):

            rgb_img = batch['rgb_img'].to(torch.float32).cuda(device)
            pam_img = batch['pam_img'].to(torch.float32).cuda(device)
            pam_depth = batch['pam_depth'].to(torch.float32).cuda(device)
            transf = batch['gt'].to(torch.float32).cuda(device)
            mask = batch['mask'].to(torch.float32).cuda(device)



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
                          'Loss_f {loss3.val:.5f} ({loss3.ave:.5f}) \t'\
                          'Loss_regular {loss4.val:.5f} ({loss4.ave:.5f}) \t'.format(
                    epoch, it, len(train_dataloader), learning_rate=optimizer.param_groups[0]['lr'],
                    batch_time=batch_time, speed=batch_size / batch_time.val, loss1=loss_log,
                    loss2 = loss_img_log, loss3 = loss_f_log, loss4 = loss_regu_log)
                LOGGER.info(message)
            start = time.time()

        # ================validation process=============================================
        with torch.no_grad():
            model.eval()
            for it, batch in enumerate(val_dataloader, 0):
                rgb_img = batch['rgb_img'].to(torch.float32).cuda(device)
                pam_img = batch['pam_img'].to(torch.float32).cuda(device)
                pam_depth = batch['pam_depth'].to(torch.float32).cuda(device)
                transf = batch['gt'].to(torch.float32).cuda(device)
                batch_size = len(rgb_img)

                rgb_img_list = []
                pam_img_list = []
                for idx in range(len(inshape)):
                    rgb_img_re = model.resize_layers[idx](rgb_img)
                    rgb_img_list.append(rgb_img_re)
                    pam_img_re = model.resize_layers[idx](pam_img)
                    pam_img_list.append(pam_img_re)

                rgb_img_warp_list, pred_transf_list = model(pam_img_list, rgb_img_list, registration=True)

                loss_term = Loss_img(rgb_img_list[1],rgb_img_warp_list[1])

                loss_val_log.update(loss_term.detach(), len(rgb_img))


            message = '2D Image Evaluation=== MSE_val_Image {loss1.ave:.5f} \t'.format(loss1=loss_val_log)
            LOGGER.info(message)

        if best_perf > loss_val_log.ave:
            best_perf = loss_val_log.ave
            checkpoint_dir = os.path.join(logdir, 'checkpoints')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            LOGGER.info('=> saving checkpoint to {}'.format(checkpoint_dir))
            states = dict()
            states['model_state_dict'] = model.state_dict()
            states['optimizer_state_dict'] = optimizer.state_dict()
            torch.save(states, os.path.join(checkpoint_dir, 'best_perf.tar'))

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
    LOGGER.info('Start Testing the saved model')

    # test the model on the best MAE performance or on the last epoch
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_perf.tar'))
    # checkpoint = torch.load(os.path.join(checkpoint_dir, 'last.tar'))

    model.load_state_dict(checkpoint['model_state_dict'])
    # model = vxm.networks.VxmDense.load(os.path.join(checkpoint_dir, 'last.tar'),device)
    model = model.cuda(device)
    LOGGER.info(f'---------------Finishing loading models----------------')

    mse_log = AverageMeter()

    with torch.no_grad():
        model.eval()
        for it, batch in enumerate(test_dataloader, 0):
            rgb_img = batch['rgb_img'].to(torch.float32).cuda(device)
            pam_img = batch['pam_img'].to(torch.float32).cuda(device)
            pam_depth = batch['pam_depth'].to(torch.float32).cuda(device)
            transf = batch['gt'].to(torch.float32).cuda(device)
            batch_size = len(rgb_img)
            # =========feed into the network=========================
            rgb_img_list = []
            pam_img_list = []
            for idx in range(len(inshape)):
                rgb_img_re = model.resize_layers[idx](rgb_img)
                rgb_img_list.append(rgb_img_re)
                pam_img_re = model.resize_layers[idx](pam_img)
                pam_img_list.append(pam_img_re)

            rgb_img_warp_list, pred_transf_list = model(pam_img_list, rgb_img_list, registration=True)

            mse_term = Loss_img(rgb_img_list[1],rgb_img_warp_list[1])
            mse_log.update(mse_term.item(),len(rgb_img))
        message = '2D Image Evaluation=== MSE_test: {loss1.ave:.5f} '.format(loss1=mse_log)
        LOGGER.info(message)
    LOGGER.info('Finish Testing')

def _test_offline(checkpoint=''):
    args = get_args()
    if checkpoint:
        args.load_model = checkpoint
    LOGGER = ConsoleLogger('RegHie_test_'+args.trial, 'test')
    logdir = LOGGER.getLogFolder()
    LOGGER.info(args)

    # ====================================================================
    weak_label = 0
    save_gray = 1


    # for in-vivo data
    rgb_img_folder = '/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Regis_cut/split_nopad_3/color_gray_pretrain_train'
    # pam_img_folder = '/home/work/Yuxuan/Code/Seg_PA_2D/experiments/Reg_test_baseline/2_0.1_gray'
    pam_img_folder = '/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Regis_cut/split_nopad_3/pam_gray_pretrain_train'
    regis_img_folder = ''

    # rgb_img_folder = '/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Regis_cut/split_nopad_3/color_gray_pretrain_test'
    # pam_img_folder = '/home/work/Yuxuan/Code/Seg_PA_2D/experiments/Reg_test_baseline/2_0.1_gray'
    # pam_img_folder = '/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Regis_cut/split_nopad_3/pam_gray_pretrain_test'
    # regis_img_folder = '/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Regis_cut/split_nopad_3/regis_results_mat_test'
    # regis_img_folder = '/home/work/Yuxuan/Code/Seg_PA_2D/in_vivo_test/regis_data/affine_nonlinear/regis_results_vis_test_nonlinear'


    # rgb_img_folder = '/home/work/Yuxuan/Code/Seg_PA_2D/in_vivo_test/regis_data/intra_subject/rgb_gray_pad_test'
    # pam_img_folder = '/home/work/Yuxuan/Code/Seg_PA_2D/in_vivo_test/regis_data/intra_subject/pam_gray_pad_test'
    # regis_img_folder = '/home/work/Yuxuan/Code/Seg_PA_2D/in_vivo_test/regis_data/intra_subject/regis_results_vis_test'


    '''
    # for synthetic data
    rgb_img_folder = '/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Syn_gen/Data_Syn_III/split/test/rgb_seg_pretrain'
    pam_img_folder = '/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Syn_gen/Data_Syn_III/split/test/pam_seg'
    # pam_img_folder = '/home/work/Yuxuan/Code/Seg_PA_2D/Aff_trans/experiments/Reg_test_affine/5_gray'
    pam_depth_folder = '/home/work/Yuxuan/Data/Photoacoustic/Seg_PAM_Data/Syn_gen/Data_Syn_III/split/test/pam'
    regis_img_folder = ''
    '''






    img_listdir = sorted(os.listdir(rgb_img_folder))


    device = torch.device(f'cuda:{args.gpu}')
    # device = torch.device('cpu')
    # model
    if args.model_type == 'hierarchical':
        bidir = args.bidir
        inshape = [32,64,128]
        # inshape = [32,64]
        enc_nf = args.enc if args.enc else [16, 32, 32, 32]
        dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]


        model = vxm.networks.VxmHieDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=bidir,
            int_steps=args.int_steps,
            int_downsize=args.int_downsize
        )

        LOGGER.info('Initial Model Finished : ' + args.model_type + ' Image size : ' + str(
            args.image_size) + ' Downsampling factor : ' + str(args.int_downsize) + ' If use depth : ' + str(
            args.if_depth_concat))

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



            if args.if_rgb_seg == 1:
                rgb_img_info[rgb_img_info > 0.1] = 1.0

            # rgb_img_info = rgb_img_info ** 0.5

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

            render_img_small = np.zeros((h_ori, w_ori, 3))
            render_img_small[:, :, 0] = abs(rgb_img_info)
            render_img_small[:, :, 1] = abs(pam_img_info)

            render_img_big = np.zeros((h_ori, w_ori, 3))
            render_img_big[:, :, 0] = abs(rgb_img_info)
            render_img_big[:, :, 1] = abs(img_warp_big)

            # render_img_small = np.zeros((h_ori, w_ori, 3))
            # render_img_small[:, :, 1] = abs(rgb_img_info)
            # render_img_small[:, :, 2] = abs(rgb_img_info)
            # render_img_small[:, :, 0] = abs(pam_img_info)
            #
            # render_img_big = np.zeros((h_ori, w_ori, 3))
            # render_img_big[:, :, 1] = abs(rgb_img_info)
            # render_img_big[:, :, 2] = abs(rgb_img_info)
            # render_img_big[:, :, 0] = abs(img_warp_big)

            # plt.imsave(os.path.join(logdir,filename+'_s.png'),render_img_small/np.max(render_img_small))
            # plt.imsave(os.path.join(logdir,filename+'_b.png'),render_img_big/np.max(render_img_big))

            np.save(os.path.join(logdir,filename+'.npy'),pred_transf)

            if save_gray == 1:
                # Image.fromarray(np.array(255 * (img_warp_big[0,0]/np.max(img_warp_big)),dtype=np.uint8)).save(os.path.join(logdir,filename+'_g.png'))
                Image.fromarray(np.array(255 * (img_warp_big[0,0]/np.max(img_warp_big)),dtype=np.uint8)).save(os.path.join(logdir,filename+'.png'))



        time_list = np.array(time_list)
        time_avg = np.mean(time_list)
        LOGGER.info('Average Time: {:.5f}'.format(time_avg))
        LOGGER.info('Fininshing.')



if __name__ == "__main__":
    args = get_args()

    if args.mode == 'train':
        train()
    if args.mode == 'test_offline':
        _test_offline()

    # check the model loading
    # args.image_size = 128
    # bidir = args.bidir
    # inshape = [args.image_size, args.image_size]
    # enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    # dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]
    #
    # model = vxm.networks.VxmDense(
    #     inshape=inshape,
    #     nb_unet_features=[enc_nf, dec_nf],
    #     bidir=bidir,
    #     int_steps=args.int_steps,
    #     int_downsize=args.int_downsize
    # )
    # model_path = '/home/work/Yuxuan/Code/Seg_PA_2D/experiments/Regis_baseline/2_0.1/checkpoints/last.tar'
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint['model_state_dict'])

     # check the dataset
    # import matplotlib.pyplot as plt
    # args.image_size = 256
    # args.dataset_type = 'syn'
    # # args.if_rgb_seg = 1
    # # dataset = PA_dataset(args.train_rgb_img_root, args.train_pam_img_root,args.train_pam_depth_root, args.train_data_root,args,stage='Train')
    # # dataset = PA_dataset(args.train_rgb_img_root_syn, args.train_pam_img_root_syn, args.train_pam_depth_root_syn,
    # #                      args.train_data_root_syn, args, stage='Train')
    # # dataset = PA_dataset(args.test_rgb_img_root, args.test_pam_img_root, args.test_pam_depth_root,
    # #                      args.test_data_root, args, stage='Test')
    # dataset = PA_dataset(args.test_rgb_img_root_syn, args.test_pam_img_root_syn, args.test_pam_depth_root_syn,
    #                      args.test_data_root_syn, args, stage='Test')
    # # dataset = PA_dataset(args.val_img_root, args.val_data_root,args,stage='Test')
    # a = dataset[10]
    # plt.figure()
    # plt.subplot(1,3,1)
    # plt.imshow(a['rgb_img'][0],cmap='gray')
    # plt.subplot(1,3,2)
    # plt.imshow(a['pam_img'][0],cmap='gray')
    # plt.subplot(1,3,3)
    # plt.imshow(a['pam_depth'][0],cmap='gray')
    # plt.show()













