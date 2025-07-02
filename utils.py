import logging
import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image
import math
import torch.nn.functional as F
import torch
import lpips
import SimpleITK as  sitk

# Generallly a refers to prediction and b refers to groundtruth

def calc_rmse(a, b, minmax=np.array([0,1])):

    a = a * (minmax[1] - minmax[0]) + minmax[0]
    b = b * (minmax[1] - minmax[0]) + minmax[0]

    return np.sqrt(np.mean(np.power(a - b, 2)))


def calc_rmse_3D(a, b):

    # the voxel of a and b should be normed to [0,1]
    a = (a - np.min(a)) / (np.max(a) - np.min(a))
    b = (b - np.min(b)) / (np.max(b) - np.min(b))

    return np.sqrt(np.mean(np.power(a - b, 2)))


def cal_VOP_3D(a,b,thres=0.0):
    # the voxel of a and b should be normed to [0,1]
    # b should be the groundtruth data
    a = (a - np.min(a)) / (np.max(a) - np.min(a))
    b = (b - np.min(b)) / (np.max(b) - np.min(b))

    a[a<thres]=0
    a[a>=thres]=1
    b[b<thres]=0
    b[b>=thres]=1

    inter = a*b
    return np.sum(inter)/np.sum(b)


def calc_psnr(a, b,minmax=np.array([0,1])):
    # img1 and img2 have range [0, 255]
    a = a * (minmax[1] - minmax[0]) + minmax[0]
    b = b * (minmax[1] - minmax[0]) + minmax[0]
    img1 = (a/np.max(a))*255
    img2 = (b/np.max(b))*255
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calc_psnr_3d(a, b):
    # img1 and img2 have range [0, 255]
    a = (a - np.min(a)) / (np.max(a) - np.min(a))
    b = (b - np.min(b)) / (np.max(b) - np.min(b))
    img1 = a*255
    img2 = b*255
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calc_ssim(a, b,minmax=np.array([0,1])):
    a = a * (minmax[1] - minmax[0]) + minmax[0]
    b = b * (minmax[1] - minmax[0]) + minmax[0]
    img1 = (a / np.max(a)) * 255
    img2 = (b / np.max(b)) * 255
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calc_lpips(a, b, minmax=np.array([0,1]),loss=lpips.LPIPS(net='alex')):
    H,W = a.shape
    a = np.clip(a,-1,1)
    b = np.clip(b,-1,1)
    a = a.reshape(1,1,H,W)
    a = np.repeat(a,3,axis=1)
    b = b.reshape(1, 1, H, W)
    b = np.repeat(b, 3, axis=1)
    with torch.no_grad():
        return loss(torch.from_numpy(a).to(torch.float32),torch.from_numpy(b).to(torch.float32))

def calc_snr(img, mask, minmax=np.array([0,1])):
    # for the mask, 1 represents the foreground and 0 represents the background
    img = img * (minmax[1] - minmax[0]) + minmax[0]
    f=np.where(mask==1)
    img_f = img[f]
    b=np.where(mask==0)
    img_b = img[b]
    # print(img_f.shape,img_b.shape)
    snr = 20 * math.log10(np.mean(img_f)/np.std(img_b))

    return snr

def calc_cnr(img, mask, minmax=np.array([0,1])):
    # for the mask, 1 represents the foreground and 0 represents the background
    img = img * (minmax[1] - minmax[0]) + minmax[0]
    f=np.where(mask==1)
    img_f = img[f]
    b=np.where(mask==0)
    img_b = img[b]
    cnr = 20 * math.log10(abs(np.mean(img_f)-np.mean(img_b))/np.sqrt(np.var(img_b)+np.var(img_f)))

    return cnr

class CustomFormatter(logging.Formatter):
    DATE = '\033[94m'
    GREEN = '\033[92m'
    WHITE = '\033[0m'
    WARNING = '\033[93m'
    RED = '\033[91m'

    def __init__(self):
        orig_fmt = "%(name)s: %(message)s"
        datefmt = "%H:%M:%S"
        super().__init__(orig_fmt, datefmt)

    def format(self, record):
        color = self.WHITE
        if record.levelno == logging.INFO:
            color = self.GREEN
        if record.levelno == logging.WARN:
            color = self.WARNING
        if record.levelno == logging.ERROR:
            color = self.RED
        self._style._fmt = "{}%(asctime)s {}[%(levelname)s]{} {}: %(message)s".format(
            self.DATE, color, self.DATE, self.WHITE)
        return logging.Formatter.format(self, record)


class ConsoleLogger():
    def __init__(self, training_type, phase='train'):
        super().__init__()
        self._logger = logging.getLogger(training_type)
        self._logger.setLevel(logging.INFO)
        formatter = CustomFormatter()
        console_log = logging.StreamHandler()
        console_log.setLevel(logging.INFO)
        console_log.setFormatter(formatter)
        self._logger.addHandler(console_log)
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
        self.logfile_dir = os.path.join('experiments/', training_type, time_str)
        os.makedirs(self.logfile_dir)
        logfile = os.path.join(self.logfile_dir, f'{phase}.log')
        file_log = logging.FileHandler(logfile, mode='a')
        file_log.setLevel(logging.INFO)
        file_log.setFormatter(formatter)
        self._logger.addHandler(file_log)

    def info(self, *args, **kwargs):
        """info"""
        self._logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        """warning"""
        self._logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        """error"""
        self._logger.error(*args, **kwargs)
        exit(-1)

    def getLogFolder(self):
        return self.logfile_dir


class AverageMeter():
    def __init__(self):
        self.val = 0
        self.count = 0
        self.sum = 0
        self.ave = 0

    def update(self, val, num=1):
        self.count = self.count + num
        self.val = val
        self.sum = self.sum + num * val
        self.ave = self.sum / self.count if self.count != 0 else 0.0

def save_results(gt,pred,down,save_root_prefix,minmax=[0,1],gray=1):
    image_pred = pred * (minmax[1] - minmax[0]) + minmax[0]
    image_pred[image_pred < 0] = 0
    # image_pred[image_pred > 255] = 255
    image_gt = gt * (minmax[1] - minmax[0]) + minmax[0]
    image_gt[image_gt<0] = 0
    image_down = down * (minmax[1] - minmax[0]) + minmax[0]
    image_down[image_down < 0] = 0
    # image_down[image_down > 255] = 255
    if gray == 1:

        image_pred = np.array(image_pred,dtype=np.uint8)
        image_pred = Image.fromarray(image_pred)
        image_pred.save(save_root_prefix+'_pred.png')



        image_gt = np.array(image_gt,dtype=np.uint8)
        image_gt = Image.fromarray(image_gt)
        image_gt.save(save_root_prefix + '_gt.png')


        image_down = np.array(image_down, dtype=np.uint8)
        image_down = Image.fromarray(image_down)
        image_down.save(save_root_prefix + '_bi.png')

    if gray == 0:
        image_pred = np.array(image_pred, dtype=np.float64)
        plt.imsave(save_root_prefix + '_pred.png',image_pred,cmap='jet')

        image_gt = np.array(image_gt, dtype=np.float64)
        plt.imsave(save_root_prefix + '_gt.png', image_gt, cmap='hot')

        image_down = np.array(image_down, dtype=np.float64)
        plt.imsave(save_root_prefix + '_bi.png',image_down,cmap='hot')

def save_results_reg(gt_f,pred_f,img_f,img_m,img_f_warp,save_root_prefix):

    h,w = img_f.shape
    pred_f[0,:,:] = pred_f[0,:,:] / (h-1)
    pred_f[1,:,:] = pred_f[1,:,:] / (w-1)
    pred_f = np.transpose(pred_f,[1,2,0])

    Image.fromarray(np.array(255 * img_f,dtype=np.uint8)).save(save_root_prefix+'_fix.png')
    Image.fromarray(np.array(255 * img_m,dtype=np.uint8)).save(save_root_prefix+'_mov.png')
    np.save(save_root_prefix+'_gtf.npy',gt_f)
    np.save(save_root_prefix+'_predf.npy',pred_f)

    h,w = img_m.shape
    torch_gt_f = torch.Tensor(gt_f).unsqueeze_(0).expand(1, -1, -1, -1)
    torch_pred_f = torch.Tensor(pred_f).unsqueeze_(0).expand(1,-1,-1,-1)
    img_m_th = torch.Tensor(img_m).view(1, 1, h, w).expand(1, -1, -1, -1)

    img_out_gt = warp_image(img_m_th, torch_gt_f).numpy()
    img_out_pred = warp_image(img_m_th, torch_pred_f).numpy()

    render_img_gt = np.zeros((h, w, 3))
    render_img_gt[:, :, 0] = abs(img_f)
    render_img_gt[:, :, 1] = abs(img_out_gt)

    render_img_pred = np.zeros((h, w, 3))
    render_img_pred[:, :, 0] = abs(img_f)
    render_img_pred[:, :, 1] = abs(img_out_pred)

    render_img_netout = np.zeros((h, w, 3))
    render_img_netout[:, :, 0] = abs(img_f)
    render_img_netout[:, :, 1] = abs(img_f_warp)

    render_img_ori = np.zeros((h, w, 3))
    render_img_ori[:, :, 0] = abs(img_f)
    render_img_ori[:, :, 1] = abs(img_m)

    plt.imsave(save_root_prefix+'_gt.png',render_img_gt/np.max(render_img_gt))
    plt.imsave(save_root_prefix+'_pred.png',render_img_pred/np.max(render_img_pred))
    plt.imsave(save_root_prefix + '_out.png', render_img_netout / np.max(render_img_netout))
    plt.imsave(save_root_prefix + '_ori.png', render_img_ori / np.max(render_img_ori))




def save_results_3d(gt,pred,down,save_root_prefix):
    gt_save = (gt - np.min(gt)) / (np.max(gt) - np.min(gt))
    pred_save = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    down_save = (down - np.min(down)) / (np.max(down) - np.min(down))

    sitk.WriteImage(sitk.GetImageFromArray(gt_save),save_root_prefix+'_gt.nii')
    sitk.WriteImage(sitk.GetImageFromArray(pred_save), save_root_prefix + '_pred.nii')
    sitk.WriteImage(sitk.GetImageFromArray(down_save), save_root_prefix + '_down.nii')


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def plot_deformation_field(deformation_array, rstride=3, cstride=3, ax=None, **kwargs):
    '''
     Args:
        deformation_array: 2-dim deformation field [H,W,2].
        rstride defines how many rows to skip
        cstride defines how many columns to skip
    '''
    if ax is None: ax = plt.gca()
    # some defaults
    args = {
        'color':'k'
    }
    args.update(kwargs)

    patch_size = deformation_array.shape[0:2]
    # x0 = np.arange(0,patch_size[0])
    # y0 = np.arange(0,patch_size[1])
    # X0, Y0 = np.meshgrid(x0,y0,indexing='ij')
    # phiX00 = displacement_array[:,:,0]+Y0
    # phiX10 = displacement_array[:,:,1]+X0
    phiX00 = deformation_array[:,:,1]
    phiX10 = deformation_array[:,:,0]
    # plot rows
    for i in range(0, phiX00.shape[0], rstride):
        ax.plot(phiX00[i,:], phiX10[i,:], **args)
    if i < phiX00.shape[0]-1:
        ax.plot(phiX00[-1,:], phiX10[-1,:], **args)
    # plot columns
    for j in range(0,phiX00.shape[1],cstride):
        ax.plot(phiX00[:,j], phiX10[:,j], **args)
    if j < phiX00.shape[1]-1:
        ax.plot(phiX00[:,-1], phiX10[:,-1], **args)
    ax.invert_yaxis()

def compute_grid(image_size, dtype=torch.float32, device='cpu'):

    dim = len(image_size)

    if dim == 2:
        nx = image_size[0]
        ny = image_size[1]

        x = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)
        y = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)

        x = x.expand(nx, -1)
        y = y.expand(ny, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(3)
        y.unsqueeze_(0).unsqueeze_(3)

        return torch.cat((x, y), 3).to(dtype=dtype, device=device)

    elif dim == 3:
        nz = image_size[0]
        ny = image_size[1]
        nx = image_size[2]

        x = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)
        y = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)
        z = torch.linspace(-1, 1, steps=nz).to(dtype=dtype)

        x = x.expand(ny, -1).expand(nz, -1, -1)
        y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2)
        z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(4)
        y.unsqueeze_(0).unsqueeze_(4)
        z.unsqueeze_(0).unsqueeze_(4)

        return torch.cat((x, y, z), 4).to(dtype=dtype, device=device)
    else:
        print("Error " + dim + "is not a valid grid type")

def warp_image(image, displacement):

    image_size = image.shape[2:]

    grid = compute_grid(image_size, dtype=image.dtype, device=image.device)

    # warp image
    warped_image = F.grid_sample(image, displacement + grid)

    return warped_image

