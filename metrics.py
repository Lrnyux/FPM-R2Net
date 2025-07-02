import torch
import numpy as np
import cv2
import pystrum.pynd.ndutils as nd

def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """
    h,w,c = disp.shape
    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)
    # J = np.gradient(grid)

    dfdx = J[0]
    dfdy = J[1]

    return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

def get_metrics(predict, target):

    '''
    The predict  and target images are binary images with np.ndarray format
    1 represents foreground and 0 represents background
    '''
    if torch.is_tensor(predict):
        predict_b = predict.cpu().detach().numpy().flatten()
    else :
        predict_b = predict.flatten()


    if torch.is_tensor(target):
        target = target.cpu().detach().numpy().flatten()
    else:
        target = target.flatten()


    tp = (predict_b * target).sum()
    tn = ((1 - predict_b) * (1 - target)).sum()
    fp = ((1 - target) * predict_b).sum()
    fn = ((1 - predict_b) * target).sum()
    # auc = roc_auc_score(target, predict, multi_class='ovo')
    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    iou = tp / (tp + fp + fn)
    f1 = 2 * pre * sen / (pre + sen)
    return {
        # "AUC": np.round(auc, 4),
        "F1": np.round(f1, 4),
        "Acc": np.round(acc, 4),
        "Sen": np.round(sen, 4),
        "Spe": np.round(spe, 4),
        "pre": np.round(pre, 4),
        "IOU": np.round(iou, 4),
    }

def get_field_metrics(transf,mask):
    # shape of the transf [H,W,2] with range [0,H]
    h,w,c = transf.shape
    assert c == 2
    J_sample = jacobian_determinant(transf)

    J_less_zero = np.zeros_like(J_sample)
    J_less_zero[(J_sample*mask)<0] = 1
    J_less_number = np.sum(J_less_zero)
    J_less_percent = J_less_number / (h * w)


    index = np.where(mask==0)
    J_sample_roi = J_sample[index]


    return {
        'J_less_zero': np.round(J_less_percent,4),
        'J_mean': np.round(np.mean(J_sample_roi),4),
        'J_std': np.round(np.std(J_sample_roi),4),
    }




def count_connect_component(predict, target, connectivity=8):
    if torch.is_tensor(predict):
        predict = predict.cpu().detach().numpy()
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    pre_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        predict, dtype=np.uint8)*255, connectivity=connectivity)
    gt_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        target, dtype=np.uint8)*255, connectivity=connectivity)
    return pre_n/gt_n



def dual_threshold(predict_prob,th_high=0.7,th_low=0.3):
    if torch.is_tensor(predict_prob):
        predict_prob = predict_prob.cpu().numpy()

    mask_info = predict_prob.copy()
    mask_info[mask_info >= th_high] = 1.0
    h, w = mask_info.shape
    bin = np.where(mask_info >= th_high, 1.0, 0).astype(np.float64)
    gbin = bin.copy()
    gbin_pre = gbin - 1
    while (gbin_pre.sum() != gbin.sum()):
        gbin_pre = gbin.copy()
        for i in range(0, h - 1):
            for j in range(0, w - 1):
                if gbin[i][j] == 0 and mask_info[i][j] < th_high and mask_info[i][j] >= th_low:
                    if gbin[i - 1][j - 1] or gbin[i - 1][j] or gbin[i - 1][j + 1] or gbin[i][j - 1] or gbin[i][j + 1] or \
                            gbin[i + 1][j - 1] or gbin[i + 1][j] or gbin[i + 1][j + 1]:
                        gbin[i][j] = 1
    return gbin

def binary_threshold(predict_prob,th):
    if torch.is_tensor(predict_prob):
        predict_prob = predict_prob.cpu().numpy()

    mask_info = predict_prob.copy()
    mask_info[mask_info >= th ]=1.0
    mask_info[mask_info < th] = 0
    return mask_info
