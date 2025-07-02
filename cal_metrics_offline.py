import numpy as np
from PIL import Image
import os


save_results = 0

# for collected data
gt_folder = ''
sepe_folder = ''
pred_folder = ''


filelist = sorted(os.listdir(gt_folder))

select = 0

log_name = ['IOU','Dice','IOU_s','Dice_s','IOU-vis','Dice-vis','IOU_s-vis','Dice_s-vis','IOU-invis','Dice-invis','IOU_s-invis','Dice_s-invis']
log_array = np.zeros((len(filelist),len(log_name)))

for idx in range(len(filelist)):
    gt_root = os.path.join(gt_folder,filelist[idx])
    sep_root = os.path.join(sepe_folder,filelist[idx])
    pred_root = os.path.join(pred_folder,filelist[idx])

    gt_img = np.array(Image.open(gt_root))/255.0
    sep_img = np.array(Image.open(sep_root))/255.0
    pred_img = np.array(Image.open(pred_root))/255.0


    # binary metrics
    gt_thres = 0.5
    sep_thres = 0.5
    pred_thres = 0.5


    # for overall part
    gt_mask = np.zeros_like(gt_img)
    gt_mask[gt_img > gt_thres] = 1.0
    sep_mask = np.zeros_like(sep_img)
    sep_mask[sep_img > sep_thres] = 1.0
    pred_mask = np.zeros_like(pred_img)
    pred_mask[pred_img > pred_thres] = 1.0

    tp = (pred_mask * gt_mask).sum()
    tn = ((1 - pred_mask) * (1 - gt_mask)).sum()
    fp = ((1 - gt_mask) * pred_mask).sum()
    fn = ((1 - pred_mask) * gt_mask).sum()

    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    iou = tp / (tp + fp + fn)

    dice = 2 * pre * sen / (pre + sen)

    iou_s = np.minimum(pred_img, gt_img).sum() / np.maximum(pred_img, gt_img).sum()
    dice_s = 2 * np.minimum(pred_img,gt_img).sum()/(pred_img.sum()+gt_img.sum())

    # for seen part
    gt_mask = np.zeros_like(gt_img)
    gt_mask[gt_img > gt_thres] = 1.0
    sep_mask = np.zeros_like(sep_img)
    sep_mask[sep_img > sep_thres] = 1.0
    pred_mask = np.zeros_like(pred_img)
    pred_mask[pred_img > pred_thres] = 1.0

    pred_mask = pred_mask * sep_mask
    gt_mask = gt_mask * sep_mask

    tp_se = (pred_mask * gt_mask).sum()
    tn_se = ((1 - pred_mask) * (1 - gt_mask)).sum()
    fp_se = ((1 - gt_mask) * pred_mask).sum()
    fn_se = ((1 - pred_mask) * gt_mask).sum()

    acc_se = (tp_se + tn_se) / (tp_se + fp_se + fn_se + tn_se)
    pre_se = tp_se / (tp_se + fp_se)
    sen_se = tp_se / (tp_se + fn_se)
    spe_se = tn_se / (tn_se + fp_se)
    iou_se = tp_se / (tp_se + fp_se + fn_se)

    dice_se = 2 * pre_se * sen_se / (pre_se + sen_se)

    iou_s_se = np.minimum(pred_img * sep_mask, gt_img * sep_mask).sum() / np.maximum(pred_img * sep_mask, gt_img * sep_mask).sum()
    dice_s_se = 2 * np.minimum(pred_img * sep_mask, gt_img * sep_mask).sum() / ((pred_img * sep_mask).sum() + (gt_img * sep_mask).sum())

    # for unseen part
    gt_mask = np.zeros_like(gt_img)
    gt_mask[gt_img > gt_thres] = 1.0
    sep_mask = np.zeros_like(sep_img)
    sep_mask[sep_img > sep_thres] = 1.0
    sep_mask = 1.0 - sep_mask
    pred_mask = np.zeros_like(pred_img)
    pred_mask[pred_img > pred_thres] = 1.0

    pred_mask = pred_mask * sep_mask
    gt_mask = gt_mask * sep_mask

    tp_unse = (pred_mask * gt_mask).sum()
    tn_unse = ((1 - pred_mask) * (1 - gt_mask)).sum()
    fp_unse = ((1 - gt_mask) * pred_mask).sum()
    fn_unse = ((1 - pred_mask) * gt_mask).sum()

    acc_unse = (tp_unse + tn_unse) / (tp_unse + fp_unse + fn_unse + tn_unse)
    pre_unse = tp_unse / (tp_unse + fp_unse)
    sen_unse = tp_unse / (tp_unse + fn_unse)
    spe_unse = tn_unse / (tn_unse + fp_unse)
    iou_unse = tp_unse / (tp_unse + fp_unse + fn_unse)

    dice_unse = 2 * pre_unse * sen_unse / (pre_unse + sen_unse)

    iou_s_unse = np.minimum(pred_img * sep_mask, gt_img * sep_mask).sum() / np.maximum(pred_img * sep_mask,
                                                                                     gt_img * sep_mask).sum()
    dice_s_unse = 2 * np.minimum(pred_img * sep_mask, gt_img * sep_mask).sum() / (
                (pred_img * sep_mask).sum() + (gt_img * sep_mask).sum())

    log_array[idx,:] = [iou,dice,iou_s,dice_s,iou_se,dice_se,iou_s_se,dice_s_se,iou_unse,dice_unse,iou_s_unse,dice_s_unse]



log_array[np.isnan(log_array)]=0
log_array = np.mean(log_array,axis=0).reshape(-1)
for kdx in range(len(log_name)):
    print(log_name[kdx]+':{:.5f}'.format(log_array[kdx]))
info = ''
for kdx in [4,5,6,7,8,9,10,11,0,1,2,3]:
    info = info + ' & {:.4f} '.format(log_array[kdx])
print(info)

