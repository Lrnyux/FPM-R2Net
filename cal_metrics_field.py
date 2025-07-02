
from PIL import Image
import os
from metrics import *




# for collected data
gt_folder = ''
sepe_folder = ''
pred_folder = ''


filelist = sorted(os.listdir(gt_folder))
jac = np.zeros((len(filelist),1))
total_pix = np.zeros((len(filelist),1))


for idx in range(len(filelist)):
    filename = filelist[idx].replace('.png','')
    gt_root = os.path.join(gt_folder,filelist[idx])
    sep_root = os.path.join(sepe_folder,filelist[idx])
    pred_img_root = os.path.join(pred_folder,filename+'.png')
    pred_f_root = os.path.join(pred_folder,filename+'.npy')

    gt_img = np.array(Image.open(gt_root))/255.0
    sep_img = np.array(Image.open(sep_root))/255.0
    pred_img = np.array(Image.open(pred_img_root))/255.0
    pred_f = np.load(pred_f_root)

    h,w = gt_img.shape
    total_pixs = h * w
    pred_f[:,:,0] = pred_f[:,:,0] * (h-1)
    pred_f[:,:,1] = pred_f[:,:,1] * (w-1)

    # binary metrics
    gt_thres = 0.5
    sep_thres = 0.5
    pred_thres = 0.5

    gt_mask = np.zeros_like(gt_img)
    gt_mask[gt_img > gt_thres] = 1.0
    sep_mask = np.zeros_like(sep_img)
    sep_mask[sep_img > sep_thres] = 1.0
    pred_mask = np.zeros_like(pred_img)
    pred_mask[pred_img > pred_thres] = 1.0


    pred_mask_or = pred_mask * sep_mask
    # pred_mask_or = gt_mask

    eval_metrics_field = get_field_metrics(pred_f, pred_mask_or)

    jac[idx,0] = eval_metrics_field['J_less_zero'] * total_pixs
    total_pix[idx,0] = total_pixs

print(np.sum(jac)/np.sum(total_pix))
print(np.mean(jac/total_pix))
print(np.mean(jac))




