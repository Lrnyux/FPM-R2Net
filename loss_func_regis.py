import torch.nn.functional as F
import torch
import numpy as np
from torch.nn.modules.loss import _Loss
import math


class mse(_Loss):
    def __init__(self, args, **kwargs):
        super(mse, self).__init__()
        self.args = args

    def forward(self, pred, gt, **kwargs):
        return {'mse_loss':F.mse_loss(pred,gt)}

class ncc(_Loss):
    def __init__(self, args, **kwargs):
        super(ncc,self).__init__()
        self.args = args
    def forward(self, pred, gt, **kwargs):
        Ii = gt
        Ji = pred
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(pred.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return {'ncc_loss':-torch.mean(cc)}


class dice(_Loss):
    def __init__(self, args, **kwargs):
        super(dice, self).__init__()
        self.args = args


    def forward(self, pred, gt, **kwargs):

        ndims = len(list(pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (gt * pred).sum(dim=vol_axes)
        bottom = torch.clamp((gt + pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)

        return {'dice_loss':-dice}

class grad(_Loss):
    def __init__(self, args, **kwargs):
        super(grad, self).__init__()
        self.args = args
        self.penalty = 'l1'

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def forward(self, pred, gt, **kwargs):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        return {'grad_loss':grad.mean()}

class mse_reg(_Loss):
    def __init__(self, args, **kwargs):
        super(mse_reg, self).__init__()
        self.args = args
        self.mse_loss = mse(self.args)
        self.grad_loss = grad(self.args)


    def forward(self, img_pred, img_gt, f_pred=None, f_gt=None, **kwargs):
        mse_term_img = self.mse_loss(img_pred,img_gt)['mse_loss']
        if (f_pred != None) and (f_gt != None):
            mse_term_f = self.mse_loss(f_pred,f_gt)['mse_loss']
        else:
            mse_term_f = None
        grad_term = self.grad_loss(img_pred,img_gt)['grad_loss']
        return {'mse_loss_img':mse_term_img,'mse_loss_f':mse_term_f,'grad_loss':grad_term}

if __name__ == '__main__':
    img_pred = torch.ones((2,1,256,256))
    img_gt = torch.zeros((2,1,256,256))
    f_pred = torch.ones((2,256,256,2))/1.5
    f_gt = torch.ones((2,256,256,2))/1.3
    Loss_mse_reg = mse_reg(args=None)
    loss = Loss_mse_reg(img_pred,img_gt,f_pred,f_gt)
    print('finish')

