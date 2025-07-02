import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math
import layers
from modelio import LoadableModel, store_config_args
import skimage.transform as transform


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class Conv2dBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,stride=1,padding=1,kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.ac1 = nn.ReLU()
        self.dp1 = nn.Dropout()
        self.conv2 = nn.Conv2d(out_channels, out_channels, stride=1, padding=1, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ac2 = nn.ReLU()
        self.avg = nn.AvgPool2d(kernel_size=2)
    def forward(self,x):
        x = self.conv1(x)
        x = self.dp1(self.ac1(self.bn1(x)))
        x = self.conv2(x)
        x = self.ac2(self.bn2(x))
        out = self.avg(x)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Unet(nn.Module):
    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3]

        self.half_res = half_res
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x



class ModelHieDense(LoadableModel):

    def __init__(self,
                 inshape=[64,128,256],
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        super().__init__()

        self.training = True
        self.inshape = inshape

        ndims = 2
        num_stage = len(inshape)
        self.num_stage = num_stage
        self.unet_model = Unet(
            inshape=[inshape[0],inshape[0]],
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )
        tal_num = sum(p.numel() for p in self.unet_model.parameters())
        print('The total number of parameters of unet is : {}'.format(tal_num))


        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None


        self.bidir = bidir
        self.integrate_layers = []
        self.transformer_layers = []
        self.resize_layers = []


        for kdx in range(num_stage):
            input_shape = [inshape[kdx],inshape[kdx]]
            down_shape = [int(dim / int_downsize) for dim in input_shape]
            self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
            self.transformer = layers.SpatialTransformer(input_shape)
            self.integrate_layers.append(self.integrate)
            self.transformer_layers.append(self.transformer)
            self.resize_layers.append(transforms.Resize(input_shape))


    def forward(self, source, target, registration=False):
        y_source_list = []
        preint_flow_list = []
        pos_flow_list = []
        img_src = source[0]
        pos_flow_cum = torch.zeros_like(img_src).expand(-1,2,-1,-1)
        for kdx in range(self.num_stage):
            img_tgt = target[kdx]
            x = torch.cat([img_src, img_tgt], dim=1)
            x = self.unet_model(x)
            flow_field = self.flow(x)
            pos_flow = flow_field
            if self.resize:
                pos_flow = self.resize(pos_flow)
            preint_flow = pos_flow

            if self.integrate:
                pos_flow = self.integrate_layers[kdx](pos_flow)
                if self.fullsize:
                    pos_flow = self.fullsize(pos_flow)

            y_source = self.transformer_layers[kdx](img_src, pos_flow)
            y_source_list.append(y_source)
            preint_flow_list.append(preint_flow)
            pos_flow_list.append(pos_flow)
            if kdx != self.num_stage-1:
                pos_flow_re = pos_flow / (self.inshape[kdx]-1)
                pos_flow_re = self.resize_layers[kdx+1](pos_flow_re)
                pos_flow_cum = self.resize_layers[kdx+1](pos_flow_cum)
                pos_flow_cum = pos_flow_cum + pos_flow_re
                pos_flow_re = pos_flow_cum * (self.inshape[kdx+1]-1)
                img_src = source[kdx+1]
                img_src = self.transformer_layers[kdx+1](img_src,pos_flow_re)

        if not registration:
            return (y_source_list, preint_flow_list, pos_flow_list)
        else:
            return y_source_list, pos_flow_list




