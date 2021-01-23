from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import random
import numpy as np

# set seed
'''
torch.manual_seed(8)
torch.cuda.manual_seed(8)
torch.cuda.manual_seed_all(8)
random.seed(8)
np.random.seed(8)
'''

import os
import math
import argparse
from random import choice
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from Normalize import Normalize
import cv2
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser(description='PyTorch Attack')
# parser.add_argument('--data', default='/home/chengyaya/data/code_attack/data/ImageNet', metavar='DIR', help='path to dataset')
parser.add_argument('--data', default='/home/chengyaya/data/code_attack/data/ImageNet_chosen_dn-res-incv3-vgg19_5perclass', metavar='DIR', help='path to dataset')
parser.add_argument('--mean', type=float, default=np.array([0.485, 0.456, 0.406]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.229, 0.224, 0.225]), help='std.')
parser.add_argument('--eps', default=0.07, type=float, metavar='N', help='epsilon for attack perturbation')
parser.add_argument('--decay', default=1.0, type=float, metavar='N', help='decay for attack momentum')
parser.add_argument('--iteration', default=20, type=int, metavar='N', help='number of attack iteration')
parser.add_argument('-b', '--batchsize', default=50, type=int, metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--size', default=224, type=int, metavar='N', help='the size of image')
parser.add_argument('--resize', default=299, type=int, metavar='N', help='the resize of image')
parser.add_argument('--prob', default=0.5, type=float, metavar='N', help='probability of using diverse inputs.')
parser.add_argument('--num', '--data_num', default=5000, type=int, metavar='N', help='the num of test images')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--mmd_gaussian_multi', type=float, default=1.0, help='the gaussian-multiplication in mmd kernels')
parser.add_argument('--kernel_type', type=str, default='gussi',  help='the type of kernel')
parser.add_argument('--c', type=float, default=0.0,  help='the param of poly kernel')
parser.add_argument('--style_weight', type=float, default=5.0, help='the weight for the style image')
parser.add_argument('--bn_loss', type=int, default=0, help='use bn loss instead of mmd loss')
parser.add_argument('--kernel_for_furthe', type=str, default='l_gussi', help='if use bn loss instead of mmd loss')
parser.add_argument('--secondTarClass', type=int, default=1, help='choose the second possible class as the target class, is not, randomly choose one')
parser.add_argument('--mmdMethod', type=int, default=1, help='1 mmd Method, 2 ori method, 3 gram Method')
parser.add_argument('--targetcls', type=int, default=2, help='select the target class indix, 2,10,100,500,1000')

layer_block_dict = {
    0: (1),
    1: (2),
    6: (6, 10),
    7: (6, 12),
    10:(6, 12, 14),
    15:(6, 12, 24),
    17:(6, 12, 24, 8),
    21: (6, 12, 24, 16),
}

class MMD_gussi(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 1):
        super(MMD_gussi, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=1, fix_sigma=None):
        batch_size = source.shape[0]
        channel = source.shape[1]
        h = source.shape[2]
        w = source.shape[3]
        source = source.view(batch_size, channel, h*w)
        target = target.view(batch_size, channel, h*w)
        source = source.permute(0, 2, 1)
        target = target.permute(0, 2, 1)

        # n_samples = int(source.size()[0])+int(target.size()[0])
        n_samples = int(source.size()[1])+int(target.size()[1])
        # total = torch.cat([source, target], dim=0)
        total = torch.cat([source, target], dim=1)
    
        # total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

        total0 = total.unsqueeze(1).expand(batch_size, int(total.size(1)), int(total.size(1)), int(total.size(2)))
        total1 = total.unsqueeze(2).expand(batch_size, int(total.size(1)), int(total.size(1)), int(total.size(2)))
        
        # L2_distance = ((total0-total1)**2).sum(2)
        L2_distance = ((total0-total1)**2).sum((3))

        # return sum(torch.exp(-L2_distance/sigma))

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance, axis=(1, 2), keepdim=True) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        n,c,h,w = source.shape
        batch_size = h*w
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:, :batch_size, :batch_size]
        YY = kernels[:, batch_size:, batch_size:]
        XY = kernels[:, :batch_size, batch_size:]
        YX = kernels[:, batch_size:, :batch_size]
        loss = torch.sum((XX + YY - XY -YX), dim=(1,2))/(c*c*w*w*h*h)
        return loss

def split_even_odd(x):
    """
    split a list into two different lists by the even and odd entries
    :param x: the list
    :return: two lists with even and odd entries of x respectively
    """
    n, M, c = x.size()
    # split even, odd
    n0 = M - M % 2
    return x[:, range(0, n0, 2), :], x[:, range(1, n0, 2), :], n0

def gaussian_kernel(diff_, gamma):
    """
    compute a Gaussian kernel for vector x and y
    :param x: data list
    :param y: data list
    :param gamma: parameter for the Gaussian kernel
    :return: the Gaussian kernel
    """
    # e^(-a * |x - y|^2)
    return torch.exp(-gamma * diff_)

def h(x_odd, y_odd, x_even, y_even, n0):
    """
    helper function for the MMD O(n) computation
    :param x_i: odd entries of x
    :param y_i: odd entries of y
    :param x_j: even entries of x
    :param y_j: even entries of y
    :param n0: the parameter for the Gaussian kernel
    :return: the value for the Gaussian kernel
    """
    # use variance as gamma
    diffx = torch.sum((x_even - x_odd)**2, axis=(2))
    diffy = torch.sum((y_even - y_odd)**2, axis=(2))
    diffxy = torch.sum((x_even - y_odd)**2, axis=(2))
    diffyx = torch.sum((y_even - x_odd)**2, axis=(2))
    gamma = args.mmd_gaussian_multi * n0 * 2 /(torch.sum(diffx, axis=1, keepdim=True) + torch.sum(diffy, axis=1, keepdim=True) + torch.sum(diffxy, axis=1, keepdim=True) + torch.sum(diffyx, axis=1, keepdim=True))

    # compute kernel values
    s1 = gaussian_kernel(diffx, gamma)
    s2 = gaussian_kernel(diffy, gamma)
    s3 = gaussian_kernel(diffxy, gamma)
    s4 = gaussian_kernel(diffyx, gamma)

    # return result of h
    s = s1 + s2 - s3 - s4
    return s

def MMD_linearTime_gussi(x, y):
    """
    compute the linear time O(n) approximation of the MMD
    :param x: source_feature
    :param y: target_feature
    :param alpha:
    :return:
    """
    # split tensors x and y channel-wise based on its index
    n, c, h_, w = x.size()
    x = x.view(n, c, h_*w)
    y = y.view(n, c, h_*w)
    # permute shape to [n, h*w, c]
    x = x.permute(0, 2, 1)
    y = y.permute(0, 2, 1)
    x_even, x_odd, n0 = split_even_odd(x)
    y_even, y_odd, n0 = split_even_odd(y)

    # return mmd approximation
    return torch.abs(torch.sum(h(x_odd, y_odd, x_even, y_even, n0), axis=1))/(c*c*h_*h_*w*w)

def MMD_poly(source, target, c):
    n, ch, h_, w = source.size()
    source = source.view(n, ch, h_*w)
    target = target.view(n, ch, h_*w)
    # permute shape to [n, h*w, c]
    # source = source.permute(0, 2, 1)
    # target = target.permute(0, 2, 1)

    diffx = torch.sum((source.transpose(1,2).bmm(source))**2, axis=(1,2)) + 2*c*torch.sum((source.transpose(1,2).bmm(source)), axis=(1,2))
    diffy = torch.sum((target.transpose(1,2).bmm(target))**2, axis=(1,2)) + 2*c*torch.sum(target.transpose(1,2).bmm(target), axis=(1,2)) 
    diffxy = torch.sum((source.bmm(target.transpose(1,2)))**2, axis=(1,2)) + 2*c*torch.sum(source.bmm(target.transpose(1,2)), axis=(1,2))
    diff = diffx + diffy - 2*diffxy
    return diff/(4.0*h_*h_*w*w*ch*ch)

def MMD_linearTime_poly(source, target, c):
    d = 2
    n, ch, h_, w = source.size()
    source = source.view(n, ch, h_*w)
    target = target.view(n, ch, h_*w)
    '''
    idx = torch.randperm(ch)
    idy = torch.randperm(ch)
    source = source[:, idx, ...]
    target = target[:, idy, ...]
    source = source.permute(0, 2, 1)
    target = target.permute(0, 2, 1)
    '''

    idx = torch.randperm(h_*w)
    idy = torch.randperm(h_*w)
    source = source[..., idx]
    target = target[..., idy]
    #source = source.permute(0, 2, 1)
    #target = target.permute(0, 2, 1)

    x_even, x_odd, n0 = split_even_odd(source)
    y_even, y_odd, n0 = split_even_odd(target)

    diffx = torch.sum((x_even.transpose(1,2).bmm(x_odd))**2, axis=(1,2)) + 2*c*torch.sum((x_even.transpose(1,2).bmm(x_odd)), axis=(1,2))
    diffy = torch.sum((y_even.transpose(1,2).bmm(y_odd))**2, axis=(1,2)) + 2*c*torch.sum(y_even.transpose(1,2).bmm(y_odd), axis=(1,2)) 
    diffxy = torch.sum((x_even.bmm(y_odd.transpose(1,2)))**2, axis=(1,2)) + 2*c*torch.sum(x_even.bmm(y_odd.transpose(1,2)), axis=(1,2))
    diffyx = torch.sum((x_odd.bmm(y_even.transpose(1,2)))**2, axis=(1,2)) + 2*c*torch.sum(x_odd.bmm(y_even.transpose(1,2)), axis=(1,2))
    diff = diffx + diffy - diffxy - diffyx

    return diff/(4.0*h_*h_*w*w*ch*ch)

def MMD_linearTime_linear(source, target):
    n, ch, h_, w = source.size()
    source = source.view(n, ch, h_*w)
    target = target.view(n, ch, h_*w)
    idx = torch.randperm(ch)
    idy = torch.randperm(ch)
    source = source[:, idx, ...]
    target = target[:, idy, ...]
    source = source.permute(0, 2, 1)
    target = target.permute(0, 2, 1)

    x_even, x_odd, n0 = split_even_odd(source)
    y_even, y_odd, n0 = split_even_odd(target)

    diffx = torch.sum((x_even.transpose(1,2).bmm(x_odd)), axis=(1,2))
    diffy = torch.sum((y_even.transpose(1,2).bmm(y_odd)), axis=(1,2))
    diffxy = torch.sum((x_even.bmm(y_odd.transpose(1,2))), axis=(1,2))
    diffyx = torch.sum((x_odd.bmm(y_even.transpose(1,2))), axis=(1,2))
    diff = diffx + diffy - diffxy - diffyx

    return diff/(h_*h_*w*w*ch*ch)

def MMD_linear(source, target):
    batch, ch, h_, w = source.size()
    source = source.view(batch, ch, h_*w)
    target = target.view(batch, ch, h_*w)
    # permute shape to [n, h*w, c]
    x = source.permute(0, 2, 1)
    y = target.permute(0, 2, 1)

    diffx = torch.sum(x.bmm(x.transpose(1,2)), axis=(1,2))
    diffy = torch.sum(y.bmm(y.transpose(1,2)), axis=(1,2))
    diffxy = torch.sum(x.bmm(y.transpose(1,2)), axis=(1,2))
    diff = diffx + diffy - 2 * diffxy
    return diff/(h_*h_*w*w*ch*ch)

def Style_bn(source, target):
    batch, ch, h_, w = source.size()
    source = source.view(batch, ch, h_*w)
    target = target.view(batch, ch, h_*w)
    # permute shape to [n, h*w, c]
    x = source.permute(0, 2, 1)
    y = target.permute(0, 2, 1)

    n = 2*h_*w
    p = ch
    ux = torch.sum(x, axis=1)*2.0/n
    uy = torch.sum(y, axis=1)*2.0/n
    diffu = torch.sum((ux-uy)**2, axis=1)

    vx = torch.sqrt(torch.sum((x-torch.unsqueeze(ux, 1))**2, axis=1)*2.0/n)
    vy = torch.sqrt(torch.sum((y-torch.unsqueeze(uy, 1))**2, axis=1)*2.0/n)
    diffv = torch.sum((vx-vy)**2, axis=1)

    diff = ((diffu+diffv)/(p*p))*h_*w
    return diff

class StyleLoss(nn.Module):
    """
    Style Loss, mmd and bn_loss
    """
    def __init__(self, kernel_type, bn_loss, c):
        super(StyleLoss, self).__init__()
        self.loss = 0
        self.kernel_type = kernel_type
        self.c = c
        self.bn_loss = bn_loss

    def forward(self, source_feature, target_feature):
        if self.bn_loss==1:
            self.loss = Style_bn(source_feature, target_feature)
        else:
            if self.kernel_type == 'gussi':
                MMD_gussian = MMD_gussi()
                self.loss = MMD_gussian(source_feature, target_feature)
            elif self.kernel_type == 'linear':
                self.loss = MMD_linear(source_feature, target_feature)
            elif self.kernel_type == 'poly':
                self.loss = MMD_poly(source_feature, target_feature, self.c)
            elif self.kernel_type == 'l_poly':
                self.loss = MMD_linearTime_poly(source_feature, target_feature, self.c)
            elif self.kernel_type == 'l_gussi':
                self.loss = MMD_linearTime_gussi(source_feature, target_feature)
            elif self.kernel_type == 'l_linear':
                self.loss = MMD_linearTime_linear(source_feature, target_feature)
        return self.loss

def mmd_furthest(s, t, kernel_type, bn_loss, c, batch_size):
    mmd_loss = StyleLoss(kernel_type, bn_loss, c)
    index = []
    distance = mmd_loss(s, t)
    for i in range(batch_size):
        index.append(torch.argmax(distance[i * 20: (i + 1) * 20], dim=0) + i * 20)
    if len(t.shape) == 2:
        t = t[:, None, None, :]
    return t[index, :, :, :]

def rn_select(y, num, batch_size):
    target = []
    y = y.numpy().tolist()
    for i in range(batch_size):
        target.append(choice([j for j in range(0, num) if j != y[i]]))
    return np.array(target)

def gram_metrix(input):
    a,b,c,d = input.size()
    feature = input.view(a, b, c*d)
    gram = feature.bmm(feature.transpose(1,2))
    return gram/(b*c*d)

def gram_furthest(s, t, batch_size):
    # find the furthest feature of gram metrics for each input feature respectively
    index = []
    distance = gram_distance(s, t)
    for i in range(batch_size):
        index.append(torch.argmax(distance[i * 20: (i + 1) * 20], dim=0) + i * 20)
    if len(t.shape) == 2:
        t = t[:, None, None, :]
    return t[index, :, :, :]

def furthest(s, t, batch_size):
    # find the furthest feature for each input feature respectively
    index = []
    distance = l2_norm(t - s)
    for i in range(batch_size):
        index.append(torch.argmax(distance[i * 20: (i + 1) * 20], dim=0) + i * 20)
    if len(t.shape) == 2:
        t = t[:, None, None, :]
    return t[index, :, :, :]

def gram_distance(s, t):
    s_gram = gram_metrix(s)
    t_gram = gram_metrix(t)
    return ((s_gram - t_gram) ** 2).sum((1, 2))/4.0

def attack_fp(x, t_f, model, kernel_type, bn_loss, c, batch_size):
    # optim = torch.optim.SGD([x], lr=1e-4)
    alpha = args.eps / args.iteration
    momentum = torch.zeros([batch_size, 3, args.size, args.size]).cuda()

    mmd_loss = StyleLoss(kernel_type, bn_loss, c)

    # start attack
    for i in range(args.iteration):
        
        ori_out = model(x)
        s_f = mid_inputs
        # s_f = mid_outputs  # use 1.features.norm5's output as the last layer

        # l2_loss = l2_norm(t_f - s_f).sum() + 1e6*gram_distance(t_f, s_f).sum()
        if args.mmdMethod==1:
            l2_loss = mmd_loss(t_f, s_f).sum()
        elif args.mmdMethod==2:
            l2_loss = l2_norm(t_f - s_f).sum()
        elif args.mmdMethod==3:
            l2_loss = gram_distance(t_f, s_f).sum()
        # optim.zero_grad()
        l2_loss.backward()
        # optim.step()
        noise = x.grad.data

        l1_noise = torch.sum(torch.abs(noise), dim=(1, 2, 3))
        l1_noise = l1_noise[:, None, None, None]
        noise = noise / l1_noise
        momentum = momentum * args.decay + noise
        x = x - alpha * torch.sign(momentum)

        # no sign
        '''
        sum = torch.sum(momentum**2, axis=(1,2,3), keepdim=True)
        sum[sum<1e-20] = 1e-6
        sign_sum = torch.sum(torch.sign(momentum)**2, axis=(1,2,3), keepdim=True)
        scale = torch.sqrt(sign_sum/sum)
        x = x - alpha * scale * (momentum)
        '''

        assert not torch.any(torch.isnan(x))
        x = torch.clamp(x, 0, 1).detach()
        x.requires_grad = True

    return x

def attack_mi(x, t_y, model, images_min, images_max):
    '''
    alpha = args.eps / args.iteration
    momentum = torch.zeros([x.shape[0], 3, args.size, args.size]).cuda()
    # start attack
    for i in range(args.iteration):
        pred_logit = model(x)
        ce_loss = F.cross_entropy(pred_logit.cuda(), t_y.cuda(), reduction='sum').cuda()
        ce_loss.backward()
        noise = x.grad.data
        l1_noise = torch.sum(torch.abs(noise), dim=(1, 2, 3))
        l1_noise = l1_noise[:, None, None, None]
        noise = noise / l1_noise
        momentum = momentum * args.decay + noise
        x = x - alpha * torch.sign(momentum)
        x = torch.clamp(x, 0, 1).detach()
        x.requires_grad = True
    return x
    '''
    accumulatedGrad = torch.zeros_like(x)
    alpha = args.eps / args.iteration
    for _ in range(args.iteration):
        pred_logit = model(x)
        ce_loss = F.cross_entropy(pred_logit.cuda(), t_y.cuda(), reduction='sum').cuda()
        ce_loss.backward()
        noise = x.grad.data
        noise = noise / torch.abs(noise).mean((1, 2, 3), keepdim=True)
        accumulatedGrad = args.decay * accumulatedGrad + noise
        x = x - alpha * torch.sign(accumulatedGrad)
        x = clip_by_tensor(x, images_min, images_max) 
        x = torch.autograd.Variable(x, requires_grad = True)
    return x.detach()


def attack_Mdi(x, t_y, model, images_min, images_max):
    alpha = args.eps / args.iteration
    momentum = torch.zeros([x.shape[0], 3, args.size, args.size]).cuda()

    # start attack
    accumulatedGrad = torch.zeros_like(x)
    for i in range(args.iteration):
        pred_logit = model(input_diversity(x))
        ce_loss = F.cross_entropy(pred_logit.cuda(), t_y.cuda(), reduction='sum').cuda()
        ce_loss.backward()
        noise = x.grad.data
        noise = noise / torch.abs(noise).mean((1, 2, 3), keepdim=True)
        accumulatedGrad = args.decay * accumulatedGrad + noise
        x = x - alpha * torch.sign(accumulatedGrad)
        x = clip_by_tensor(x, images_min, images_max) 
        x = torch.autograd.Variable(x, requires_grad = True)
    return x.detach()

def gkern(kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        import scipy.stats as st
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel.astype(np.float32)
        # stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

def attack_Ti(x, t_y, model, images_min, images_max):
    alpha = args.eps / args.iteration
    T_kern = torch.from_numpy(gkern(15, 3)).cuda()
    for i in range(args.iteration):
        pred_logit = model(x)
        ce_loss = F.cross_entropy(pred_logit.cuda(), t_y.cuda(), reduction='sum').cuda()
        ce_loss.backward()
        noise = x.grad.data
        noise = F.conv2d(noise, T_kern, padding = (7, 7), groups=3)
        x = x - alpha * torch.sign(noise)
        x = clip_by_tensor(x, images_min, images_max) 
        x = torch.autograd.Variable(x, requires_grad = True)
    return x.detach()

def input_diversity(x):
    rnd = torch.randint(args.size, args.resize, ())
    rescaled = nn.functional.interpolate(x, [rnd, rnd])
    h_rem = args.resize - rnd
    w_rem = args.resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    padded = nn.functional.interpolate(padded, [args.resize, args.resize])
    return padded if torch.rand(()) < args.prob else x

def squared_l2_norm(x):
    flattened = x.reshape([x.shape[0], -1]).contiguous()
    flattened = flattened ** 2
    return torch.sum(flattened, dim=1)

def l2_norm(x):
    return squared_l2_norm(x) ** 0.5

@torch.no_grad()
def test(x_adv, y, target_y, DenseNet_121, Vgg_19, Inc_v3, ResNet_50, list_121, list_y, num, utr, tsuc, ttr):

    x_adv = x_adv.cuda()
    y = y.cuda()
    pred_adv_vgg = torch.argmax(Vgg_19(x_adv), dim=1)
    pred_adv_incv3 = torch.argmax(Inc_v3(x_adv), dim=1)
    pred_adv_121 = torch.argmax(DenseNet_121(x_adv), dim=1)
    pred_adv_50 = torch.argmax(ResNet_50(x_adv), dim=1)

    # White Box Model
    num[0] += torch.sum(pred_adv_121 != y)
    tsuc[0] += torch.sum(pred_adv_121 == target_y)
    idx_121 = pred_adv_121 != y
    idx_121_t = pred_adv_121 == target_y

    # Save White Box Model Tsuc List
    for img in x_adv[idx_121_t]:
        list_121.append(img.detach().cpu().numpy())
    for t_y in target_y[idx_121_t]:
        list_y.append(t_y.detach().cpu().numpy())
    
    # Black Box Model
    num[1] += torch.sum(pred_adv_vgg  != y)
    tsuc[1] += torch.sum(pred_adv_vgg  == target_y)
    idx_vgg = pred_adv_vgg != y
    idx_vgg_t = pred_adv_vgg == target_y

    num[2] += torch.sum(pred_adv_incv3 != y)
    tsuc[2] += torch.sum(pred_adv_incv3 == target_y)
    idx_incv3 = pred_adv_incv3 != y
    idx_incv3_t = pred_adv_incv3 == target_y

    num[3] += torch.sum(pred_adv_50 != y)
    tsuc[3] += torch.sum(pred_adv_50 == target_y)
    idx_50 = pred_adv_50 != y
    idx_50_t = pred_adv_50 == target_y

    utr[0] += torch.sum(idx_121 & idx_vgg)
    utr[1] += torch.sum(idx_121 & idx_incv3)
    utr[2] += torch.sum(idx_121 & idx_50)
    ttr[0] += torch.sum(idx_121_t & idx_vgg_t)
    ttr[1] += torch.sum(idx_121_t & idx_incv3_t)
    ttr[2] += torch.sum(idx_121_t & idx_50_t)

    return list_121, list_y, num, utr, tsuc, ttr

def transfer_test(DenseNet_121, ResNet_50, list_121, list_y, front):
    list_y = np.array(list_y)
    list_121 = np.array(list_121)
    y = torch.from_numpy(list_y).cuda()
    list_121 = torch.from_numpy(list_121).cuda()
    pred_121 = []
    total_num = 0

    with torch.no_grad():
        for i in range(len(list_121)):
            pred_121.append(DenseNet_121(list_121[i].unsqueeze(0)))
        pred_121 = torch.cat(pred_121, dim=0).cuda()
        value_121, idx_121 = torch.max(pred_121, 1)
        _, sorted_idx = value_121.sort(0, True)
        for i in range(min(front // 100, math.ceil(len(sorted_idx) / 100))):
            front_50 = torch.argmax(ResNet_50(list_121[sorted_idx[i*100:i*100+100]]), dim=1)
            uTR_50 = torch.sum(front_50 == y[sorted_idx[i*100:i*100+100]])
            total_num = total_num + uTR_50.item()
        print('Transfer num of black-box model(the front of %d):' %(front), total_num)

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    # t = t.float()
    # t_min = t_min.cuda()
    # t_max = t_max.cuda()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def save_img(save_path, img):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = (img * 255).permute(0, 2, 3, 1).detach().cpu()
    # print(img.shape[0])
    for i in range(img.shape[0]):
        img_name = os.path.join(save_path, str(i) + '.png')
        Image.fromarray(np.array(img[i].squeeze(0)).astype('uint8')).save(img_name)

def main():

    global args

    # print('loading feature library...')
    library = np.load('/home/chengyaya/data/code_attack/data/ImageNet_image_library.npy')
    library = torch.from_numpy(library)
    # print('loading feature library done...')

    # Data loading code
    args = parser.parse_args()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]
                                     )
    
    print('dataset: ', args.data)
    # densenet_121_f = _model_('densenet_f', layer_block_dict[7]).cuda()
    # densenet_121 = _model_('densenet121').cuda()
    # densenet_121 = torch.nn.DataParallel(torch.nn.Sequential(Normalize(args.mean, args.std), models.densenet121(pretrained=True).eval().cuda()))
    densenet_121 = torch.nn.Sequential(Normalize(args.mean, args.std), models.densenet121(pretrained=True).eval().cuda())
    resnet_50 = torch.nn.Sequential(Normalize(args.mean, args.std), models.resnet50(pretrained=True).eval().cuda())
    vgg_19 = torch.nn.Sequential(Normalize(args.mean, args.std), models.vgg19_bn(pretrained=True).eval().cuda())
    inc_v3 = torch.nn.Sequential(Normalize(args.mean, args.std), models.inception_v3(pretrained=True).eval().cuda())

    global mid_outputs
    global mid_inputs
    mid_outputs = None
    mid_inputs = None
    layer_list = []
    vgg19_layer_list = []
    # root = "/home/chengyaya/data/code_attack/Non-Targeted-Adversarial-Attacks/way1_advImg_forview"
    root = "Non-Targeted-Adversarial-Attacks/test_stylemmd"

    def get_mid_output(m, i, o):
        global mid_outputs 
        mid_outputs = o

    def get_mid_input(m, i, o):
        global mid_inputs
        if isinstance(i[0], list):
            mid_inputs = torch.cat(i[0], 1)
        else:
            mid_inputs = i[0]

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()
    
    for (name, module) in densenet_121.named_modules():
        if isinstance(module, models.densenet._DenseLayer):
            layer_list.append(name)
        if isinstance(module, models.densenet._Transition):
            layer_list.append(name)
        if isinstance(module, nn.Linear):
            layer_list.append(name)
        if name == '1.features.norm5':
            layer_list.append(name)
    
    # fetch intput of last layer as current layer's output, way1
    # layer_list_1 = ["1.features.denseblock1.denselayer5", "1.features.transition1", "1.features.denseblock2.denselayer3", "1.features.denseblock2.denselayer5","1.features.denseblock2.denselayer7","1.features.denseblock2.denselayer9", "1.features.denseblock2.denselayer11", "1.features.transition2", "1.features.denseblock3.denselayer3","1.features.denseblock3.denselayer5","1.features.denseblock3.denselayer15","1.features.denseblock3.denselayer17","1.features.denseblock3.denselayer19","1.features.denseblock3.denselayer21","1.features.denseblock3.denselayer23","1.features.transition3","1.features.denseblock4.denselayer3","1.features.denseblock4.denselayer9","1.features.denseblock4.denselayer11","1.features.denseblock4.denselayer13","1.features.denseblock4.denselayer15","1.classifier"]

    # fetch output of current layer as output, way2
    # layer_list_2 = ["1.features.denseblock1.denselayer4", "1.features.denseblock1.denselayer6", "1.features.denseblock2.denselayer2", "1.features.denseblock2.denselayer4","1.features.denseblock2.denselayer6","1.features.denseblock2.denselayer8", "1.features.denseblock2.denselayer10","1.features.denseblock2.denselayer12", "1.features.denseblock3.denselayer2","1.features.denseblock3.denselayer4","1.features.denseblock3.denselayer14","1.features.denseblock3.denselayer16","1.features.denseblock3.denselayer18","1.features.denseblock3.denselayer20","1.features.denseblock3.denselayer22","1.features.denseblock3.denselayer24","1.features.denseblock4.denselayer2","1.features.denseblock4.denselayer8","1.features.denseblock4.denselayer10","1.features.denseblock4.denselayer12","1.features.denseblock4.denselayer14","1.features.denseblock4.denselayer16","1.features.denseblock1","1.features.denseblock2", "1.features.denseblock3", "1.features.denseblock4"]

    # for gussi kernel
    layer_list = ['1.features.denseblock2.denselayer3','1.features.denseblock2.denselayer5','1.features.denseblock2.denselayer7', '1.features.denseblock2.denselayer11','1.features.transition2','1.features.denseblock3.denselayer2','1.features.denseblock3.denselayer4','1.features.denseblock3.denselayer6','1.features.denseblock3.denselayer9','1.features.denseblock3.denselayer11','1.features.denseblock3.denselayer15','1.features.denseblock3.denselayer17','1.features.denseblock3.denselayer20','1.features.denseblock3.denselayer23','1.features.transition3','1.features.denseblock4.denselayer3','1.features.denseblock4.denselayer6','1.features.denseblock4.denselayer13','1.features.denseblock4.denselayer14','1.features.denseblock4.denselayer15', '1.features.norm5']
    batch_size = [2,1,1,1,1,15,12,10,10,8,7,7,7,7,6,80,70,60,55,60, 60]

    # dense121 chosen layer
    layer_list = ['1.features.denseblock1.denselayer1', '1.features.denseblock1.denselayer3','1.features.denseblock1.denselayer5','1.features.transition1','1.features.denseblock2.denselayer3', '1.features.denseblock2.denselayer5', '1.features.denseblock2.denselayer7', '1.features.denseblock2.denselayer11', '1.features.transition2', '1.features.denseblock3.denselayer2', '1.features.denseblock3.denselayer4', '1.features.denseblock3.denselayer6', '1.features.denseblock3.denselayer9', '1.features.denseblock3.denselayer11', '1.features.denseblock3.denselayer15', '1.features.denseblock3.denselayer17', '1.features.denseblock3.denselayer20', '1.features.denseblock3.denselayer23', '1.features.transition3', '1.features.denseblock4.denselayer3', '1.features.denseblock4.denselayer6', '1.features.denseblock4.denselayer13', '1.features.denseblock4.denselayer14', '1.features.denseblock4.denselayer15', '1.features.norm5']
    batch_size = [9,7,7,7,60,60,60,60,60,70,70,70,70,70,60,60,60,60,70,80,80,80,60,60,60]

    # for gussi kernel, random target class
    layer_list = ['1.features.denseblock4.denselayer15']
    batch_size = [60]

    # den121, targetclass, optimal layer
    layer_list = ['1.features.denseblock4.denselayer13']
    batch_size = [100]

    # optimal layer in poly kernel
    # layer_list = ['1.features.denseblock4.denselayer13']
    # batch_size = [60]

    c = args.c
    bn_loss = args.bn_loss
    kernel_type = args.kernel_type

    print('os.environ["CUDA_VISIBLE_DEVICES"]: ', os.environ["CUDA_VISIBLE_DEVICES"])
    print('kernel type: ', kernel_type)
    print('bn_loss: ', bn_loss)
    print('kernel_for_furthe: ', args.kernel_for_furthe)
    print('args.secondTarClass: ', args.secondTarClass)
    print('mmdMethod: ', args.mmdMethod)
    print('iteration: ', args.iteration)
    print('targetcls: ', args.targetcls)
    print('c: ', c)
    root = 'Non-Targeted-Adversarial-Attacks/den121MIFGSM'
    
    # for FGSM-based
    '''
    val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data, transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=args.batchsize, shuffle=True,
    num_workers=args.workers, pin_memory=True)

    for targetcls in(-1,2):
        print('targetcls: ', targetcls)
        num = [0]*4
        tsuc = [0]*4
        utr = [0]*3
        ttr = [0]*3
        list_50 = []
        list_y = []
        for i, (x, y) in tqdm(enumerate(val_loader)):
            if i != args.num // args.batchsize:
                x = Variable(x.cuda(), requires_grad=True) 
                images_min = clip_by_tensor(x - 0.07, 0.0, 1.0)
                images_max = clip_by_tensor(x + 0.07, 0.0, 1.0)  
                with torch.no_grad():
                    source_out = resnet_50(x)
                    # select target image feature set, use the second possible class as target class, or randomly choose one
                    if targetcls == -1:
                        target_y = rn_select(y, 1000, args.batchsize)
                        target_y = torch.from_numpy(target_y).cuda()
                    else:
                        softMax = torch.softmax(source_out, -1)  #[1,1000]
                        _, indices = torch.topk(softMax, k=targetcls, dim=-1, sorted=True)
                        theSecond = indices[:, -1]
                        target_y = theSecond
                
                # x_adv = attack_mi(x, target_y, inc_v3, images_min, images_max).detach_() 
                x_adv = attack_Ti(x, target_y, resnet_50, images_min, images_max).detach_() 
                list_50, list_target_y, num, utr, tsuc, ttr = test(x_adv, y, target_y, resnet_50, vgg_19, inc_v3, densenet_121, list_50, list_y, num, utr, tsuc, ttr)
            else:
                break
        
        D_adv = float(args.num)
        print('Error for res50: %10.4f' % float(100 * (float(num[0]) / D_adv)))
        print('tSuc for res50: %10.4f' % float(100 * (float(tsuc[0]) / D_adv)))

        print('Error for vgg19: %10.4f' % float(100 * (float(num[1]) / D_adv)))
        print('uTR for vgg19: %10.4f' % float(100 * (float(utr[0]) / float(num[0]))))
        print('tSuc for vgg19: %10.4f' % float(100 * (float(tsuc[1]) / D_adv)))
        print('tTR for vgg19: %10.4f' % float(100 * (float(ttr[0]) / float(tsuc[0]))))

        print('Error for inc_v3: %10.4f' % float(100 * (float(num[2]) / D_adv)))
        print('uTR for inc_v3: %10.4f' % float(100 * (float(utr[1]) / float(num[0]))))
        print('tSuc for inc_v3: %10.4f' % float(100 * (float(tsuc[2]) / D_adv)))
        print('tTR for inc_v3: %10.4f' % float(100 * (float(ttr[1]) / float(tsuc[0]))))

        print('Error for dense121: %10.4f' % float(100 * (float(num[3]) / D_adv)))
        print('uTR for dense121: %10.4f' % float(100 * (float(utr[2]) / float(num[0]))))
        print('tSuc for dense121: %10.4f' % float(100 * (float(tsuc[3]) / D_adv)))
        print('tTR for dense121: %10.4f' % float(100 * (float(ttr[2]) / float(tsuc[0]))))
        break
    '''
    for targeClas in (2,10,100,500,1000):
        print('targeClas: ', targeClas)
        ba_s=0
        for layer_name in layer_list:
            handlers = []
            for (name, module) in densenet_121.named_modules():
                if name == layer_name:
                    print('\n')
                    print('\n')
                    print(name)
                    print('batch_size: ', batch_size[ba_s])
                    val_loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(args.data, transforms.Compose([
                        transforms.ToTensor(),
                    ])),
                    batch_size=batch_size[ba_s], shuffle=True,
                    num_workers=args.workers, pin_memory=True)
                    handlers.append(module.register_forward_hook(get_mid_input))
                    # handlers.append(module.register_forward_hook(get_mid_output))

            num = [0]*4
            tsuc = [0]*4
            utr = [0]*3
            ttr = [0]*3
            list_121 = []
            list_y = []

            for i, (x, y) in tqdm(enumerate(val_loader)):

                if i != args.num // batch_size[ba_s]:
                    x = Variable(x.cuda(), requires_grad=True)
                    lower = (x - 0.07).clamp(0.0, 1.0)
                    upper = (x + 0.07).clamp(0.0, 1.0)

                    target_feature = []
                    source_feature_tiled = []
                    with torch.no_grad():
                        # select target image feature set, use the second possible class as target class, or randomly choose one

                        # gain original image feature set
                        source_out = densenet_121(x)
                        source_feature = mid_inputs
                        
                        if args.secondTarClass==1:
                            softMax = torch.softmax(source_out, -1)  #[1,1000]
                            _, indices = torch.topk(softMax, k=targeClas, dim=-1, sorted=True)
                            theSecond = indices[:, -1]
                            target_y = theSecond
                        else:
                            target_y = rn_select(y, 1000, batch_size[ba_s])
                            target_y = torch.from_numpy(target_y).cuda()
                        
                        for j in target_y:
                            target_out = densenet_121(library[j].cuda())  
                            target_feature.append(mid_inputs)
                        target_feature = torch.cat(target_feature, axis=0)

                        # get the mean distribution of target feature from 20 target images
                        # target_feature = torch.stack(target_feature, axis=0)
                        # target_feature = torch.mean(target_feature, dim=1)
                        
                        for j in range(batch_size[ba_s]):
                            source_feature_tiled.append(torch.unsqueeze(source_feature[j], 0).repeat(20, 1, 1, 1))
                        source_feature_tiled = torch.cat(source_feature_tiled, dim=0).cuda()

                        if args.mmdMethod==1:
                            furthest_feature = mmd_furthest(source_feature_tiled, target_feature, args.kernel_for_furthe, bn_loss, c, batch_size[ba_s])
                        elif args.mmdMethod==2:
                            furthest_feature = furthest(source_feature_tiled, target_feature, batch_size[ba_s])
                        elif args.mmdMethod==3:
                            furthest_feature = gram_furthest(source_feature_tiled, target_feature, batch_size[ba_s])
                    
                    x_adv = attack_fp(x, furthest_feature.detach(), densenet_121, kernel_type, bn_loss, c, batch_size[ba_s]).detach_()
                    # save adv img
                    #save_path = os.path.join(root, str(i))
                    #save_img(save_path, x_adv)
                    list_121, list_y, num, utr, tsuc, ttr = test(x_adv, y, target_y, densenet_121, vgg_19, inc_v3, resnet_50, list_121, list_y, num, utr, tsuc, ttr)
                else:
                    break

            D_adv = float(args.num)
            print('Error for dense121: %10.4f' % float(100 * (float(num[0]) / D_adv)))
            print('tSuc for dense121: %10.4f' % float(100 * (float(tsuc[0]) / D_adv)))

            print('Error for vgg19: %10.4f' % float(100 * (float(num[1]) / D_adv)))
            print('uTR for vgg19: %10.4f' % float(100 * (float(utr[0]) / float(num[0]))))
            print('tSuc for vgg19: %10.4f' % float(100 * (float(tsuc[1]) / D_adv)))
            if tsuc[0]<1e-20:
                print('tTR for vgg19: %10.4f' % float(100 * (0.0)))
            else:
                print('tTR for vgg19: %10.4f' % float(100 * (float(ttr[0]) / float(tsuc[0]))))

            print('Error for inc_v3: %10.4f' % float(100 * (float(num[2]) / D_adv)))
            print('uTR for inc_v3: %10.4f' % float(100 * (float(utr[1]) / float(num[0]))))
            print('tSuc for inc_v3: %10.4f' % float(100 * (float(tsuc[2]) / D_adv)))
            if tsuc[0]<1e-20:
                print('tTR for inc_v3: %10.4f' % float(100 * (0.0)))
            else:
                print('tTR for inc_v3: %10.4f' % float(100 * (float(ttr[1]) / float(tsuc[0]))))

            print('Error for res50: %10.4f' % float(100 * (float(num[3]) / D_adv)))
            print('uTR for res50: %10.4f' % float(100 * (float(utr[2]) / float(num[0]))))
            print('tSuc for res50: %10.4f' % float(100 * (float(tsuc[3]) / D_adv)))
            if tsuc[0]<1e-20:
                print('tTR for res50: %10.4f' % float(100 * (0.0)))
            else:
                print('tTR for res50: %10.4f' % float(100 * (float(ttr[2]) / float(tsuc[0]))))

            del list_121, list_y, num, utr, tsuc, ttr, x_adv
            ba_s+=1

            for h in handlers:
                h.remove()
    
if __name__ == "__main__":
    main()
