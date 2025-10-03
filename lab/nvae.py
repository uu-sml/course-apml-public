# ---------------------------------------------------------------
# This file has been modified from multiple files in the following
# repo:
# https://github.com/NVlabs/NVAE
#
# The license for the original version of these files can be found
# in this directory (LICENSE_nvidia).
#
# Original copyright statement:
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA

import requests
import gdown

from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli

import collections
import math
import re

CHANNEL_MULT = 2

BN_EPS = 1e-5

OPS = collections.OrderedDict([
    ('res_elu', lambda Cin, Cout, stride: ELUConv(Cin, Cout, 3, stride, 1)),
    ('res_bnelu', lambda Cin, Cout, stride: BNELUConv(Cin, Cout, 3, stride, 1)),
    ('res_bnswish', lambda Cin, Cout, stride: BNSwishConv(Cin, Cout, 3, stride, 1)),
    ('res_bnswish5', lambda Cin, Cout, stride: BNSwishConv(Cin, Cout, 3, stride, 2, 2)),
    ('mconv_e6k5g0', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=6, dil=1, k=5, g=0)),
    ('mconv_e3k5g0', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=3, dil=1, k=5, g=0)),
    ('mconv_e3k5g8', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=3, dil=1, k=5, g=8)),
    ('mconv_e6k11g0', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=6, dil=1, k=11, g=0)),
])

AROPS = collections.OrderedDict([
    ('conv_3x3', lambda C, masked, zero_diag: ELUConv(C, C, 3, 1, 1, masked=masked, zero_diag=zero_diag))
])

# checkpoints for pretrained models on Google Drive
PRETRAINED_MODELS = {
  "mnist": "1ZsjgAkDRUtOQMbSzt9nq6QgIRwpZyIM3",
  "celeba_64": "1IKAzjnq-2mN36xl1XEIaTrqxEHo7PM4b",
  "celeba_256a": "1XR8iqNCJiDFS410NJErTdovW2jxiYGpx",
  "celeba_256b": "19_dTKMIwlyYjSD0I5mV9AFMIs6yeg3f8",
  "cifar10a": "1SQkXTZ5uzaHqXXT-JcO6YzFWmiT8xiNB",
  "cifar10b": "1prsUSMuK0ttwZqhsRujZaYgrJ82mbhJZ",
  "ffhq": "1lQzywY5O71Z5NqAUJUcWy2Q1K2hPFO6j",
}

## Helper functions (added by us)

def get_confirm_token(response):
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      return value
  return None

def save_response_content(response, destination, chunk_size=32768):
  with open(destination, "wb") as f:
    for chunk in response.iter_content(chunk_size):
      if chunk: # filter out keep-alive new chunks
        f.write(chunk)

def download_gdrive(id, destination):
  url = "https://drive.google.com/u/0/uc?export=download"
  session = requests.Session()
  response = session.get(url, params = { 'id' : id, 'confirm' : 't' }, stream = True)
  token = get_confirm_token(response)
  if token:
    params = { 'id' : id, 'confirm' : token }
    response = session.get(url, params = params, stream = True)
  save_response_content(response, destination)

def get_args(checkpoint):
    """Extract and fix arguments stored in the checkpoint."""
    args = checkpoint["args"]
    if not hasattr(args, 'ada_groups'):
        args.ada_groups = False
    if not hasattr(args, 'min_groups_per_scale'):
        args.min_groups_per_scale = 1
    if not hasattr(args, 'num_mixture_dec'):
        args.num_mixture_dec = 10
    return args

def get_statedict(checkpoint):
    """Fix old and incorrect states."""
    regex = r'^(.*)\.bn\.(weight|bias|running_mean|running_var|num_batches_tracked)'
    return collections.OrderedDict(
        (re.sub(regex, r'\1.\2', k), v) for k, v in filter(
            lambda kv: not kv[0].endswith(".weight_normalized"), checkpoint["state_dict"].items()
        )
    )

def load_pretrained_model(name, use_gdown: bool = False):
    filename = "checkpoint_{}.pt".format(name)
    # load saved checkpoint of model
    try:
        data = torch.load(filename, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print("downloading checkpoint of pretrained model for", name, "...")
        id = PRETRAINED_MODELS[name]
        if use_gdown:
            url = f"https://drive.google.com/uc?id={id}"
            gdown.download(url, filename, quiet=False)
        else:
            download_gdrive(PRETRAINED_MODELS[name], filename)
        data = torch.load(filename, map_location='cpu', weights_only=False)

    # extract arguments and states
    args = get_args(data)
    states = get_statedict(data)

    # rebuild model
    arch_instance = get_arch_cells(args.arch_instance)
    model = AutoEncoder(args, arch_instance)
    model.load_state_dict(states)
    model.eval()

    # disable running statistics of BatchNorm2D - maybe it improves performance?
    # # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/57
    for m in model.modules():
        for child in m.children():
            if type(child) == nn.BatchNorm2d:
                child.track_running_stats = False

    return model

## The following lines are copied from utils.py

def count_parameters(model):
    return sum(v.numel() for name, v in model.named_parameters() if "auxiliary" not in name)

def get_stride_for_cell_type(cell_type):
    if cell_type.startswith('normal') or cell_type.startswith('combiner'):
        stride = 1
    elif cell_type.startswith('down'):
        stride = 2
    elif cell_type.startswith('up'):
        stride = -1
    else:
        raise NotImplementedError(cell_type)

    return stride


def one_hot(indices, depth, dim):
    indices = indices.unsqueeze(dim)
    size = list(indices.size())
    size[dim] = depth
    y_onehot = torch.zeros(size, device=indices.device).scatter_(dim, indices, 1)

    return y_onehot


def get_input_size(dataset):
    if dataset == 'mnist':
        return 32
    elif dataset == 'cifar10':
        return 32
    elif dataset.startswith('celeba') or dataset.startswith('imagenet') or dataset.startswith('lsun'):
        size = int(dataset.split('_')[-1])
        return size
    elif dataset == 'ffhq':
        return 256
    else:
        raise NotImplementedError


def pre_process(x, num_bits):
    if num_bits != 8:
        x = torch.floor(x * 255 / 2 ** (8 - num_bits))
        x /= (2 ** num_bits - 1)
    return x


def get_arch_cells(arch_type):
    if arch_type == 'res_elu':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_elu', 'res_elu']
        arch_cells['down_enc'] = ['res_elu', 'res_elu']
        arch_cells['normal_dec'] = ['res_elu', 'res_elu']
        arch_cells['up_dec'] = ['res_elu', 'res_elu']
        arch_cells['normal_pre'] = ['res_elu', 'res_elu']
        arch_cells['down_pre'] = ['res_elu', 'res_elu']
        arch_cells['normal_post'] = ['res_elu', 'res_elu']
        arch_cells['up_post'] = ['res_elu', 'res_elu']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_bnelu':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnelu', 'res_bnelu']
        arch_cells['down_enc'] = ['res_bnelu', 'res_bnelu']
        arch_cells['normal_dec'] = ['res_bnelu', 'res_bnelu']
        arch_cells['up_dec'] = ['res_bnelu', 'res_bnelu']
        arch_cells['normal_pre'] = ['res_bnelu', 'res_bnelu']
        arch_cells['down_pre'] = ['res_bnelu', 'res_bnelu']
        arch_cells['normal_post'] = ['res_bnelu', 'res_bnelu']
        arch_cells['up_post'] = ['res_bnelu', 'res_bnelu']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_bnswish':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_sep':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k5g0']
        arch_cells['down_enc'] = ['mconv_e6k5g0']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['mconv_e3k5g0']
        arch_cells['down_pre'] = ['mconv_e3k5g0']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_sep11':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k11g0']
        arch_cells['down_enc'] = ['mconv_e6k11g0']
        arch_cells['normal_dec'] = ['mconv_e6k11g0']
        arch_cells['up_dec'] = ['mconv_e6k11g0']
        arch_cells['normal_pre'] = ['mconv_e3k5g0']
        arch_cells['down_pre'] = ['mconv_e3k5g0']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res53_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res35_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res55_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['down_enc'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['down_pre'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_mbconv9':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_dec'] = ['mconv_e6k9g0']
        arch_cells['up_dec'] = ['mconv_e6k9g0']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_post'] = ['mconv_e3k9g0']
        arch_cells['up_post'] = ['mconv_e3k9g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_res':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k5g0']
        arch_cells['down_enc'] = ['mconv_e6k5g0']
        arch_cells['normal_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_pre'] = ['mconv_e3k5g0']
        arch_cells['down_pre'] = ['mconv_e3k5g0']
        arch_cells['normal_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_den':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k5g0']
        arch_cells['down_enc'] = ['mconv_e6k5g0']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['mconv_e3k5g8']
        arch_cells['down_pre'] = ['mconv_e3k5g8']
        arch_cells['normal_post'] = ['mconv_e3k5g8']
        arch_cells['up_post'] = ['mconv_e3k5g8']
        arch_cells['ar_nn'] = ['']
    else:
        raise NotImplementedError

    return arch_cells


def groups_per_scale(num_scales, num_groups_per_scale, is_adaptive, divider=2, minimum_groups=1):
    g = []
    n = num_groups_per_scale
    for s in range(num_scales):
        assert n >= 1
        g.append(n)
        if is_adaptive:
            n = n // divider
            n = max(minimum_groups, n)
    return g

## The following lines are copied from distributions.py

@torch.jit.script
def normal_sigma(logsigma):
    x = 5.0 * torch.tanh(logsigma / 5.0)  # soft differentiable clamp between [-5, 5]
    return torch.exp(x)

class DiscLogistic:
    def __init__(self, param):
        B, C, H, W = param.size()
        self.num_c = C // 2
        self.means = param[:, :self.num_c, :, :]                              # B, 3, H, W
        self.log_scales = torch.clamp(param[:, self.num_c:, :, :], min=-8.0)  # B, 3, H, W

    def log_prob(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0
        # convert samples to be in [-1, 1]
        samples = 2 * samples - 1.0

        B, C, H, W = samples.size()
        assert C == self.num_c

        centered = samples - self.means                                         # B, 3, H, W
        inv_stdv = torch.exp(- self.log_scales)
        plus_in = inv_stdv * (centered + 1. / 255.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1. / 255.)
        cdf_min = torch.sigmoid(min_in)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = - F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - self.log_scales - 2. * F.softplus(mid_in)

        log_prob_mid_safe = torch.where(cdf_delta > 1e-5,
                                        torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                        log_pdf_mid - torch.log(127.5))
        # woow the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.99, log_one_minus_cdf_min,
                                                                            log_prob_mid_safe))   # B, 3, H, W

        return log_probs

    def sample(self):
        u = torch.empty_like(self.means).uniform_(1e-5, 1. - 1e-5) # B, 3, H, W
        x = self.means + torch.exp(self.log_scales) * torch.logit(u) # B, 3, H, W
        x = torch.clamp(x, -1, 1.)
        x = x / 2. + 0.5
        return x


class DiscMixLogistic:
    def __init__(self, param, num_mix=10, num_bits=8):
        B, C, H, W = param.size()
        self.num_mix = num_mix
        self.logit_probs = param[:, :num_mix, :, :]                                   # B, M, H, W
        l = param[:, num_mix:, :, :].view(B, 3, 3 * num_mix, H, W)                    # B, 3, 3 * M, H, W
        self.means = l[:, :, :num_mix, :, :]                                          # B, 3, M, H, W
        self.log_scales = torch.clamp(l[:, :, num_mix:2 * num_mix, :, :], min=-7.0)   # B, 3, M, H, W
        self.coeffs = torch.tanh(l[:, :, 2 * num_mix:3 * num_mix, :, :])              # B, 3, M, H, W
        self.max_val = 2. ** num_bits - 1

    def log_prob(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0
        # convert samples to be in [-1, 1]
        samples = 2 * samples - 1.0

        B, C, H, W = samples.size()
        assert C == 3, 'only RGB images are considered.'

        samples = samples.unsqueeze(4)                                                  # B, 3, H , W
        samples = samples.expand(-1, -1, -1, -1, self.num_mix).permute(0, 1, 4, 2, 3)   # B, 3, M, H, W
        mean1 = self.means[:, 0, :, :, :]                                               # B, M, H, W
        mean2 = self.means[:, 1, :, :, :] + \
                self.coeffs[:, 0, :, :, :] * samples[:, 0, :, :, :]                     # B, M, H, W
        mean3 = self.means[:, 2, :, :, :] + \
                self.coeffs[:, 1, :, :, :] * samples[:, 0, :, :, :] + \
                self.coeffs[:, 2, :, :, :] * samples[:, 1, :, :, :]                     # B, M, H, W

        mean1 = mean1.unsqueeze(1)                          # B, 1, M, H, W
        mean2 = mean2.unsqueeze(1)                          # B, 1, M, H, W
        mean3 = mean3.unsqueeze(1)                          # B, 1, M, H, W
        means = torch.cat([mean1, mean2, mean3], dim=1)     # B, 3, M, H, W
        centered = samples - means                          # B, 3, M, H, W

        inv_stdv = torch.exp(- self.log_scales)
        plus_in = inv_stdv * (centered + 1. / self.max_val)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1. / self.max_val)
        cdf_min = torch.sigmoid(min_in)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = - F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - self.log_scales - 2. * F.softplus(mid_in)

        log_prob_mid_safe = torch.where(cdf_delta > 1e-5,
                                        torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                        log_pdf_mid - torch.log(self.max_val / 2))
        # the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.99, log_one_minus_cdf_min,
                                                                            log_prob_mid_safe))   # B, 3, M, H, W

        log_probs = torch.sum(log_probs, 1) + F.log_softmax(self.logit_probs, dim=1)  # B, M, H, W
        return torch.logsumexp(log_probs, dim=1)                                      # B, H, W

    def sample(self, t=1.):
        gumbel = -torch.log(- torch.log(torch.empty_like(self.logit_probs).uniform_(1e-5, 1. - 1e-5))) # B, M, H, W
        sel = one_hot(torch.argmax(self.logit_probs / t + gumbel, 1), self.num_mix, dim=1) # B, M, H, W
        sel = sel.unsqueeze(1) # B, 1, M, H, W

        # select logistic parameters
        means = torch.sum(self.means * sel, dim=2) # B, 3, H, W
        log_scales = torch.sum(self.log_scales * sel, dim=2) # B, 3, H, W
        coeffs = torch.sum(self.coeffs * sel, dim=2) # B, 3, H, W

        # cells from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = torch.empty_like(means).uniform_(1e-5, 1. - 1e-5) # B, 3, H, W
        x = means + torch.exp(log_scales) / t * torch.logit(u) # B, 3, H, W

        x0 = torch.clamp(x[:, 0, :, :], -1, 1.) # B, H, W
        x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :] * x0, -1, 1) # B, H, W
        x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] * x0 + coeffs[:, 2, :, :] * x1, -1, 1)  # B, H, W

        x0 = x0.unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        x = torch.cat([x0, x1, x2], 1)
        x = x / 2. + 0.5
        return x

## The following lines are copied from neural_operations.py
    
def get_skip_connection(C, stride, affine, channel_mult):
    if stride == 1:
        return nn.Identity()
    elif stride == 2:
        return FactorizedReduce(C, int(channel_mult * C))
    elif stride == -1:
        return nn.Sequential(UpSample(), Conv2D(C, int(C / channel_mult), kernel_size=1))


@torch.jit.script
def normalize_weight_jit(log_weight_norm, weight):
    n = torch.exp(log_weight_norm)
    wn = LA.vector_norm(weight, dim=[1, 2, 3], keepdim=True)
    weight = n * weight / (wn + 1e-5)
    return weight


class Conv2D(nn.Conv2d):
    """Allows for weights as input."""

    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, data_init=False,
                 weight_norm=True):
        """
        Args:
            use_shared (bool): Use weights for this layer or not?
        """
        super(Conv2D, self).__init__(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias)

        self.log_weight_norm = None
        if weight_norm:
            init = LA.vector_norm(self.weight, dim=[1, 2, 3], keepdim=True)
            self.log_weight_norm = nn.Parameter(torch.log(init + 1e-2), requires_grad=True)

        self.data_init = data_init
        self.init_done = False

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W).
            params (ConvParam): containing `weight` and `bias` (optional) of conv operation.
        """
        # do data based initialization
        if self.data_init and not self.init_done:
            with torch.no_grad():
                wn = LA.vector_norm(self.weight, dim=[1, 2, 3], keepdim=True)
                weight = self.weight / (wn + 1e-5)
                bias = None
                out = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
                mn = torch.mean(out, dim=[0, 2, 3])
                st = 5 * torch.std(out, dim=[0, 2, 3])

                if self.bias is not None:
                    self.bias.data = - mn / (st + 1e-5)
                self.log_weight_norm.data = -torch.log((st.view(-1, 1, 1, 1) + 1e-5))
                self.init_done = True

        weight_normalized = self.normalize_weight()

        bias = self.bias
        return F.conv2d(x, weight_normalized, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def normalize_weight(self):
        """ applies weight normalization """
        if self.log_weight_norm is not None:
            weight = normalize_weight_jit(self.log_weight_norm, self.weight)
        else:
            weight = self.weight

        return weight

class ELUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1):
        super(ELUConv, self).__init__()
        self.upsample = stride == -1
        stride = abs(stride)
        self.conv_0 = Conv2D(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=True, dilation=dilation,
                             data_init=True)

    def forward(self, x):
        out = F.elu(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv_0(out)
        return out


class BNELUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1, eps=BN_EPS):
        super(BNELUConv, self).__init__()
        self.upsample = stride == -1
        stride = abs(stride)
        self.bn = nn.BatchNorm2d(C_in, eps=eps, momentum=0.05)
        self.conv_0 = Conv2D(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=True, dilation=dilation)

    def forward(self, x):
        x = self.bn(x)
        out = F.elu(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv_0(out)
        return out


class BNSwishConv(nn.Module):
    """ReLU + Conv2d + BN."""

    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1, eps=BN_EPS):
        super(BNSwishConv, self).__init__()
        self.upsample = stride == -1
        stride = abs(stride)
        self.bn_act = nn.BatchNorm2d(C_in, eps=eps, momentum=0.05)
        self.conv_0 = Conv2D(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=True, dilation=dilation)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W)
        """
        x = self.bn_act(x)
        out = F.silu(x, inplace=True)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv_0(out)
        return out



class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.conv_1 = Conv2D(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_2 = Conv2D(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_3 = Conv2D(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_4 = Conv2D(C_in, C_out - 3 * (C_out // 4), 1, stride=2, padding=0, bias=True)

    def forward(self, x):
        out = F.silu(x, inplace=True)
        conv1 = self.conv_1(out)
        conv2 = self.conv_2(out[:, :, 1:, 1:])
        conv3 = self.conv_3(out[:, :, :, 1:])
        conv4 = self.conv_4(out[:, :, 1:, :])
        out = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        return out


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        pass

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


class EncCombinerCell(nn.Module):
    def __init__(self, Cin1, Cin2, Cout, cell_type):
        super(EncCombinerCell, self).__init__()
        self.cell_type = cell_type
        # Cin = Cin1 + Cin2
        self.conv = Conv2D(Cin2, Cout, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        x2 = self.conv(x2)
        out = x1 + x2
        return out


# original combiner
class DecCombinerCell(nn.Module):
    def __init__(self, Cin1, Cin2, Cout, cell_type):
        super(DecCombinerCell, self).__init__()
        self.cell_type = cell_type
        self.conv = Conv2D(Cin1 + Cin2, Cout, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=1)
        out = self.conv(out)
        return out


class ConvBNSwish(nn.Module):
    def __init__(self, Cin, Cout, k=3, stride=1, groups=1, dilation=1, eps=BN_EPS):
        padding = dilation * (k - 1) // 2
        super(ConvBNSwish, self).__init__()

        self.conv = nn.Sequential(
            Conv2D(Cin, Cout, k, stride, padding, groups=groups, bias=False, dilation=dilation, weight_norm=False),
            nn.BatchNorm2d(Cout, eps=eps, momentum=0.05),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class SE(nn.Module):
    def __init__(self, Cin, Cout):
        super(SE, self).__init__()
        num_hidden = max(Cout // 16, 4)
        self.se = nn.Sequential(nn.Linear(Cin, num_hidden), nn.ReLU(inplace=True),
                                nn.Linear(num_hidden, Cout), nn.Sigmoid())

    def forward(self, x):
        se = torch.mean(x, dim=[2, 3])
        se = se.view(se.size(0), -1)
        se = self.se(se)
        se = se.view(se.size(0), -1, 1, 1)
        return x * se


class InvertedResidual(nn.Module):
    def __init__(self, Cin, Cout, stride, ex, dil, k, g, eps=BN_EPS):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2, -1]

        hidden_dim = int(round(Cin * ex))
        self.use_res_connect = self.stride == 1 and Cin == Cout
        self.upsample = self.stride == -1
        self.stride = abs(self.stride)
        groups = hidden_dim if g == 0 else g

        layers0 = [nn.UpsamplingNearest2d(scale_factor=2)] if self.upsample else []
        layers = [nn.BatchNorm2d(Cin, eps=eps, momentum=0.05),
                  ConvBNSwish(Cin, hidden_dim, k=1),
                  ConvBNSwish(hidden_dim, hidden_dim, stride=self.stride, groups=groups, k=k, dilation=dil),
                  Conv2D(hidden_dim, Cout, 1, 1, 0, bias=False, weight_norm=False),
                  nn.BatchNorm2d(Cout, eps=eps, momentum=0.05)]

        layers0.extend(layers)
        self.conv = nn.Sequential(*layers0)

    def forward(self, x):
        return self.conv(x)

## The following lines are copied from neural_ar_operations.py

def channel_mask(c_in, g_in, c_out, zero_diag):
    assert c_in % c_out == 0 or c_out % c_in == 0, "%d - %d" % (c_in, c_out)
    assert g_in == 1 or g_in == c_in

    if g_in == 1:
        mask = torch.ones(c_out, c_in)
        if c_out >= c_in:
            ratio = c_out // c_in
            for i in range(c_in):
                mask[i * ratio:(i + 1) * ratio, i + 1:] = 0
                if zero_diag:
                    mask[i * ratio:(i + 1) * ratio, i:i + 1] = 0
        else:
            ratio = c_in // c_out
            for i in range(c_out):
                mask[i:i + 1, (i + 1) * ratio:] = 0
                if zero_diag:
                    mask[i:i + 1, i * ratio:(i + 1) * ratio:] = 0
    elif g_in == c_in:
        mask = torch.ones(c_out, c_in // g_in)
        if zero_diag:
            mask = 0. * mask

    return mask


def create_conv_mask(kernel_size, c_in, g_in, c_out, zero_diag, mirror):
    m = (kernel_size - 1) // 2
    mask = torch.ones(c_out, c_in // g_in, kernel_size, kernel_size)
    mask[:, :, m:, :] = 0
    mask[:, :, m, :m] = 1
    mask[:, :, m, m] = channel_mask(c_in, g_in, c_out, zero_diag)
    if mirror:
        mask = mask.flip([2, 3])
    return mask


class ARConv2d(nn.Conv2d):
    """Allows for weights as input."""

    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 masked=False, zero_diag=False, mirror=False):
        """
        Args:
            use_shared (bool): Use weights for this layer or not?
        """
        super(ARConv2d, self).__init__(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias)

        self.masked = masked
        if self.masked:
            assert kernel_size % 2 == 1, 'kernel size should be an odd value.'
            # `register_buffer` ensures that the tensor is moved to the GPU when
            # `cuda()` or `to(device)` is called
            self.register_buffer(
                'mask',
                create_conv_mask(kernel_size, C_in, groups, C_out, zero_diag, mirror),
                persistent=False, # do not save it in the state_dict
            )
        else:
            self.mask = 1.0

        # init weight normalizaition parameters
        init = torch.log(LA.vector_norm(self.weight * self.mask, dim=[1, 2, 3], keepdim=True) + 1e-2)
        self.log_weight_norm = nn.Parameter(init, requires_grad=True)

    def normalize_weight(self):
        weight = self.weight
        if self.masked:
            assert self.mask.size() == weight.size()
            weight = weight * self.mask

        # weight normalization
        weight = normalize_weight_jit(self.log_weight_norm, weight)
        return weight

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W).
            params (ConvParam): containing `weight` and `bias` (optional) of conv operation.
        """
        weight_normalized = self.normalize_weight()
        bias = self.bias
        return F.conv2d(x, weight_normalized, bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ELUConv(nn.Module):
    """ReLU + Conv2d + BN."""

    def __init__(self, C_in, C_out, kernel_size, padding=0, dilation=1, masked=True, zero_diag=True,
                 weight_init_coeff=1.0, mirror=False):
        super(ELUConv, self).__init__()
        self.conv_0 = ARConv2d(C_in, C_out, kernel_size, stride=1, padding=padding, bias=True, dilation=dilation,
                               masked=masked, zero_diag=zero_diag, mirror=mirror)
        # change the initialized log weight norm
        self.conv_0.log_weight_norm.data += math.log(weight_init_coeff)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W)
        """
        out = F.elu(x)
        out = self.conv_0(out)
        return out


class ARInvertedResidual(nn.Module):
    def __init__(self, inz, inf, ex=6, dil=1, k=5, mirror=False):
        super(ARInvertedResidual, self).__init__()
        hidden_dim = int(round(inz * ex))
        padding = dil * (k - 1) // 2
        layers = []
        layers.extend([ARConv2d(inz, hidden_dim, kernel_size=3, padding=1, masked=True, mirror=mirror, zero_diag=True),
                       nn.ELU(inplace=True)])
        layers.extend([ARConv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=k, padding=padding, dilation=dil,
                                masked=True, mirror=mirror, zero_diag=False),
                      nn.ELU(inplace=True)])
        self.convz = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim

    def forward(self, z, ftr):
        z = self.convz(z)
        return z


class MixLogCDFParam(nn.Module):
    def __init__(self, num_z, num_mix, num_ftr, mirror):
        super(MixLogCDFParam, self).__init__()

        num_out = num_z * (3 * num_mix + 3)
        self.conv = ELUConv(num_ftr, num_out, kernel_size=1, padding=0, masked=True, zero_diag=False,
                            weight_init_coeff=0.1, mirror=mirror)
        self.num_z = num_z
        self.num_mix = num_mix

    def forward(self, ftr):
        out = self.conv(ftr)
        b, c, h, w = out.size()
        out = out.view(b, self.num_z, c // self.num_z,  h, w)
        m = self.num_mix
        logit_pi, mu, log_s, log_a, b, _ = torch.split(out, [m, m, m, 1, 1, 1], dim=2)  # the last one is dummy
        return logit_pi, mu, log_s, log_a, b


def mix_log_cdf_flow(z1, logit_pi, mu, log_s, log_a, b):
    # z         b, n, 1, h, w
    # logit_pi  b, n, k, h, w
    # mu        b, n, k, h, w
    # log_s     b, n, k, h, w
    # log_a     b, n, 1, h, w
    # b         b, n, 1, h, w

    log_s = torch.clamp(log_s, min=-7)

    z = z1.unsqueeze(dim=2)
    log_pi = torch.log_softmax(logit_pi, dim=2)  # normalize log_pi
    u = - (z - mu) * torch.exp(-log_s)
    softplus_u = F.softplus(u)
    log_mix_cdf = log_pi - softplus_u
    log_one_minus_mix_cdf = log_mix_cdf + u
    log_mix_cdf = torch.logsumexp(log_mix_cdf, dim=2)
    log_one_minus_mix_cdf = torch.logsumexp(log_one_minus_mix_cdf, dim=2)

    log_a = log_a.squeeze_(dim=2)
    b = b.squeeze_(dim=2)
    new_z = torch.exp(log_a) * (log_mix_cdf - log_one_minus_mix_cdf) + b

    # compute log determinant Jac
    log_mix_pdf = torch.logsumexp(log_pi + u - log_s - 2 * softplus_u, dim=2)
    log_det = log_a - log_mix_cdf - log_one_minus_mix_cdf + log_mix_pdf

    return new_z, log_det


class Cell(nn.Module):
    def __init__(self, Cin, Cout, cell_type, arch, use_se):
        super(Cell, self).__init__()
        self.cell_type = cell_type

        stride = get_stride_for_cell_type(self.cell_type)
        self.skip = get_skip_connection(Cin, stride, affine=False, channel_mult=CHANNEL_MULT)
        self.use_se = use_se
        self._num_nodes = len(arch)
        self._ops = nn.ModuleList()
        for i in range(self._num_nodes):
            stride = get_stride_for_cell_type(self.cell_type) if i == 0 else 1
            C = Cin if i == 0 else Cout
            primitive = arch[i]
            op = OPS[primitive](C, Cout, stride)
            self._ops.append(op)

        # SE
        if self.use_se:
            self.se = SE(Cout, Cout)

    def forward(self, s):
        # skip branch
        skip = self.skip(s)
        for i in range(self._num_nodes):
            s = self._ops[i](s)

        s = self.se(s) if self.use_se else s
        return skip + 0.1 * s


class CellAR(nn.Module):
    def __init__(self, num_z, num_ftr, num_c, arch, mirror):
        super(CellAR, self).__init__()
        assert num_c % num_z == 0

        self.cell_type = 'ar_nn'

        # s0 will the random samples
        ex = 6
        self.conv = ARInvertedResidual(num_z, num_ftr, ex=ex, mirror=mirror)

        self.use_mix_log_cdf = False
        if self.use_mix_log_cdf:
            self.param = MixLogCDFParam(num_z, num_mix=3, num_ftr=self.conv.hidden_dim, mirror=mirror)
        else:
            # 0.1 helps bring mu closer to 0 initially
            self.mu = ELUConv(self.conv.hidden_dim, num_z, kernel_size=1, padding=0, masked=True, zero_diag=False,
                                weight_init_coeff=0.1, mirror=mirror)

    def forward(self, z, ftr):
        s = self.conv(z, ftr)

        if self.use_mix_log_cdf:
            logit_pi, mu, log_s, log_a, b = self.param(s)
            new_z, log_det = mix_log_cdf_flow(z, logit_pi, mu, log_s, log_a, b)
        else:
            mu = self.mu(s)
            new_z = (z - mu)
            log_det = torch.zeros_like(new_z)

        return new_z, log_det


class PairedCellAR(nn.Module):
    def __init__(self, num_z, num_ftr, num_c, arch=None):
        super(PairedCellAR, self).__init__()
        self.cell1 = CellAR(num_z, num_ftr, num_c, arch, mirror=False)
        self.cell2 = CellAR(num_z, num_ftr, num_c, arch, mirror=True)

    def forward(self, z, ftr):
        new_z, log_det1 = self.cell1(z, ftr)
        new_z, log_det2 = self.cell2(new_z, ftr)

        log_det1 += log_det2
        return new_z, log_det1


class AutoEncoder(nn.Module):
    def __init__(self, args, arch_instance):
        super(AutoEncoder, self).__init__()
        self.arch_instance = arch_instance
        self.dataset = args.dataset
        self.use_se = args.use_se
        self.res_dist = args.res_dist
        self.num_bits = args.num_x_bits

        self.num_latent_scales = args.num_latent_scales         # number of spatial scales that latent layers will reside
        self.num_groups_per_scale = args.num_groups_per_scale   # number of groups of latent vars. per scale
        self.num_latent_per_group = args.num_latent_per_group   # number of latent vars. per group
        self.groups_per_scale = groups_per_scale(self.num_latent_scales, self.num_groups_per_scale, args.ada_groups,
                                                 minimum_groups=args.min_groups_per_scale)

        self.vanilla_vae = self.num_latent_scales == 1 and self.num_groups_per_scale == 1

        # encoder parameteres
        self.num_channels_enc = args.num_channels_enc
        self.num_channels_dec = args.num_channels_dec
        self.num_preprocess_blocks = args.num_preprocess_blocks  # block is defined as series of Normal followed by Down
        self.num_preprocess_cells = args.num_preprocess_cells   # number of cells per block
        self.num_cell_per_cond_enc = args.num_cell_per_cond_enc  # number of cell for each conditional in encoder

        # decoder parameters
        # self.num_channels_dec = args.num_channels_dec
        self.num_postprocess_blocks = args.num_postprocess_blocks
        self.num_postprocess_cells = args.num_postprocess_cells
        self.num_cell_per_cond_dec = args.num_cell_per_cond_dec  # number of cell for each conditional in decoder

        # general cell parameters
        self.input_size = get_input_size(self.dataset)

        # decoder param
        self.num_mix_output = 10

        # used for generative purpose
        c_scaling = CHANNEL_MULT ** (self.num_preprocess_blocks + self.num_latent_scales - 1)
        spatial_scaling = 2 ** (self.num_preprocess_blocks + self.num_latent_scales - 1)
        prior_ftr0_size = (int(c_scaling * self.num_channels_dec), self.input_size // spatial_scaling,
                           self.input_size // spatial_scaling)
        self.prior_ftr0 = nn.Parameter(torch.rand(size=prior_ftr0_size), requires_grad=True)
        self.z0_size = [self.num_latent_per_group, self.input_size // spatial_scaling, self.input_size // spatial_scaling]

        self.stem = self.init_stem()
        self.pre_process, mult = self.init_pre_process(mult=1)

        if self.vanilla_vae:
            self.enc_tower = []
        else:
            self.enc_tower, mult = self.init_encoder_tower(mult)

        self.with_nf = args.num_nf > 0
        self.num_flows = args.num_nf

        self.enc0 = self.init_encoder0(mult)
        self.enc_sampler, self.dec_sampler, self.nf_cells, self.enc_kv, self.dec_kv, self.query = \
            self.init_normal_sampler(mult)

        if self.vanilla_vae:
            self.dec_tower = []
            self.stem_decoder = Conv2D(self.num_latent_per_group, mult * self.num_channels_enc, (1, 1), bias=True)
        else:
            self.dec_tower, mult = self.init_decoder_tower(mult)

        self.post_process, mult = self.init_post_process(mult)

        self.image_conditional = self.init_image_conditional(mult)

        # collect all norm params in Conv2D and gamma param in batchnorm
        self.all_log_norm = []
        self.all_conv_layers = []
        self.all_bn_layers = []
        for n, layer in self.named_modules():
            # if isinstance(layer, Conv2D) and '_ops' in n:   # only chose those in cell
            if isinstance(layer, Conv2D) or isinstance(layer, ARConv2d):
                self.all_log_norm.append(layer.log_weight_norm)
                self.all_conv_layers.append(layer)
            if isinstance(layer, nn.BatchNorm2d):
                self.all_bn_layers.append(layer)

        print('len log norm:', len(self.all_log_norm))
        print('len bn:', len(self.all_bn_layers))
        # left/right singular vectors used for SR
        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4

    def init_stem(self):
        Cout = self.num_channels_enc
        Cin = 1 if self.dataset == 'mnist' else 3
        stem = Conv2D(Cin, Cout, 3, padding=1, bias=True)
        return stem

    def init_pre_process(self, mult):
        pre_process = nn.ModuleList()
        for b in range(self.num_preprocess_blocks):
            for c in range(self.num_preprocess_cells):
                if c == self.num_preprocess_cells - 1:
                    arch = self.arch_instance['down_pre']
                    num_ci = int(self.num_channels_enc * mult)
                    num_co = int(CHANNEL_MULT * num_ci)
                    cell = Cell(num_ci, num_co, cell_type='down_pre', arch=arch, use_se=self.use_se)
                    mult = CHANNEL_MULT * mult
                else:
                    arch = self.arch_instance['normal_pre']
                    num_c = self.num_channels_enc * mult
                    cell = Cell(num_c, num_c, cell_type='normal_pre', arch=arch, use_se=self.use_se)

                pre_process.append(cell)

        return pre_process, mult

    def init_encoder_tower(self, mult):
        enc_tower = nn.ModuleList()
        for s in range(self.num_latent_scales):
            for g in range(self.groups_per_scale[s]):
                for c in range(self.num_cell_per_cond_enc):
                    arch = self.arch_instance['normal_enc']
                    num_c = int(self.num_channels_enc * mult)
                    cell = Cell(num_c, num_c, cell_type='normal_enc', arch=arch, use_se=self.use_se)
                    enc_tower.append(cell)

                # add encoder combiner
                if not (s == self.num_latent_scales - 1 and g == self.groups_per_scale[s] - 1):
                    num_ce = int(self.num_channels_enc * mult)
                    num_cd = int(self.num_channels_dec * mult)
                    cell = EncCombinerCell(num_ce, num_cd, num_ce, cell_type='combiner_enc')
                    enc_tower.append(cell)

            # down cells after finishing a scale
            if s < self.num_latent_scales - 1:
                arch = self.arch_instance['down_enc']
                num_ci = int(self.num_channels_enc * mult)
                num_co = int(CHANNEL_MULT * num_ci)
                cell = Cell(num_ci, num_co, cell_type='down_enc', arch=arch, use_se=self.use_se)
                enc_tower.append(cell)
                mult = CHANNEL_MULT * mult

        return enc_tower, mult

    def init_encoder0(self, mult):
        num_c = int(self.num_channels_enc * mult)
        cell = nn.Sequential(
            nn.ELU(),
            Conv2D(num_c, num_c, kernel_size=1, bias=True),
            nn.ELU())
        return cell

    def init_normal_sampler(self, mult):
        enc_sampler, dec_sampler, nf_cells = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        enc_kv, dec_kv, query = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for s in range(self.num_latent_scales):
            for g in range(self.groups_per_scale[self.num_latent_scales - s - 1]):
                # build mu, sigma generator for encoder
                num_c = int(self.num_channels_enc * mult)
                cell = Conv2D(num_c, 2 * self.num_latent_per_group, kernel_size=3, padding=1, bias=True)
                enc_sampler.append(cell)
                # build NF
                for n in range(self.num_flows):
                    arch = self.arch_instance['ar_nn']
                    num_c1 = int(self.num_channels_enc * mult)
                    num_c2 = 8 * self.num_latent_per_group  # use 8x features
                    nf_cells.append(PairedCellAR(self.num_latent_per_group, num_c1, num_c2, arch))
                if not (s == 0 and g == 0):  # for the first group, we use a fixed standard Normal.
                    num_c = int(self.num_channels_dec * mult)
                    cell = nn.Sequential(
                        nn.ELU(),
                        Conv2D(num_c, 2 * self.num_latent_per_group, kernel_size=1, padding=0, bias=True))
                    dec_sampler.append(cell)

            mult = mult / CHANNEL_MULT

        return enc_sampler, dec_sampler, nf_cells, enc_kv, dec_kv, query

    
    def init_decoder_tower(self, mult):
        # create decoder tower
        dec_tower = nn.ModuleList()
        for s in range(self.num_latent_scales):
            for g in range(self.groups_per_scale[self.num_latent_scales - s - 1]):
                num_c = int(self.num_channels_dec * mult)
                if not (s == 0 and g == 0):
                    for c in range(self.num_cell_per_cond_dec):
                        arch = self.arch_instance['normal_dec']
                        cell = Cell(num_c, num_c, cell_type='normal_dec', arch=arch, use_se=self.use_se)
                        dec_tower.append(cell)

                cell = DecCombinerCell(num_c, self.num_latent_per_group, num_c, cell_type='combiner_dec')
                dec_tower.append(cell)

            # down cells after finishing a scale
            if s < self.num_latent_scales - 1:
                arch = self.arch_instance['up_dec']
                num_ci = int(self.num_channels_dec * mult)
                num_co = int(num_ci / CHANNEL_MULT)
                cell = Cell(num_ci, num_co, cell_type='up_dec', arch=arch, use_se=self.use_se)
                dec_tower.append(cell)
                mult = mult / CHANNEL_MULT

        return dec_tower, mult

    def init_post_process(self, mult):
        post_process = nn.ModuleList()
        for b in range(self.num_postprocess_blocks):
            for c in range(self.num_postprocess_cells):
                if c == 0:
                    arch = self.arch_instance['up_post']
                    num_ci = int(self.num_channels_dec * mult)
                    num_co = int(num_ci / CHANNEL_MULT)
                    cell = Cell(num_ci, num_co, cell_type='up_post', arch=arch, use_se=self.use_se)
                    mult = mult / CHANNEL_MULT
                else:
                    arch = self.arch_instance['normal_post']
                    num_c = int(self.num_channels_dec * mult)
                    cell = Cell(num_c, num_c, cell_type='normal_post', arch=arch, use_se=self.use_se)

                post_process.append(cell)

        return post_process, mult

    def init_image_conditional(self, mult):
        C_in = int(self.num_channels_dec * mult)
        C_out = 1 if self.dataset == 'mnist' else 10 * self.num_mix_output
        return nn.Sequential(nn.ELU(),
                             Conv2D(C_in, C_out, 3, padding=1, bias=True))

    def forward(self, x):
        s = self.stem(2 * x - 1.0)

        # perform pre-processing
        for cell in self.pre_process:
            s = cell(s)

        # run the main encoder tower
        combiner_cells_enc = []
        combiner_cells_s = []
        for cell in self.enc_tower:
            if cell.cell_type == 'combiner_enc':
                combiner_cells_enc.append(cell)
                combiner_cells_s.append(s)
            else:
                s = cell(s)

        # reverse combiner cells and their input for decoder
        combiner_cells_enc.reverse()
        combiner_cells_s.reverse()

        idx_dec = 0
        ftr = self.enc0(s)                            # this reduces the channel dimension
        param0 = self.enc_sampler[idx_dec](ftr)
        mu_q, log_sig_q = torch.chunk(param0, 2, dim=1)
        dist = Normal(mu_q, normal_sigma(log_sig_q))   # for the first approx. posterior
        z = dist.sample()
        log_q_conv = dist.log_prob(z)

        # apply normalizing flows
        nf_offset = 0
        for n in range(self.num_flows):
            z, log_det = self.nf_cells[n](z, ftr)
            log_q_conv -= log_det
        nf_offset += self.num_flows
        all_q = [dist]
        all_log_q = [log_q_conv]

        # To make sure we do not pass any deterministic features from x to decoder.
        s = 0

        # prior for z0
        dist = Normal(torch.zeros_like(z), torch.ones_like(z))
        log_p_conv = dist.log_prob(z)
        all_p = [dist]
        all_log_p = [log_p_conv]

        idx_dec = 0
        s = self.prior_ftr0.unsqueeze(0)
        batch_size = z.size(0)
        s = s.expand(batch_size, -1, -1, -1)
        for cell in self.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    # form prior
                    param = self.dec_sampler[idx_dec - 1](s)
                    mu_p, log_sig_p = torch.chunk(param, 2, dim=1)

                    # form encoder
                    ftr = combiner_cells_enc[idx_dec - 1](combiner_cells_s[idx_dec - 1], s)
                    param = self.enc_sampler[idx_dec](ftr)
                    mu_q, log_sig_q = torch.chunk(param, 2, dim=1)
                    dist = Normal(mu_p + mu_q, normal_sigma(log_sig_p + log_sig_q)) if self.res_dist else Normal(mu_q, normal_sigma(log_sig_q))
                    z = dist.sample()
                    log_q_conv = dist.log_prob(z)
                    # apply NF
                    for n in range(self.num_flows):
                        z, log_det = self.nf_cells[nf_offset + n](z, ftr)
                        log_q_conv -= log_det
                    nf_offset += self.num_flows
                    all_log_q.append(log_q_conv)
                    all_q.append(dist)

                    # evaluate log_p(z)
                    dist = Normal(mu_p, normal_sigma(log_sig_p))
                    log_p_conv = dist.log_prob(z)
                    all_p.append(dist)
                    all_log_p.append(log_p_conv)

                # 'combiner_dec'
                s = cell(s, z)
                idx_dec += 1
            else:
                s = cell(s)

        if self.vanilla_vae:
            s = self.stem_decoder(z)

        for cell in self.post_process:
            s = cell(s)

        logits = self.image_conditional(s)

        # compute kl
        kl_all = []
        kl_diag = []
        log_p, log_q = 0., 0.
        for q, p, log_q_conv, log_p_conv in zip(all_q, all_p, all_log_q, all_log_p):
            if self.with_nf:
                kl_per_var = log_q_conv - log_p_conv
            else:
                kl_per_var = torch.distributions.kl.kl_divergence(q, p)

            kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[2, 3]), dim=0))
            kl_all.append(torch.sum(kl_per_var, dim=[1, 2, 3]))
            log_q += torch.sum(log_q_conv, dim=[1, 2, 3])
            log_p += torch.sum(log_p_conv, dim=[1, 2, 3])

        return logits, log_q, log_p, kl_all, kl_diag

    def sample(self, num_samples, t):
        z0_size = [num_samples] + self.z0_size
        z = torch.normal(0, t, z0_size, device=self.prior_ftr0.device)
        return self.sample_given_z(z, t)

    def sample_given_z(self, z, t):
        idx_dec = 0
        s = self.prior_ftr0.unsqueeze(0)
        batch_size = z.size(0)
        s = s.expand(batch_size, -1, -1, -1)
        for cell in self.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    # form prior
                    param = self.dec_sampler[idx_dec - 1](s)
                    mu, log_sigma = torch.chunk(param, 2, dim=1)
                    z = torch.normal(mu, normal_sigma(log_sigma) if t == 1.0 else t * normal_sigma(log_sigma))

                # 'combiner_dec'
                s = cell(s, z)
                idx_dec += 1
            else:
                s = cell(s)

        if self.vanilla_vae:
            s = self.stem_decoder(z)

        for cell in self.post_process:
            s = cell(s)

        logits = self.image_conditional(s)
        return logits

    def decoder_output(self, logits):
        if self.dataset == 'mnist':
            return Bernoulli(logits=logits)
        elif self.dataset in {'cifar10', 'celeba_64', 'celeba_256', 'imagenet_32', 'imagenet_64', 'ffhq',
                              'lsun_bedroom_128', 'lsun_bedroom_256'}:
            return DiscMixLogistic(logits, self.num_mix_output, num_bits=self.num_bits)
        else:
            raise NotImplementedError
