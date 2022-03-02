import math
import torch
import functools
import numpy as np

from torch import nn as nn
from torch.nn import init
from torch.nn import functional as F


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


# from CHAOS
def print_model_parm_nums(net):  # count and print total number of parameters
    total = sum([param.nelement() for param in net.parameters()])
    print('  + Number of params: %.4fM' % (total / 1e6))  # 每一百万为一个单位
    print('  + Memory of params(float32 x4): %.4fM' % (4 * total / 1e6))  # 每一百万为一个单位
    print('  + Total Memory(SGD x3;Adam x4) of params(float32 x4): %.4fM' % (3 * 4 * total / 1e6))  # 每一百万为一个单位


def print_model_parm_flops(net, input, need_idx=True):  # 得到模型计算量
    prods = {}
    if len(input.size()) == 5:
        print('batch_size, output_channels, output_depth, output_height, output_width, params, feature pixel')
    else:
        print('batch_size, output_channels, output_height, output_width, params, feature pixel')

    def save_hook(name):
        def hook_per(self, input, output):
            # print 'flops:{}'.format(self.__class__.__name__)
            # print 'input:{}'.format(input)
            # print '_dim:{}'.format(input[0].dim())
            # print 'input_shape:{}'.format(np.prod(input[0].shape))
            # prods.append(np.prod(input[0].shape))
            prods[name] = np.prod(input[0].shape)
            # prods.append(np.prod(input[0].shape))

        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}

    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    list_feature_pixel = []
    multiply_adds = False
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        feature_pixel = batch_size * output_channels * output_height * output_width

        print('conv:', batch_size, output_channels, output_height, output_width, params, feature_pixel)

        list_feature_pixel.append(feature_pixel)

        list_conv.append(flops)

    list_conv3d = []

    def conv3d_hook(self, input, output):
        batch_size, input_channels, input_depth, input_height, input_width = input[0].size()
        output_channels, output_depth, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * (
                    self.in_channels / self.groups) * (
                         2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_depth * output_height * output_width

        feature_pixel = batch_size * output_channels * output_depth * output_height * output_width

        print('conv3d:', batch_size, output_channels, output_depth, output_height, output_width, params, feature_pixel)

        list_feature_pixel.append(feature_pixel)

        list_conv3d.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_gn = []

    def gn_hook(self, input, output):
        list_gn.append(input[0].nelement())

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_leakyrelu = []

    def leakyrelu_hook(self, input, output):
        list_leakyrelu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        feature_pixel = batch_size * output_channels * output_height * output_width

        print('pool:', batch_size, output_channels, output_height, output_width, params, feature_pixel)

        list_feature_pixel.append(feature_pixel)

        list_pooling.append(flops)

    list_pooling3d = []

    def pooling3d_hook(self, input, output):
        batch_size, input_channels, input_depth, input_height, input_width = input[0].size()
        output_channels, output_depth, output_height, output_width = output[0].size()

        if isinstance(self.kernel_size, int):
            kernel_ops = self.kernel_size ** 3
        else:
            kernel_ops = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_depth * output_height * output_width

        feature_pixel = batch_size * output_channels * output_depth * output_height * output_width

        print('pool3d:', batch_size, output_channels, output_depth, output_height, output_width, params, feature_pixel)

        list_feature_pixel.append(feature_pixel)

        list_pooling3d.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Conv3d):
                net.register_forward_hook(conv3d_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d) or isinstance(net, torch.nn.BatchNorm3d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.GroupNorm):
                net.register_forward_hook(gn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.LeakyReLU):
                net.register_forward_hook(leakyrelu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.MaxPool3d) or isinstance(net, torch.nn.AvgPool3d):
                net.register_forward_hook(pooling3d_hook)
            return
        for c in childrens:
            foo(c)

    foo(net)

    if need_idx is True:
        out = net(input, 0)
    else:
        out = net(input)

    num_conv = sum(list_conv)
    num_conv3d = sum(list_conv3d)
    num_linear = sum(list_linear)
    num_bn = sum(list_bn)
    num_gn = sum(list_gn)
    num_relu = sum(list_relu)
    num_leakyrelu = sum(list_leakyrelu)
    num_pooling = sum(list_pooling)
    num_pooling3d = sum(list_pooling3d)
    num_feature_pixel = sum(list_feature_pixel)
    total_flops = (
                num_conv + num_conv3d + num_linear + num_bn + num_gn + num_relu + num_leakyrelu + num_pooling + num_pooling3d)

    print('  + Number of FLOPs: %.4fG' % (total_flops / 1e9))
    print('  + Pixel of Feature: %.4fM' % (num_feature_pixel / 1e6))
    print('  + Memory of Feature(float32 x4): %.4fG' % (4 * num_feature_pixel / 1e9))
    print('  + Total Memory(fp&bp) of Feature(float32 x4): %.4fG' % (2 * 4 * num_feature_pixel / 1e9))

    print(f"""
    num_conv = {num_conv}
    num_conv3d = {num_conv3d}
    num_linear = {num_linear}
    num_bn = {num_bn}
    num_gn = {num_gn}
    num_relu = {num_relu}
    num_leakyrelu = {num_leakyrelu}
    num_pooling = {num_pooling}
    num_pooling3d = {num_pooling3d}
    """)


def find_maximum_patch_size(model, device):
    """Tries to find the biggest patch size that can be send to GPU for inference
    without throwing CUDA out of memory"""
    in_channels = model.in_channels

    patch_shapes = [(64, 128, 128), (96, 128, 128),
                    (64, 160, 160), (96, 160, 160),
                    (64, 192, 192), (96, 192, 192)]

    for shape in patch_shapes:
        # generate random patch of a given size
        patch = np.random.randn(*shape).astype('float32')

        patch = torch \
            .from_numpy(patch) \
            .view((1, in_channels) + patch.shape) \
            .to(device)

        print(f"Current patch size: {shape}")
        model(patch)


# ---------------------------------custom--------------------------------------------

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    init_func = get_init_func(init_type, init_gain)
    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
        # net = net.module
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_network_numpool(patch_size, maxpool_cap=999, min_feature_map_size=4):
    network_numpool_per_axis = np.floor([np.log(i / min_feature_map_size) / np.log(2) for i in patch_size]).astype(int)
    network_numpool_per_axis = [min(i, maxpool_cap) for i in network_numpool_per_axis]
    return network_numpool_per_axis


def get_shape_must_be_divisible_by(net_numpool_per_axis):
    return 2 ** np.array(net_numpool_per_axis)


def pad_shape(shape, must_be_divisible_by):
    """
    pads shape so that it is divisibly by must_be_divisible_by
    :param shape:
    :param must_be_divisible_by:
    :return:
    """
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    else:
        assert len(must_be_divisible_by) == len(shape)

    new_shp = [shape[i] + must_be_divisible_by[i] - shape[i] % must_be_divisible_by[i] for i in range(len(shape))]

    for i in range(len(shape)):
        if shape[i] % must_be_divisible_by[i] == 0:
            new_shp[i] -= must_be_divisible_by[i]
    new_shp = np.array(new_shp).astype(int)
    return new_shp


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


# -------------------------------------------class-------------------------------------------------------
class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Mish(nn.Module):
    '''
    x * torch.tanh(torch.nn.functional.softplus(x))
    '''
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * (F.tanh(F.softplus(x)))
        return x


class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'mish':
            self.activation = Mish()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif name == 'argmax3d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)


# -------------------------------------------function---------------
# [N, 1, *]
def make_one_hot(tensor, num_classes=None, with_channel=True, ignore_index=None):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         tensor: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
         with_channel:
         ignore_index
    Shapes:
        predict: A tensor of shape [N, *] without sigmoid activation function applied
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    if not with_channel:
        tensor = tensor.unsqueeze(1)
    if num_classes is None:
        num_classes = tensor.max() + 1
    elif num_classes == 1:
        return tensor
    shape = np.array(tensor.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).scatter_(1, tensor.cpu().long(), 1)  # dim=1, index=input.cpu().long(), src=1

    if ignore_index is not None:
        # result[tensor.expand(shape) == ignore_index] = ignore_index  # 该样本所有类别通道置ignore_index
        result[tensor.expand(shape) == ignore_index] = 0          # 该样本所有类别通道清零，则该样本无类别

        # # create ignore_index mask for the result
        # expanded_src = src.expand(shape)
        # mask = expanded_src == ignore_index
        # # clone the src tensor and zero out ignore_index in the input
        # src = src.clone()
        # src[src == ignore_index] = 0      # 将该样本类别变成第0类
        # # scatter to get the one-hot tensor
        # result = torch.zeros(shape).to(input.device).scatter_(1, src.long(), 1)
        # # bring back the ignore_index in the result
        # result[mask] = ignore_index
        # return result

    return result


def flatten_by_class(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    Expand by class
    """
    if tensor.dim() >= 3:
        C = tensor.size(1)
        # new axis order
        axis_order = (1, 0) + tuple(range(2, tensor.dim()))
        # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
        transposed = tensor.permute(axis_order)
        # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)

        return transposed.contiguous().view(C, -1)
    else:
        return tensor.view(1, -1)


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    input = input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)

# --------------------------------------------api-----------------------------------------------------------------


def get_activation(activation, **kwargs):
    if str(activation).lower() == "relu":
        return nn.ReLU(inplace=True)
    elif str(activation).lower() == "elu":
        return nn.ELU(alpha=1., inplace=True)
    elif str(activation).lower() == "leakyrelu":
        return nn.LeakyReLU(negative_slope=1e-2, inplace=True)
    elif str(activation).lower() == "prelu":
        return nn.PReLU(num_parameters=1, init=0.25)
    elif str(activation).lower() == "relu6":
        return nn.ReLU6(inplace=True)
    elif str(activation).lower() == "rrelu":
        return nn.RReLU(inplace=True)
    elif str(activation).lower() == "rrelu":
        return nn.CELU(inplace=True)
    elif str(activation).lower() == "selu":
        return nn.SELU(inplace=True)
    elif str(activation).lower() == "mish":
        return Mish()
    elif str(activation).lower() == "sigmoid":
        return nn.Sigmoid()
    elif str(activation).lower() == "softmax":
        return nn.Softmax(dim=1)
    elif str(activation).lower() == 'tanh':
        return nn.Tanh()
    elif str(activation).lower() == 'softplus':
        return nn.Softplus()
    elif str(activation).lower() == 'argmax':
        return ArgMax(**kwargs)
    elif str(activation).lower() == 'argmax2d':
        return ArgMax(dim=1, **kwargs)
    elif callable(activation):
        return activation(**kwargs)
    else:
        raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(activation))


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_3dnorm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_init_func(init_type='gaussian', init_gain=math.sqrt(2), init_std=0.02):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, mean=0.0, std=0.02)
            elif init_type == 'normal':
                init.normal_(m.weight.data, mean=0.0, std=0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)     # math.sqrt(2)
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                # m.bias.data.zero_()
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:
            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return init_fun


def create_conv(conv_type):
    pass


def get_attn(attn_type):
    pass
# def get_attn(attn_type):
#     if isinstance(attn_type, torch.nn.Module):
#         return attn_type
#     module_cls = None
#     if attn_type is not None:
#         if isinstance(attn_type, str):
#             attn_type = attn_type.lower()
#             # Lightweight attention modules (channel and/or coarse spatial).
#             # Typically added to existing network architecture blocks in addition to existing convolutions.
#             if attn_type == 'se':
#                 module_cls = SEModule
#             elif attn_type == 'ese':
#                 module_cls = EffectiveSEModule
#             elif attn_type == 'eca':
#                 module_cls = EcaModule
#             elif attn_type == 'ecam':
#                 module_cls = partial(EcaModule, use_mlp=True)
#             elif attn_type == 'ceca':
#                 module_cls = CecaModule
#             elif attn_type == 'ge':
#                 module_cls = GatherExcite
#             elif attn_type == 'gc':
#                 module_cls = GlobalContext
#             elif attn_type == 'cbam':
#                 module_cls = CbamModule
#             elif attn_type == 'lcbam':
#                 module_cls = LightCbamModule
#
#             # Attention / attention-like modules w/ significant params
#             # Typically replace some of the existing workhorse convs in a network architecture.
#             # All of these accept a stride argument and can spatially downsample the input.
#             elif attn_type == 'sk':
#                 module_cls = SelectiveKernel
#             elif attn_type == 'splat':
#                 module_cls = SplitAttn
#
#             # Self-attention / attention-like modules w/ significant compute and/or params
#             # Typically replace some of the existing workhorse convs in a network architecture.
#             # All of these accept a stride argument and can spatially downsample the input.
#             elif attn_type == 'lambda':
#                 return LambdaLayer
#             elif attn_type == 'bottleneck':
#                 return BottleneckAttn
#             elif attn_type == 'halo':
#                 return HaloAttn
#             elif attn_type == 'swin':
#                 return WindowAttention
#             elif attn_type == 'involution':
#                 return Involution
#             elif attn_type == 'nl':
#                 module_cls = NonLocalAttn
#             elif attn_type == 'bat':
#                 module_cls = BatNonLocalAttn
#
#             # Woops!
#             else:
#                 assert False, "Invalid attn module (%s)" % attn_type
#         elif isinstance(attn_type, bool):
#             if attn_type:
#                 module_cls = SEModule
#         else:
#             module_cls = attn_type
#     return module_cls
#
#
# def create_attn(attn_type, channels, **kwargs):
#     module_cls = get_attn(attn_type)
#     if module_cls is not None:
#         # NOTE: it's expected the first (positional) argument of all attention layers is the # input channels
#         return module_cls(channels, **kwargs)
#     return None
