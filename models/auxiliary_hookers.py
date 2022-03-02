import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.utils as vutils
from collections import OrderedDict
from graphviz import Digraph


def print_value():
    grads = {}

    def save_grad(name):
        def hook(grad):
            grads[name] = grad
        return hook

    x = Variable(torch.randn(1, 1), requires_grad=True)
    y = 3 * x
    z = y ** 2

    # In here, save_grad('y') returns a hook (a function) that keeps 'y' as name
    y.register_hook(save_grad('y'))
    z.register_hook(save_grad('z'))
    z.backward()
    print('HW')
    print("grads['y']: {}".format(grads['y']))
    print(grads['z'])


def print_layers_num(net):  # 得到模型的层数
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                print(' ')
                # 可以用来统计不同层的个数
                # net.register_backward_hook(print)
            return 1
        count = 0
        for c in childrens:
            count += foo(c)
        return count

    print(foo(net))


def print_forward():
    model = torchvision.models.resnet18()
    select_layer = model.layer1[0].conv1

    grads = {}

    def save_grad(name):
        def hook(module, input, output):
            grads[name] = input
        return hook

    select_layer.register_forward_hook(save_grad('select_layer'))

    input = Variable(torch.rand(3, 224, 224).unsqueeze(0), requires_grad=True)
    out = model(input)
    # print grads['select_layer']
    print(grads)


def print_autograd_graph():  # 得到模型的计算图
    def make_dot(var, params=None):
        """ Produces Graphviz representation of PyTorch autograd graph

        Blue nodes are the Variables that require grad, orange are Tensors
        saved for backward in torch.autograd.Function

        Args:
            var: output Variable
            params: dict of (name, Variable) to add names to node that
                require grad
        """
        if params is not None:
            # assert all(isinstance(p, Variable) for p in params.values())
            param_map = {id(v): k for k, v in params.items()}

        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='12',
                         ranksep='0.1',
                         height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
        seen = set()

        def size_to_str(size):
            return '(' + (', ').join(['%d' % v for v in size]) + ')'

        def add_nodes(var):
            if var not in seen:
                if torch.is_tensor(var):
                    dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
                elif hasattr(var, 'variable'):
                    u = var.variable
                    # name = param_map[id(u)] if params is not None else ''
                    # node_name = '%s\n %s' % (name, size_to_str(u.size()))
                    node_name = '%s\n %s' % (param_map.get(id(u.data)), size_to_str(u.size()))
                    dot.node(str(id(var)), node_name, fillcolor='lightblue')

                else:
                    dot.node(str(id(var)), str(type(var).__name__))
                seen.add(var)
                if hasattr(var, 'next_functions'):
                    for u in var.next_functions:
                        if u[0] is not None:
                            dot.edge(str(id(u[0])), str(id(var)))
                            add_nodes(u[0])
                if hasattr(var, 'saved_tensors'):
                    for t in var.saved_tensors:
                        dot.edge(str(id(t)), str(id(var)))
                        add_nodes(t)

        add_nodes(var.grad_fn)
        return dot

    from torchvision import models

    torch.manual_seed(1)
    inputs = torch.randn(1, 3, 224, 224)
    model = models.resnet18(pretrained=False)
    y = model(Variable(inputs))
    # print(y)

    g = make_dot(y, params=model.state_dict())
    g.view()
    # g


def show_save_tensor():
    import torch
    from torchvision import utils
    import torchvision.models as models
    from matplotlib import pyplot as plt

    def vis_tensor(tensor, ch=0, all_kernels=False, nrow=8, padding=2):
        '''
        ch: channel for visualization
        allkernels: all kernels for visualization
        '''
        n, c, h, w = tensor.shape
        if all_kernels:
            tensor = tensor.view(n * c, -1, w, h)
        elif c != 3:
            tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

        rows = np.min((tensor.shape[0] // nrow + 1, 64))
        grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
        # plt.figure(figsize=(nrow,rows))
        plt.imshow(grid.numpy().transpose((1, 2, 0)))  # CHW HWC

    def save_tensor(tensor, filename, ch=0, all_kernels=False, nrow=8, padding=2):
        n, c, h, w = tensor.shape
        if all_kernels:
            tensor = tensor.view(n * c, -1, w, h)
        elif c != 3:
            tensor = tensor[:, ch, :, :].unsqueeze(dim=1)
        utils.save_image(tensor, filename, nrow=nrow, normalize=True, padding=padding)

    vgg = models.resnet18(pretrained=True)
    mm = vgg.double()
    filters = mm.modules
    body_model = [i for i in mm.children()][0]
    # layer1 = body_model[0]
    layer1 = body_model
    tensor = layer1.weight.data.clone()
    vis_tensor(tensor)
    save_tensor(tensor, 'test.png')

    plt.axis('off')
    plt.ioff()
    plt.show()


def torch_summarize_df(input_size, model, weights=False, input_shape=True, nb_trainable=False):
    """
    Summarizes torch model by showing trainable parameters and weights.

    author: wassname
    url: https://gist.github.com/wassname/0fb8f95e4272e6bdd27bd7df386716b7
    license: MIT

    Modified from:
    - https://github.com/pytorch/pytorch/issues/2001#issuecomment-313735757
    - https://gist.github.com/wassname/0fb8f95e4272e6bdd27bd7df386716b7/

    Usage:
        import torchvision.models as models
        model = models.alexnet()
        df = torch_summarize_df(input_size=(3, 224,224), model=model)
        print(df)

        #              name class_name        input_shape       output_shape  nb_params
        # 1     features=>0     Conv2d  (-1, 3, 224, 224)   (-1, 64, 55, 55)      23296#(3*11*11+1)*64
        # 2     features=>1       ReLU   (-1, 64, 55, 55)   (-1, 64, 55, 55)          0
        # ...
    """

    def register_hook(module):
        def hook(module, input, output):
            name = ''
            for key, item in names.items():
                if item == module:
                    name = key
            # <class 'torch.nn.modules.conv.Conv2d'>
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = module_idx + 1

            summary[m_key] = OrderedDict()
            summary[m_key]['name'] = name
            summary[m_key]['class_name'] = class_name
            if input_shape:
                summary[m_key][
                    'input_shape'] = (-1,) + tuple(input[0].size())[1:]
            summary[m_key]['output_shape'] = (-1,) + tuple(output.size())[1:]
            if weights:
                summary[m_key]['weights'] = list(
                    [tuple(p.size()) for p in module.parameters()])

            #             summary[m_key]['trainable'] = any([p.requires_grad for p in module.parameters()])
            if nb_trainable:
                params_trainable = sum(
                    [torch.LongTensor(list(p.size())).prod() for p in module.parameters() if p.requires_grad])
                summary[m_key]['nb_trainable'] = params_trainable
            params = sum([torch.LongTensor(list(p.size())).prod() for p in module.parameters()])
            summary[m_key]['nb_params'] = params

        if not isinstance(module, nn.Sequential) and \
                not isinstance(module, nn.ModuleList) and \
                not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # Names are stored in parent and path+name is unique not the name
    names = get_names_dict(model)

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1, *in_size)) for in_size in input_size]
    else:
        x = Variable(torch.rand(1, *input_size))

    if next(model.parameters()).is_cuda:
        x = x.cuda()

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    # make dataframe
    df_summary = pd.DataFrame.from_dict(summary, orient='index')
    print(df_summary)


# https://gist.github.com/wassname/0fb8f95e4272e6bdd27bd7df386716b7
# summarize a torch model like in keras, showing parameters and output shape
def get_names_dict(model):
    """
    Recursive walk to get names including path
    """
    names = {}

    def _get_names(module, parent_name=''):
        for key, module in module.named_children():
            name = parent_name + '.' + key if parent_name else key
            names[name] = module
            if isinstance(module, torch.nn.Module):
                _get_names(module, parent_name=name)

    _get_names(model)
    return names


# ----------------------------------------------CUSTOM -------------------------------------
# 提取指定层输出的特征和输出特征的梯度
class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers: list):
        '''
        :param model:
        :param target_layers: list, module_name
        '''
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    # define hooker, another way is tensor.retain_graph()
    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model.named_children():
            x = module(x)
            print('name=', name)
            print('x.size()=', x.size())
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


# register_forward_hook
# register_forward_pre_hook
# 提取指定层的名字和输入输出特征图
class FeatureMapExtractor:
    def __init__(self):
        self.features_in = []
        self.features_out = []
        self.module_name_in = []
        self.module_name_out = []

    def feature_out_hooker(self, module, feature_in, feature_out):
        '''
        :param module: torch.nn.module
        :param feature_in: position argument, tuple
        :param feature_out:
        :return:
        '''
        self.module_name_out.append(module.__class__)
        # self.features_in.append(feature_in)
        self.features_out.append(feature_out)

    def feature_in_hooker(self, module, feature_in):
        '''
        :param module: torch.nn.module
        :param feature_in: position argument, tuple
        :return:
        '''
        self.module_name_in.append(module.__class__)
        self.features_in.append(feature_in)

    def reset(self):
        self.features_in = []
        self.features_out = []
        self.module_name_in = []
        self.module_name_out = []

    def hook_the_model(self, model, layers=None):
        if layers:
            for name, module in model.named_modules():
                if name in layers:
                    module.register_forward_hook(self.feature_out_hooker)
                    module.register_forward_pre_hook(self.feature_in_hooker)
        else:
            for module in model.modules():
                module.register_forward_hook(self.feature_out_hooker)
                module.register_forward_pre_hook(self.feature_in_hooker)

    def get_feature(self):
        assert self.module_name_in == self.module_name_out
        return self.module_name_in, self.features_in, self.features_out

    def get_in_feature(self):
        return self.module_name_in, self.features_in

    def get_out_feature(self):
        return self.module_name_out, self.features_out


# register_backward_hook
# 提取指定层输出特征图的梯度,
# 在测试中，这个grad_input的结果有些奇怪,毕竟grad_input是该层用于计算梯度的输入
# grad_output的结果是正确的
class FeatureGradientExtractor:
    def __init__(self):
        self.module_name = []
        self.grad_input = []
        self.grad_output = []

    def reset(self):
        self.module_name = []
        self.grad_input = []
        self.grad_output = []

    def grad_hook(self, module, grad_input, grad_output):
        self.module_name.append(module.__class__)
        self.grad_input.append(grad_input)
        self.grad_output.append(grad_output[0])

    def hook_the_model(self, model, layers=None):
        if layers:
            for name, module in model.named_modules():
                assert isinstance(module, torch.nn.Module)
                if name in layers:
                    module.register_backward_hook(self.grad_hook)
        else:
            for module in model.modules():
                module.register_backward_hook(self.grad_hook)

    def get_grad(self):
        return self.module_name, self.grad_output

    def get_all_grad(self):
        return self.module_name, self.grad_input, self.grad_output


# register_hook
# 获取指定层的权重的梯度
class WeightGradientExtractor:
    def __init__(self):
        self.grad = []

    def hook_grad(self, gard):
        self.grad.append(gard)

    def hook_model_weight_grad(self, model, layer):
        '''
        :param model: module
        :param layer: module_name
        :return:
        '''
        assert isinstance(model, torch.nn.Module)
        for name, module in model.named_modules():
            if name == layer:
                for name1, paras in module.named_parameters():
                    assert isinstance(paras, torch.Tensor)
                    # print(name1, paras.size())
                    # 先register weight 再register bias，但self.grad是bias在前，weight在后
                    paras.register_hook(self.hook_grad)

    def get_grad(self):
        return self.grad

    def reset(self):
        self.grad = []


# 提取指定层的weight
def get_model_weight(model, layer):
    weights = []
    bias = []
    for name, paras in model.named_parameters():
        if name.find(layer) != -1 and 'weight' in name:
            weights.append(paras)
        if name.find(layer) != -1 and 'bias' in name:
            bias.append(paras)
    return weights, bias


def vis_weight_2d(net, vis_fn=None):
    for k, v in net.named_parameters():
        if 'conv' in k and 'weight' in k:
            c_int = v.size()[1]     # 输入层通道数
            c_out = v.size()[0]     # 输出层通道数
            # 以feature map为单位，绘制一组卷积核，一张feature map对应的卷积核个数为输入通道数
            for j in range(c_out):
                print(k, v.size(), j)
                kernel_j = v[j, :, :, :].unsqueeze(1)       # 压缩维度，为make_grid制作输入, (c_int,1,w,h)
                kernel_grid = vutils.make_grid(kernel_j, normalize=True, scale_each=True, nrow=c_int)   # 1*输入通道数, w, h
                if vis_fn:
                    vis_fn(kernel_grid)
            # 将一个卷积层的卷积核绘制在一起，每一行是一个feature map的卷积核
            k_w, k_h = v.size()[-1], v.size()[-2]
            kernel_all = v.view(-1, 1, k_w, k_h)
            kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=c_int)  # 1*输入通道数, w, h
            if vis_fn:
                vis_fn(kernel_grid)

