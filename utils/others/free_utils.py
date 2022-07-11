import os
import copy
import numpy as np
from pprint import pprint


def print_visible(obj):
    pprint([a for a in dir(obj) if not a.startswith('_') and not a.endswith('_')])


def get_gauusian_kernel(shape):
    if len(shape) == 2:
        X, Y = shape
        target = np.zeros(shape, dtype=np.float32)
        for x in range(X):
            x_i = x if x <= (X - 1)/2 else (X - 1) - x
            for y in range(Y):
                y_i = y if y <= (Y - 1)/2 else (Y - 1) - y
                target[x, y] = x_i + y_i
        # t_max = (X - 1)//2 + (Y - 1)//2
        # assert t_max == target.max()
        # target = target / t_max
        return target


def get_gauusian_kernel_v2(shape):
    dims = len(shape)
    target = np.zeros(shape, dtype=np.float32)
    # assert dims in (1, 2, 3, 4)
    for axis in range(dims):
        shp = shape[axis]

        target_tmp = target if axis == 0 else np.swapaxes(target, 0, axis)

        for i in range(shp):
            val = i if i <= (shp - 1)/2 else (shp - 1) - i
            target_tmp[i, ...] += val

        target = target_tmp if axis == 0 else np.swapaxes(target_tmp, 0, axis)
    return target


def isVaildsStr(S, L):
    # 字符串S是否全部字符都属于字符串L
    assert isinstance(S, str)
    assert isinstance(L, str)
    if len(S) > len(L):
        return False
    vaild_char = []
    S_char = list(S)
    for i in range(len(L)):
        if S_char:
            print(S_char)
            if S_char[0] == L[i]:
                vaild_char.append(i)
                S_char.pop(0)
        else:
            print('vaild_char:', vaild_char)
            return True
    if S_char:
        return False
    else:
        print('vaild_char:', vaild_char)
        return True


# 不知道什么用，以后再看
def combineWord(start, total, *args):
    out_str = ''
    args_len = []
    args_list = []
    for arg in args:
        if isinstance(arg, str):
            args_len.append(len(arg))
            args_list.append(arg)
    args_len = args_len[:total]
    args_list = args_list[:total]
    out_str += args_list[start]
    args_len.pop(start)
    args_list.pop(start)
    for arg_l, arg in zip(args_len, args_list):
        if arg[0] != out_str[0]:
            args_len.remove(arg_l)
            args_list.remove(arg)
    max_len = max(args_len)
    for arg_l, arg in zip(args_len, args_list):
        if arg_l != max_len:
            args_len.remove(arg_l)
            args_list.remove(arg)
    min_str = args_list[0]
    for arg in args_list:
        if arg < min_str:
            min_str = arg
    out_str += min_str
    print(out_str)
    return out_str


def testGauusian_kernel():
    for tt in [(3,3,), (3,4), (3,5), (4, 4), (4,5), (5,4), (5,5), (3,3,3)]:
        print('get_gauusian_kernel:')
        print(get_gauusian_kernel(tt))
        print('get_gauusian_kernel_v2:')
        print(get_gauusian_kernel_v2(tt))


def testCombineWord(start, total,*args):
    print('args type:', type(args))
    ss='ace'
    ll='abcde'
    print(isVaildsStr(ss, ll))
    combineWord(4,6,'word','dd','da','dc','dword','d')


if __name__ == "__main__":
    pass

