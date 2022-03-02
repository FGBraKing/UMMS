""" Distributed training/validation utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import contextlib
from torch import distributed as dist


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    # ['BAND', 'BOR', 'BXOR', 'MAX', 'MIN', 'PRODUCT', 'SUM', 'name', 'value']
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def distribute_bn(model, world_size, reduce=False):
    # ensure every node has the same running bn stats
    for bn_name, bn_buf in model.named_buffers(recurse=True):
        if ('running_mean' in bn_name) or ('running_var' in bn_name):
            if reduce:
                # average bn stats across whole group
                dist.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                bn_buf /= float(world_size)
            else:
                # broadcast bn stats from rank 0 to whole group
                dist.broadcast(bn_buf, 0)


def distribute_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]


def record_distribute_ddp(opt):
    opt.world_size = dist.get_world_size()
    opt.rank = dist.get_rank()
    opt.backend = dist.get_backend()
    return opt


@contextlib.contextmanager
def torch_distributed_zero_first(rank: int):
    # rank = -1, 不执行
    # rank = 0， with语句执行后同步
    # rank = 其他， with语句执行前同步
    if rank not in [-1, 0]:
        dist.barrier()
    yield
    if rank == 0:
        dist.barrier()


def torch_condition_zero_first(condition1, condition2):
    if condition1:
        dist.barrier()
    yield
    if condition2:
        dist.barrier()

# ['AllToAllOptions',
#  'AllreduceCoalescedOptions',
#  'AllreduceOptions',
#  'Backend',
#  'BarrierOptions',
#  'BroadcastOptions',
#  'BuiltinCommHookType',
#  'Dict',
#  'FileStore',
#  'GatherOptions',
#  'GroupMember',
#  'HashStore',
#  'Optional',
#  'P2POp',
#  'PrefixStore',
#  'ProcessGroup',
#  'ProcessGroupGloo',
#  'ProcessGroupNCCL',
#  'ReduceOp',
#  'ReduceOptions',
#  'ReduceScatterOptions',
#  'Reducer',
#  'STORE_BASED_BARRIER_PREFIX',
#  'ScatterOptions',
#  'Store',
#  'TCPStore',
#  'Tuple',
#  'Union',
#  'all_gather',
#  'all_gather_coalesced',
#  'all_gather_multigpu',
#  'all_gather_object',
#  'all_reduce',
#  'all_reduce_coalesced',
#  'all_reduce_multigpu',
#  'all_to_all',
#  'all_to_all_single',
#  'autograd',
#  'barrier',
#  'batch_isend_irecv',
#  'broadcast',
#  'broadcast_multigpu',
#  'broadcast_object_list',
#  'constants',
#  'contextlib',
#  'default_pg_timeout',
#  'destroy_process_group',
#  'dist_backend',
#  'distributed_c10d',
#  'gather',
#  'gather_object',
#  'get_backend',
#  'get_rank',
#  'get_world_size',
#  'group',
#  'init_process_group',
#  'irecv',
#  'is_available',
#  'is_gloo_available',
#  'is_initialized',
#  'is_mpi_available',
#  'is_nccl_available',
#  'isend',
#  'logging',
#  'new_group',
#  'pickle',
#  'recv',
#  'reduce',
#  'reduce_multigpu',
#  'reduce_op',
#  'reduce_scatter',
#  'reduce_scatter_multigpu',
#  'register_rendezvous_handler',
#  'rendezvous',
#  'rpc',
#  'scatter',
#  'scatter_object_list',
#  'send',
#  'string_classes',
#  'supports_complex',
#  'sys',
#  'time',
#  'timedelta',
#  'torch',
#  'warnings']
