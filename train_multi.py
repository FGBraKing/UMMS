import os
import sys
import time
import logging
# import matplotlib
import numpy as np
import torch.distributed
from pprint import pprint
from matplotlib import pyplot as plt
from configs.simple_options import get_opt
from configs.utils_config import pretty_print_opt, get_pretty_opt
from data import create_dataset, create_test_dataset
from models import create_model
from utils.forLogs import Visualizer, get_logger
from utils.others.utils import init_seed, init_torch, print_numpy, mkdirs, DataPool, get_device_name
from utils.others.distributed_utils import record_distribute_ddp, torch_distributed_zero_first
from utils.others.img_io import show_array_3d, show_volume_label, show_volume_label_predict, show_image, show_paired_image
# matplotlib.use('TKAgg')

from configs.excess_config import ex_config

save_threshold = 0.70
pool_size = 3


def set_local_gpu(args):
    if not args.DDP:
        # args.local_rank = args.gpu_ids[0] if args.gpu_ids else -1
        args.local_gpu = args.gpu_ids[0] if args.gpu_ids else - 1
    elif args.dist_url == 'env://':
        args.local_rank = int(os.environ["LOCAL_RANK"])
        assert args.local_rank >= 0, 'LOCAL_RANK must >= 0'
        args.local_gpu = args.gpu_ids[args.local_rank]
    else:
        args.local_gpu = args.gpu_ids[args.local_rank]

    if args.local_gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.local_gpu)   # setup default cuda device
    return args


def train():
    opt = get_opt(args=['--config_path=configs/defaults/dsbnwithedge_train.yaml', '--use_config', '--use_current_local_rank'])
    # dsbnwithauxtask_train   dsbnpluswithedge_train  priorda_train dualstreamtranswithprior_train
    init_torch(gpu_id=opt.visible_gpu, deterministic=opt.deterministic)
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    do_train(opt)


def do_train(opt):
    print('now is in do_train, if you are using DDP, please make sure that '
          'you had got (dist_backend, dist_url, world_size, rank, local_rank) ready')
    print('CUDA_VISIBLE_DEVICES: '+os.environ['CUDA_VISIBLE_DEVICES'])

    device_name = get_device_name()
    opt.name = opt.name + '_' + device_name if device_name is not None else opt.name
    # ====================================================配置gpu等全局变量==============================================
    opt.random_state = np.random.RandomState(seed=opt.seed)

    # print(torch.cuda.is_available())
    # setup default cuda device, 配合tensor.cuda()使用
    opt = set_local_gpu(opt)
    print('local_gpu:{}'.format(opt.local_gpu))
    # print('cuda is_available:', torch.distributed.is_available())

    if opt.DDP and torch.distributed.is_available():
        torch.distributed.init_process_group(backend=opt.dist_backend,
                                             init_method=opt.dist_url,
                                             world_size=opt.world_size,
                                             rank=opt.rank)
        print('backend:{}, dist_method:{}'.format(repr(torch.distributed.get_backend()), opt.dist_url))
        print('local_rank:{}, rank:{}, world_size:{}'.format(opt.local_rank,
                                                             torch.distributed.get_rank(),
                                                             torch.distributed.get_world_size()))
        # print(opt.dist_backend, opt.dist_url)
        torch.cuda.empty_cache()
        # 通过这一步把初始化后的rank等参数存入opt，统一不同框架的用法

    on_master = (not opt.DDP) or (opt.DDP and opt.rank == 0)
    init_seed(opt.seed + (opt.rank if opt.DDP else 0))

    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)  # opt.dataset_name + opt.model_name + opt.name
    opt_save_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
    if on_master:
        mkdirs(expr_dir)
        opt_logger = get_logger(logname='opt_logger', level=logging.INFO,
                                is_save=opt.save_log and not opt.DEBUG, save_name=opt_save_name, fmt="%(message)s")
    else:
        opt_logger = get_logger(logname='opt_logger', level=logging.WARNING, is_save=False, fmt="%(message)s")
    opt_logger.info(get_pretty_opt(opt))

    # setting ddp_logger
    ddp_logger = get_logger(logname='ddp_logger', level=logging.INFO if on_master else logging.WARNING, is_save=False,
                            fmt="[%(process)d][%(filename)s][%(funcName)s]%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    time_logger = get_logger(logname='time_logger', level=logging.INFO if on_master else logging.WARNING, is_save=False,
                             fmt="[%(process)d]%(message)s")    # [%(thread)d]

    # ========================================数据,模型,初始化、优化器、学习率策略========================================

    dataloader = create_dataset(opt, opt.proxy_two)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataloader)    # get the number of images in the dataset.
    dataloader_size = dataloader.get_loader_size()
    ddp_logger.warning('The number of training images = %d' % dataset_size)

    test_dataloader = create_test_dataset(opt)
    test_dataset_size = len(test_dataloader)
    ddp_logger.warning('The number of testing images = %d' % test_dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks
    ddp_logger.warning('model get ready')

    if opt.lr_policy in ['poly', 'tanh', 'cosine']:
        # 更正num_epochs
        opt.num_epochs = model.get_schedulers()[0].get_cycle_length(opt.lr_cycle_num) + opt.cooldown_epochs

    optimize_parameters = model.optimize_parameters_with_apex if opt.APEX else model.optimize_parameters
    save_networks = model.save_for_apex if opt.APEX else model.save_networks

    if on_master:
        visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
        ddp_logger.info('visualizer get ready')
        if opt.draw_model:
            [visualizer.draw_model_graph(net, shape=[4, 1]+list(opt.crop_size[::-1])) for net in model.get_models()]
    else:
        visualizer = None

    if opt.DDP:
        torch.distributed.barrier()

    if opt.use_gradient_accumulation:
        print('using gradient accumulation, and that you can not use apex now')
        if opt.gradient_accumulation_k_step > dataloader_size:
            print('gradient_accumulation_k_step is too big, and is setted default of 1')
            opt.gradient_accumulation_k_step = 1
    else:
        opt.gradient_accumulation_k_step = 1

    # ==================================================开始训练=======================================================
    ddp_logger.warning('start training! on local_rank:{}'.format(opt.local_rank))

    total_iters = 0                # the total number of training iterations
    source_test_pool = DataPool(pool_size, save_threshold)
    target_test_pool = DataPool(pool_size, save_threshold)
    synthetic_test_pool = DataPool(pool_size, save_threshold)

    for epoch in range(opt.epoch_start, opt.num_epochs + 1):
        ex_config.current_epoch = epoch
        if epoch == 1 and opt.continue_train is False and opt.DDP is True:
            ddp_logger.info('saving networks and than load!')
            # 保证每个进程的网络初始权重相同
            load_networks = model.load_for_apex if opt.APEX else model.load_networks
            base_patten = '%s_net_apex_%s.pth' if opt.APEX else '%s_net_%s.pth'
            weitht_name = base_patten % (epoch, opt.name)
            weitht_path = os.path.join(opt.checkpoints_dir, opt.name, weitht_name)
            with torch_distributed_zero_first(opt.local_rank):
                save_networks(epoch)
            load_networks(weitht_path)
            torch.distributed.barrier()

        epoch_start_time = time.time()

        # 更新dataloader的seed和优化器的学习率
        if not opt.serial_batches:
            dataloader.set_epoch(epoch)
        model.update_learning_rate(epoch)   # update learning rates in the beginning/ending of every epoch.
        model.zero_grad_optimizers()

        time_logger.info('Time epoch prepare: %d sec' % (time.time() - epoch_start_time))
        # ++++++++++++++++++++++++++++++++++++++++++++++训练一个epoch+++++++++++++++++++++++++++++++++++++++++++
        epoch_iter = 0
        iter_data_time = time.time()
        for batch_idx, data in enumerate(dataloader, 1):
            iter_start_time = time.time()

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            if opt.DEBUG and batch_idx == 1:
                # volume_name = os.path.basename(data['mr_volume_path'][0]).split('.')[0]
                # volume = data['mr_volume']
                # label = data['mr_label']
                # test_volume = volume[0, 0].clone().detach().cpu().numpy()
                # test_label = label[0, 0].clone().detach().cpu().numpy()
                # print('{:*^100}'.format('volume'))
                # print_numpy(test_volume, shp=True)
                # # print('{:*^100}'.format('label'))
                # # print_numpy(test_label, shp=False)
                # show_volume_label(test_volume, test_label, interval=2, add_line=True,
                #                   row=3, col=2, title=f'one {volume_name}')
                # # input("Press enter key to close this window")
                # # os.system("pause")
                # time.sleep(5)
                pass

            model.set_input(data)
            optimize_parameters(batch_idx % opt.gradient_accumulation_k_step == 0)
            # 计算当前训练数据的metrics、losses、lrs， 使用visualizer绘图并打印
            if total_iters % opt.print_freq == 0 or total_iters % opt.plot_freq == 0:
                # torch.cuda.synchronize()
                t_data = iter_start_time - iter_data_time
                t_comp = (time.time() - iter_start_time) / opt.batch_size

                model.compute_metrics()         # done reduce
                metrics = model.get_current_metrics()      # done reduce
                # print(str(metrics).replace('basic_metrics', 'tp fn tn fp'))

                losses = model.get_current_losses()   # done reduce

                if on_master:
                    lrs = model.get_current_lrs()   # 学习率不需要reduce

                    if total_iters % opt.print_freq == 0:
                        visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                        visualizer.print_current_metrics(metrics, epoch, epoch_iter)
                        visualizer.add_hparams({'lr': lrs[0]}, dict(metrics),
                                               name=f'result on epoch{epoch}', global_step=total_iters)

                    if total_iters % opt.plot_freq == 0:
                        for lr_i, lr in enumerate(lrs):
                            visualizer.plot_one_scalar(lr, total_iters, name=str(lr_i+1), tag='lrs')

                        visualizer.plot_current_losses(epoch, float(epoch_iter)/dataset_size, losses, total_iters)
                        # for key, value in losses.items():
                        #     visualizer.plot_one_scalar(value, total_iters, key)
                        for key, value in metrics.items():
                            if isinstance(value, tuple):
                                metrics.pop(key)
                        visualizer.plot_current_losses(epoch, float(epoch_iter)/dataset_size, metrics, total_iters,
                                                       tag='metrics over time')

            # 获取当前训练数据的预测结果，使用visualizer展示图片；依据iter保存checkpoint
            if on_master:
                if total_iters % opt.display_freq == 0:
                    # don't need to reduce
                    model.compute_visuals()
                    visuals = model.get_current_visuals()
                    # ['predict', 'label']
                    if opt.DEBUG and batch_idx == 1:
                        # volume_name = os.path.basename(data['mr_volume_path'][0]).split('.')[0]
                        # volume = visuals['source_volume']
                        # label = visuals['source_label']
                        # predict = visuals['source_predict']
                        # test_volume = volume[0, 0].clone().detach().cpu().numpy()
                        # test_label = label[0, 0].clone().detach().cpu().numpy()
                        # test_predict = predict[0, 0].clone().detach().cpu().numpy()
                        # print('{:*^100}'.format('volume'))
                        # print_numpy(test_volume, shp=False)
                        # print('{:*^100}'.format('label'))
                        # print_numpy(test_label, shp=False)
                        # print('{:*^100}'.format('predict'))
                        # print_numpy(test_predict, shp=False)
                        # show_volume_label_predict(volume, label, predict, interval=2, add_line=True,
                        #                           row=3, col=2, title=f'one {volume_name}')

                        # show_volume_label(test_volume, test_label, row=4, col=4, title=f'one {volume_name}')
                        # show_volume_label(test_label, test_predict, row=4, col=4, title=f'two {volume_name}')
                        # , fix_num=True, max_num=8, fig_list=fig_list
                        pass
                    if opt.display_histogram:
                        for name, image in visuals.items():
                            visualizer.add_histogram(name, image, total_iters)

                    if opt.save_visuals:
                        with torch_distributed_zero_first(opt.local_rank):
                            if opt.save_only_latest:
                                name = 'latest'
                                visualizer.save_visuals(visuals, name)
                            elif total_iters % opt.save_visuals_frep == 0:
                                # name = 'epoch-{}-epoch_iter-{}'.format(epoch, epoch_iter)
                                name = 'epoch-{}-epoch_iter-{}'.format(epoch, epoch_iter)
                                visualizer.save_visuals(visuals, name)

                    if opt.display_on_tensorboard:
                        # visuals_refine = {}
                        for name, image in visuals.items():
                            if image.ndim == 5:  # N C D H W
                                N, C, D, H, W = image.shape
                                for c in range(C):
                                    if opt.play_video:
                                        visualizer.play_current_video(torch.unsqueeze(image[:, c], dim=2),
                                                                      total_iters, tag=name+'video')
                                    for d in range(D):
                                        # visualizer.show_current_images_v2(name+f'N:{d} C{c}',image[:,c:c+1,d],total_iters)
                                        for n in range(N):
                                            visualizer.show_current_images({name+'train_N:{} C:{} D:{}'.format(n, c, d): image[n, c, d]}, total_iters)
                        #                     visuals_refine[name+'N:{} C:{} D:{}'.format(n, c, d)] = image[n, c, d]
                        #     else:
                        #         visuals_refine[name] = image
                        # visualizer.show_current_images(visuals_refine, total_iters)

                if total_iters > opt.save_iter_start and (total_iters-opt.save_iter_start) % opt.save_iter_freq == 0:
                    ddp_logger.warning('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    save_networks(save_suffix)

            iter_data_time = time.time()

        time_logger.info('Time all iteration on epoch Taken: %d sec' % (time.time() - epoch_start_time))
        # ++++++++++++++++++++++++++++++++++++用测试数据测试当前epoch的模型+++++++++++++++++++++++++++++++++++++++
        if opt.test_on_train:
            if epoch % opt.val_epoch_freq == 0 and epoch > 1:
                ddp_logger.info('Test start of epoch %d / %d \t' % (epoch, opt.num_epochs))

                test_start_time = time.time()
                if opt.eval:
                    model.eval()

                test_metrics_all = []
                for test_batch_idx, test_data in enumerate(test_dataloader):
                    # if test_batch_idx==3:
                    #     print(1)
                    model.set_input(test_data)
                    model.test()

                    test_metrics = model.get_current_metrics()
                    test_visuals = model.get_current_visuals()
                    test_metrics_all.append(test_metrics)

                    if on_master:
                        visualizer.print_current_test_metrics(test_metrics, epoch, test_batch_idx)
                        visualizer.plot_current_losses(0, 0, test_metrics, test_batch_idx,
                                                       tag='test metrics over paitent')
                        if opt.DEBUG and test_batch_idx == 1 and epoch == 1:
                            pass

                        if opt.save_visuals:
                            with torch_distributed_zero_first(opt.local_rank):
                                if opt.save_only_latest:
                                    name = 'latest-test'
                                    visualizer.save_visuals(test_visuals, name)
                                elif total_iters % opt.save_visuals_frep == 0:
                                    # name = 'epoch-{}-epoch_iter-{}'.format(epoch, epoch_iter)
                                    name = 'epoch-{}-epoch_iter-{}-test'.format(epoch, epoch_iter)
                                    visualizer.save_visuals(test_visuals, name)

                        if opt.display_on_tensorboard:
                            for name, image in test_visuals.items():
                                if image.ndim == 5:  # N C D H W
                                    N, C, D, H, W = image.shape
                                    for c in range(C):
                                        for d in range(D):
                                            for n in range(N):
                                                visualizer.show_current_images({name+'test_N:{} C:{} D:{}'.format(n, c, d): image[n, c, d]}, epoch, suffix=' on test')

                now_test_metrics = combine_metrics(test_metrics_all)

                save_for_source = source_test_pool.update(epoch, now_test_metrics['sourceDC'])
                save_for_target = target_test_pool.update(epoch, now_test_metrics['targetDC'])
                save_for_synthetic = synthetic_test_pool.update(epoch, (now_test_metrics['sourceDC'] +
                                                                        now_test_metrics['targetDC'])/2)

                if on_master:
                    visualizer.print_current_test_metrics(now_test_metrics, epoch, -1)
                    visualizer.plot_current_losses(0, 0, now_test_metrics, epoch, tag='test metrics over epoch')
                    if save_for_target or save_for_source or save_for_synthetic:
                        ddp_logger.warning('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                        save_networks('latest')
                        save_networks(epoch)
                        visualizer.add_text(opt.name, f'saving checkpoint on {epoch}', total_iters)

                model.train()
                ddp_logger.info('Test end of epoch %d / %d \t Time Taken: %d sec'
                                % (epoch, opt.num_epochs, time.time()-test_start_time))

        elif on_master and epoch > opt.save_epoch_start and (epoch-opt.save_epoch_start) % opt.save_epoch_freq == 0:
            # cache our model every <save_epoch_freq> epochs
            ddp_logger.warning('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            save_networks('latest')
            save_networks(epoch)
            visualizer.add_text(opt.name, f'saving checkpoint on {epoch}', total_iters)

        ddp_logger.info('End of epoch %d / %d \t Time Taken: %d sec' %
                        (epoch, opt.num_epochs, time.time() - epoch_start_time))

    if on_master:
        ddp_logger.warning('saving the model at the end of epoch %d, iters %d' % (opt.num_epochs, total_iters))
        save_networks('latest')
        save_networks(opt.num_epochs)
        visualizer.add_text(opt.name, f'saving checkpoint on {opt.num_epochs}', total_iters)

    ddp_logger.info('end training!')
    best_source_epoch, best_source_dice = source_test_pool.get_best_data()
    best_target_epoch, best_target_dice = target_test_pool.get_best_data()
    best_synthetic_epoch, best_synthetic_dice = synthetic_test_pool.get_best_data()
    opt_logger.info(f'best_source_epoch: {best_source_epoch}\n'
                    f'best_source_dice: {best_source_dice}\n'
                    f'best_target_epoch: {best_target_epoch}\n'
                    f'best_target_dice: {best_target_dice}\n'
                    f'best_synthetic_epoch: {best_synthetic_epoch}\n'
                    f'best_synthetic_dice: {best_synthetic_dice}\n'
                    )

    if visualizer:
        visualizer.record_test_metrics_message('source:')
        visualizer.record_test_metrics_message(source_test_pool.get_complete_data())
        visualizer.record_test_metrics_message('target:')
        visualizer.record_test_metrics_message(target_test_pool.get_complete_data())
        visualizer.record_test_metrics_message('synthetic:')
        visualizer.record_test_metrics_message(synthetic_test_pool.get_complete_data())
        visualizer.close()
    ddp_logger.info('visualizer closed!')

    if opt.DDP:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    return


def combine_metrics(metrics_list):
    # combine
    patient_num = len(metrics_list)
    metrics_names = metrics_list[0].keys()
    total_metrics = {}
    for key in metrics_names:
        value = 0
        try:
            for v in metrics_list:
                value += v[key]
        except TypeError as e:
            print('some worng of key :{} with value: {}'.format(key, metrics_list[0][key]))
            continue
        value /= patient_num
        total_metrics[key] = value

    return total_metrics


if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    train()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_reserved())
    print(torch.cuda.max_memory_allocated())




