import torch
import logging
import numpy as np
import torch.nn.functional as F

from medpy import metric
from skimage import measure
from sklearn.metrics import confusion_matrix
from medpy.metric.binary import __surface_distances


# TODO:在求metrics时先判断prerdict和target在全0、全1时候的判断
# [N *]  [*]
class BinaryMetrics:
    '''
    addition parameter:
    voxelspacing
    connectivity
    '''
    def __init__(self, threshold=0.5, eps=1e-6):
        self.threshold = threshold
        self.eps = eps

    def get_basic_metrics(self, predict, target, **kwargs):
        assert isinstance(predict, np.ndarray), 'prediction should be numpy.ndarray, but got{}'.format(type(predict))
        assert isinstance(target, np.ndarray), 'target should be numpy.ndarray, but got{}'.format(type(target))
        assert predict.shape == target.shape, "Shape mismatch: {} and {}".format(predict.shape, target.shape)  # N *
        if 'mode' in kwargs.keys():
            mode = kwargs['mode']
        else:
            mode = 1

        SR = predict > self.threshold
        GT = target > self.threshold
        # GT = target.astype(np.bool)
        # if 1:
        #     from .utils import print_numpy
        #     print_numpy(predict)
        #     print_numpy(target)
        if mode:
            TP = ((SR == 1) & (GT == 1))
            FN = ((SR == 0) & (GT == 1))
            TN = ((SR == 0) & (GT == 0))
            FP = ((SR == 1) & (GT == 0))
            return np.sum(TP), np.sum(FN), np.sum(TN), np.sum(FP)
        else:
            tp = np.count_nonzero(SR & GT)
            tn = np.count_nonzero(~SR & ~GT)
            fp = np.count_nonzero(SR & ~GT)
            fn = np.count_nonzero(~SR & GT)
            # if tp==0:
            #     if predict.ndim==2:
            #         from .img_io import show_paired_image
            #         show_paired_image(predict, target, title1='predict', title2='target')
            #     if predict.ndim==3:
            #         from .img_io import show_volume_label
            #         show_volume_label(predict, target, title='predict_target')
            return tp, fn, tn, fp

    @staticmethod
    def get_size(predict, target, **kwargs):
        assert isinstance(predict, np.ndarray), 'prediction should be numpy.ndarray, but got{}'.format(type(predict))
        assert isinstance(target, np.ndarray), 'target should be numpy.ndarray, but got{}'.format(type(target))
        assert predict.shape == target.shape, "Shape mismatch: {} and {}".format(predict.shape, target.shape)  # N *
        return int(np.prod(target.shape, dtype=np.int64))

    def get_existence(self, predict, target, **kwargs):
        predict = predict > self.threshold
        target = target.astype(np.bool)
        test_empty = not np.any(predict)
        test_full = np.all(predict)
        reference_empty = not np.any(target)
        reference_full = np.all(target)
        return test_empty, test_full, reference_empty, reference_full

    def get_accuracy(self, SR, GT, **kwargs):
        TP, FN, TN, FP = self.get_basic_metrics(SR, GT)
        return float(TP + TN) / (float(TP + TN + FN + FP) + self.eps)

    def get_recall(self, predict, target, **kwargs):
        predict = predict > self.threshold
        target = target.astype(np.bool)
        return metric.recall(predict, target)

    def get_sensitivity(self, SR, GT, **kwargs):
        TP, FN, TN, FP = self.get_basic_metrics(SR, GT)
        return float(TP) / (float(TP + FN) + self.eps)

    def get_specificity(self, SR, GT, **kwargs):
        TP, FN, TN, FP = self.get_basic_metrics(SR, GT)
        return float(TN) / (float(TN + FP) + self.eps)

    def get_specificity1(self, predict, target, **kwargs):
        predict = predict > self.threshold
        target = target.astype(np.bool)
        return metric.specificity(predict, target)

    def get_false_discovery_rate(self, predict, target, **kwargs):
        """FP / (TP + FP)"""
        return 1 - self.get_precision(predict, target, **kwargs)

    def get_false_omission_rate(self, predict, target, **kwargs):
        TP, FN, TN, FP = self.get_basic_metrics(predict, target)
        return float(FN) / (float(FN + TN) + self.eps)

    def get_false_positive_rate(self, predict, target, **kwargs):
        predict = predict > self.threshold
        target = target.astype(np.bool)
        return 1 - self.get_specificity1(predict, target, **kwargs)

    def get_false_negative_rate(self, predict, target, **kwargs):
        predict = predict > self.threshold
        target = target.astype(np.bool)
        return 1 - self.get_sensitivity(predict, target, **kwargs)

    def get_true_negative_rate(self, predict, target, **kwargs):
        predict = predict > self.threshold
        target = target.astype(np.bool)
        return metric.true_negative_rate(predict, target)

    def get_true_positive_rate(self, predict, target, **kwargs):
        predict = predict > self.threshold
        target = target.astype(np.bool)
        return metric.true_positive_rate(predict, target)

    def get_positive_predictive_value(self, predict, target, **kwargs):
        predict = predict > self.threshold
        target = target.astype(np.bool)
        return metric.positive_predictive_value(predict, target)

    def get_hd(self, predict, target, **kwargs):
        if 'voxelspacing' in kwargs.keys():
            voxelspacing = kwargs['voxelspacing']
        else:
            voxelspacing = None
        if 'connectivity' in kwargs.keys():
            connectivity = kwargs['connectivity']
        else:
            connectivity = 1
        predict = predict > self.threshold
        target = target.astype(np.bool)
        return metric.hd(predict, target, voxelspacing, connectivity)

    def get_hd95(self, predict, target, **kwargs):
        if 'voxelspacing' in kwargs.keys():
            voxelspacing = kwargs['voxelspacing']
        else:
            voxelspacing = None
        if 'connectivity' in kwargs.keys():
            connectivity = kwargs['connectivity']
        else:
            connectivity = 1
        predict = predict > self.threshold
        target = target.astype(np.bool)
        return metric.hd95(predict, target, voxelspacing, connectivity)

    def get_assd(self, predict, target, **kwargs):
        if 'voxelspacing' in kwargs.keys():
            voxelspacing = kwargs['voxelspacing']
        else:
            voxelspacing = None
        if 'connectivity' in kwargs.keys():
            connectivity = kwargs['connectivity']
        else:
            connectivity = 1
        predict = predict > self.threshold
        target = target.astype(np.bool)
        return metric.assd(predict, target, voxelspacing, connectivity)

    def get_asd(self, predict, target, **kwargs):
        if 'voxelspacing' in kwargs.keys():
            voxelspacing = kwargs['voxelspacing']
        else:
            voxelspacing = None
        if 'connectivity' in kwargs.keys():
            connectivity = kwargs['connectivity']
        else:
            connectivity = 1
        predict = predict > self.threshold
        target = target.astype(np.bool)
        return metric.asd(predict, target, voxelspacing, connectivity)

    def get_ravd(self, predict, target, **kwargs):
        predict = predict > self.threshold
        target = target.astype(np.bool)
        return metric.ravd(predict, target)

    def get_volume_correlation(self, predict, target, **kwargs):
        predict = predict > self.threshold
        target = target.astype(np.bool)
        return metric.volume_correlation(predict, target)

    def get_volume_change_correlation(self, predict, target, **kwargs):
        predict = predict > self.threshold
        target = target.astype(np.bool)
        return metric.volume_change_correlation(predict, target)

    def get_obj_assd(self, predict, target, **kwargs):
        if 'voxelspacing' in kwargs.keys():
            voxelspacing = kwargs['voxelspacing']
        else:
            voxelspacing = None
        if 'connectivity' in kwargs.keys():
            connectivity = kwargs['connectivity']
        else:
            connectivity = 1
        predict = predict > self.threshold
        target = target.astype(np.bool)
        return metric.obj_assd(predict, target, voxelspacing, connectivity)

    def get_obj_asd(self, predict, target, **kwargs):
        if 'voxelspacing' in kwargs.keys():
            voxelspacing = kwargs['voxelspacing']
        else:
            voxelspacing = None
        if 'connectivity' in kwargs.keys():
            connectivity = kwargs['connectivity']
        else:
            connectivity = 1
        predict = predict > self.threshold
        target = target.astype(np.bool)
        return metric.obj_asd(predict, target, voxelspacing, connectivity)

    def get_precision(self, SR, GT, **kwargs):
        TP, FN, TN, FP = self.get_basic_metrics(SR, GT)
        return float(TP) / (float(TP + FP) + self.eps)

    def get_precision1(self, predict, target, **kwargs):
        predict = predict > self.threshold
        target = target.astype(np.bool)
        return metric.precision(predict, target)

    def get_DC(self, SR, GT, **kwargs):
        TP, FN, TN, FP = self.get_basic_metrics(SR, GT)
        return float(2*TP) / (float(2*TP + FP + FN) + self.eps)

    def get_dice(self, SR, GT, **kwargs):
        SR = SR > self.threshold
        GT = GT.astype(np.bool)

        Inter = np.sum((SR & GT))
        return float(2*Inter)/(float(np.sum(SR)+np.sum(GT)) + self.eps)

    def get_DICE(self, SR, GT, **kwargs):
        SR = SR > self.threshold
        GT = GT.astype(np.bool)
        return metric.dc(SR, GT)

    def get_F1(self, SR, GT, **kwargs):
        SE = self.get_sensitivity(SR, GT)
        PC = self.get_precision(SR, GT)

        return 2*SE*PC/(SE+PC + self.eps)

    def get_jc(self, predict, target, **kwargs):
        predict = predict > self.threshold
        target = target.astype(np.bool)
        return metric.jc(predict, target)

    def get_JS1(self, SR, GT, **kwargs):
        SR = SR > self.threshold
        GT = GT.astype(np.bool)

        Inter = np.sum((SR & GT))
        Union = np.sum((SR | GT))

        return float(Inter)/(float(Union) + self.eps)

    def get_jaccard(self, SR, GT, **kwargs):
        return self.get_JS1(SR, GT, **kwargs)

    def get_JS(self, SR, GT, **kwargs):
        TP, FN, TN, FP = self.get_basic_metrics(SR, GT)
        return float(TP) / (float(TP + FP + FN) + self.eps)

    def get_IOU(self, SR, GT, **kwargs):
        TP, FN, TN, FP = self.get_basic_metrics(SR, GT)
        return float(TP) / (float(TP + FP + FN) + self.eps)

    def __call__(self, SR, GT, *args, **kwargs):
        metrices = []
        for arg in args:
            if isinstance(arg, str):
                try:
                    get_metrice = getattr(self, 'get_'+arg)
                    metrices.append(get_metrice(SR, GT, **kwargs))
                except AttributeError as e:
                    print('warning:', e)
                    print('there is no attribute whose name is:{}'.format(arg))
                    metrices.append(None)
        return metrices

    def set_threshold(self, threshold=0.5):
        self.threshold = threshold

    def set_eps(self, eps=1e-6):
        self.eps = eps


# [N C *]
class MutiClassMetrics:
    def __init__(self, threshold=0.5, eps=1e-6):
        self.get_bin_metrics = BinaryMetrics(threshold, eps)

    def __call__(self, predict, actual, *args, **kwargs):
        '''
        :param predict:numpy.array,the shape is NC or NC*
        :param actual:  not one-hot
        :param args:
        :param kwargs:
        class_num
        is_logit
        ignore_index
        reduce
        :return:
        '''
        assert isinstance(predict, np.ndarray), 'predict must be the type of numpy.ndarray'
        data_num = predict.shape[0]
        actual = actual.astype(np.int)
        if predict.ndim == 1:
            return self.get_bin_metrics(predict, actual, *args, **kwargs)
        else:
            class_num = predict.shape[1]
            if class_num == 1:
                return self.get_bin_metrics(predict, actual, *args, **kwargs)
        if predict.ndim == 2:
            predict = predict.reshape(predict.shape+(1,))
            if actual.ndim == 1:
                actual = actual.reshape(actual.shape+(1, 1))
            elif actual.ndim == 2:
                actual = actual.reshape(actual.shape+(1,))
        else:
            predict = np.reshape(predict, (predict.shape[0], class_num, -1))
            actual = np.reshape(actual, (actual.shape[0], 1, -1))
        # ---------------------------------------------predict: N*C*M
        if 'class_num' in kwargs.keys():
            class_num = kwargs['class_num']
        # ---------------------------------------------get the ont-hot predict
        if 'is_logit' in kwargs.keys():
            is_logit = kwargs['is_logit']
            if is_logit:
                max_index = np.argmax(predict, axis=1)
                # bin_predict = np.zeros_like(predict, dtype=np.float).scat
                predict_tmp = np.zeros(shape=(predict.shape[0], class_num, predict.shape[2]), dtype=np.float)
                for i in range(predict_tmp.shape[0]):
                    for k in range(predict_tmp.shape[2]):
                        predict_tmp[i][max_index[i][k]][k] = 1
                # print(predict_tmp)
                predict = predict_tmp
        # ---------------------------------------------get the ont-hot actual
        actual_tmp = np.zeros_like(predict, dtype=np.float)
        for i in range(actual_tmp.shape[0]):
            for k in range(actual_tmp.shape[2]):
                actual_tmp[i][actual[i][0]][k] = 1
        # print(actual_tmp) # N C M
        actual = actual_tmp

        ignore_index = []
        if 'ignore_index' in kwargs.keys():
            ignore_index = kwargs['ignore_index']
            if isinstance(ignore_index, (int, float)):
                ignore_index = [int(ignore_index)]
            elif ignore_index is None:
                ignore_index = []
            elif isinstance(ignore_index, (list, tuple)):
                ignore_index = ignore_index
            else:
                print("Expect 'int|float|list|tuple', while get '{}'".format(type(ignore_index)))

        # ------------------------------------------------get the string of metrics
        metrics = []
        for arg in args:
            if isinstance(arg, str):
                metrics.append(arg)
        metrics = tuple(metrics)

        result_list = []
        for c in range(class_num):
            if 'ignore_index' in kwargs.keys() and c in ignore_index:
                continue
            c_actual = actual[:, c, :]
            c_predict = predict[:, c, :]
            c_result_list = self.get_bin_metrics(c_predict, c_actual, *metrics, **kwargs)
            c_result_dict = dict(zip(metrics, c_result_list))
            result_list.append(c_result_dict)
        if 'reduce' in kwargs.keys():
            reduce = kwargs['reduce']
            result_dict = dict()
            for m_metrics in metrics:
                m_metrics_list = []
                for c_result_dict in result_list:
                    m_metrics_list.append(c_result_dict[m_metrics])
                result_dict[m_metrics] = m_metrics_list
            if reduce == 'mean':
                for key, value in result_dict.items():
                    result_dict[key] = np.mean(value)
                return result_dict
            if reduce == 'sum':
                for key, value in result_dict.items():
                    result_dict[key] = np.sum(value)
                return result_dict
            else:
                return result_dict
        return result_list


# [*], tensor
class SoftMetrics:
    ''' basic metrics, only apply to sample set. used for training'''
    def __init__(self, smooth=0., eps=1e-6):
        self.smooth = smooth
        self.eps = eps

    @staticmethod
    def get_basic_metrics(result, target):
        assert isinstance(result, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert result.numel() == target.numel()
        assert result.max() <= 1 and result.min() >= 0
        assert target.max() <= 1 and target.min() >= 0

        # number = result.numel()
        f_result = result.contiguous().view(-1).float()  # float32
        f_target = target.contiguous().view(-1).float()

        # TP = torch.matmul(f_result, f_target)
        TP = torch.dot(f_result, f_target)
        TN = torch.dot(1-f_result, 1-f_target)
        FP = torch.dot(f_result, 1-f_target)
        FN = torch.dot(1-f_result, f_target)
        return TP, FN, TN, FP

    def get_hd(self, predict, target, **kwargs):
        if 'voxelspacing' in kwargs.keys():
            voxelspacing = kwargs['voxelspacing']
        else:
            voxelspacing = None
        if 'connectivity' in kwargs.keys():
            connectivity = kwargs['connectivity']
        else:
            connectivity = 1
        device = predict.device
        result = predict.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        result = result > 0.5
        target = target > 0.5
        if voxelspacing is not None and len(voxelspacing) > 1:
            true_dim = len(voxelspacing)
            true_shape = predict.shape[-true_dim:]

            result = np.reshape(result, [-1]+list(true_shape))
            target = np.reshape(target, [-1]+list(true_shape))
            out = []
            for i in range(predict.shape[0]):
                out.append(metric.hd(result[i], target[i], voxelspacing, connectivity))
            out = np.mean(out)
        else:
            out = metric.hd(result, target, voxelspacing, connectivity)

        return torch.tensor(out, requires_grad=False).to(device)

    def get_hd95(self, predict, target, **kwargs):
        if 'voxelspacing' in kwargs.keys():
            voxelspacing = kwargs['voxelspacing']
        else:
            voxelspacing = None
        if 'connectivity' in kwargs.keys():
            connectivity = kwargs['connectivity']
        else:
            connectivity = 1
        device = predict.device
        result = predict.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        result = result > 0.5
        target = target > 0.5
        if voxelspacing is not None and len(voxelspacing) > 1:
            true_dim = len(voxelspacing)
            true_shape = predict.shape[-true_dim:]

            result = np.reshape(result, [-1]+list(true_shape))
            target = np.reshape(target, [-1]+list(true_shape))
            out = []
            for i in range(predict.shape[0]):
                out.append(metric.hd95(result[i], target[i], voxelspacing, connectivity))
            out = np.mean(out)
        else:
            out = metric.hd95(result, target, voxelspacing, connectivity)
        return torch.tensor(out, requires_grad=False).to(device)

    def get_assd(self, predict, target, **kwargs):
        if 'voxelspacing' in kwargs.keys():
            voxelspacing = kwargs['voxelspacing']
        else:
            voxelspacing = None
        if 'connectivity' in kwargs.keys():
            connectivity = kwargs['connectivity']
        else:
            connectivity = 1

        device = predict.device
        result = predict.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        result = result > 0.5
        target = target > 0.5
        if voxelspacing is not None and len(voxelspacing) > 1:
            true_dim = len(voxelspacing)
            true_shape = predict.shape[-true_dim:]
            # now_dim = len(predict.shape)
            # used_dim = list(range(now_dim-true_dim, now_dim))

            result = np.reshape(result, [-1]+list(true_shape))
            target = np.reshape(target, [-1]+list(true_shape))
            out = []
            for i in range(predict.shape[0]):
                out.append(metric.assd(result[i], target[i], voxelspacing, connectivity))
            out = np.mean(out)
        else:
            out = metric.assd(result, target, voxelspacing, connectivity)
        return torch.tensor(out, requires_grad=False).to(device)

    def get_asd(self, predict, target, **kwargs):
        if 'voxelspacing' in kwargs.keys():
            voxelspacing = kwargs['voxelspacing']
        else:
            voxelspacing = None
        if 'connectivity' in kwargs.keys():
            connectivity = kwargs['connectivity']
        else:
            connectivity = 1

        device = predict.device
        result = predict.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        result = result > 0.5
        target = target > 0.5
        if voxelspacing is not None and len(voxelspacing) > 1:
            true_dim = len(voxelspacing)
            true_shape = predict.shape[-true_dim:]

            result = np.reshape(result, [-1]+list(true_shape))
            target = np.reshape(target, [-1]+list(true_shape))
            out = []
            for i in range(predict.shape[0]):
                out.append(metric.asd(result[i], target[i], voxelspacing, connectivity))
            out = np.mean(out)
        else:
            out = metric.asd(result, target, voxelspacing, connectivity)
        return torch.tensor(out, requires_grad=False).to(device)

    def get_ravd(self, result, target, **kwargs):

        result = (result > 0.5).float()
        target = (target > 0.5).float()

        return (result.sum()-target.sum())/target.sum()

    def get_dice(self, result, target, **kwargs):
        TP, FN, TN, FP = self.get_basic_metrics(result, target)
        return (2 * TP + self.smooth) / (2*TP + FP + FN + self.smooth + self.eps)

    def get_DICE(self, result, target, **kwargs):
        f_result = result.contiguous().view(-1).float()  # float32
        f_target = target.contiguous().view(-1).float()
        inter = torch.dot(f_result, f_target)
        denominator = torch.sum(f_target) + torch.sum(f_result)
        return (2 * inter + self.smooth) / (denominator + self.smooth + self.eps)

    def get_DC(self, result, target, **kwargs):
        TP, FN, TN, FP = self.get_basic_metrics(result, target)
        return (2 * TP + self.smooth) / (2 * TP + FP + FN + self.smooth + self.eps)

    def get_IOU(self, result, target, **kwargs):
        TP, FN, TN, FP = self.get_basic_metrics(result, target)
        return (TP + self.smooth) / (TP + FP + FN + self.smooth + self.eps)

    def get_jaccard(self, result, target, **kwargs):
        TP, FN, TN, FP = self.get_basic_metrics(result, target)
        return (TP + self.smooth) / (TP + FP + FN + self.smooth + self.eps)

    def get_accuracy(self, result, target, **kwargs):
        TP, FN, TN, FP = self.get_basic_metrics(result, target)
        return (TP + TN + self.smooth) / (TP + TN + FP + FN + self.smooth + self.eps)

    def get_recall(self, result, target, **kwargs):
        TP, FN, TN, FP = self.get_basic_metrics(result, target)
        return (TP + self.smooth) / (TP + FN + self.smooth + self.eps)

    def get_specificity(self, result, target, **kwargs):
        TP, FN, TN, FP = self.get_basic_metrics(result, target)
        return (TN + self.smooth) / (TN + FP + self.smooth + self.eps)

    def get_precision(self, result, target, **kwargs):
        TP, FN, TN, FP = self.get_basic_metrics(result, target)
        return (TP + self.smooth) / (TP + FP + self.smooth + self.eps)

    def get_roisize(self, result, target, **kwargs):
        return target.sum().item() / target.numel()

    def __call__(self, result, target, *args, **kwargs):
        metrices = []
        for arg in args:
            if isinstance(arg, str):
                try:
                    get_metrice = getattr(self, 'get_'+arg)
                    metrices.append(get_metrice(result, target, **kwargs))
                except AttributeError as e:
                    print('warning:', e)
                    print('there is no attribute whose name is:{}'.format(arg))
                    metrices.append(None)
        return metrices

    def set_smooth(self, smooth=0.5):
        self.smooth = smooth

    def set_eps(self, eps=1e-6):
        self.eps = eps


# -----------------------------------From CHAOS-----------------------------------

class AveragePrecision:
    """
    Computes Average Precision given boundary prediction and ground truth instance segmentation.
    """

    def __init__(self, threshold=0.4, iou_range=(0.5, 1.0), ignore_index=0, min_instance_size=None):
        """
        :param threshold: probability value at which the input is going to be thresholded
        :param iou_range: compute ROC curve for the the range of IoU values: range(min,max,0.05)
        :param ignore_index: label to be ignored during computation
        :param min_instance_size: minimum size of the predicted instances to be considered
        """
        self.threshold = threshold
        self.iou_range = iou_range
        self.ignore_index = ignore_index
        self.min_instance_size = min_instance_size

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch float tensor (NxCxDxHxW) / or 4D numpy.ndarray
        :param target: 4D ground truth instance segmentation torch long tensor (NxDxHxW) / or 3D numpy.ndarray
        :return: highest average precision among channels
        """
        if isinstance(input, torch.Tensor):
            assert input.dim() == 5
            # convert to numpy array
            input = input[0].cpu().numpy()  # 4D
        if isinstance(target, torch.Tensor):
            assert target.dim() == 4
            # convert to numpy array
            target = target[0].cpu().numpy()  # 3D
        if isinstance(input, np.ndarray):
            assert input.ndim == 4
        if isinstance(target, np.ndarray):
            assert target.ndim == 3

        # get ground truth label set and discard 'ignore_index'
        target_instances = set(np.unique(target))
        target_instances.discard(self.ignore_index)

        per_channel_ap = []
        n_channels = input.shape[0]
        for c in range(n_channels):
            predictions = input[c]
            # threshold probability maps
            predictions = predictions > self.threshold
            # for connected component analysis we need to treat boundary signal as background
            # assign 0-label to boundary mask
            predictions = np.logical_not(predictions).astype(np.uint8)
            # run connected components on the predicted mask; consider only 1-connectivity
            predicted = measure.label(predictions, background=0, connectivity=1)
            ap = self._calculate_average_precision(predicted, target, target_instances)
            per_channel_ap.append(ap)

        # get maximum average precision across channels
        max_ap, c_index = np.max(per_channel_ap), np.argmax(per_channel_ap)
        # LOGGER.info(f'Max average precision: {max_ap}, channel: {c_index}')
        return max_ap

    def _calculate_average_precision(self, predicted, target, target_instances):
        recall, precision = self._roc_curve(predicted, target, target_instances)
        recall.insert(0, 0.0)  # insert 0.0 at beginning of list
        recall.append(1.0)  # insert 1.0 at end of list
        precision.insert(0, 0.0)  # insert 0.0 at beginning of list
        precision.append(0.0)  # insert 0.0 at end of list
        # make the precision(recall) piece-wise constant and monotonically decreasing
        # by iterating backwards starting from the last precision value (0.0)
        # see: https://www.jeremyjordan.me/evaluating-image-segmentation-models/ e.g.
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        # compute the area under precision recall curve by simple integration of piece-wise constant function
        ap = 0.0
        for i in range(1, len(recall)):
            ap += ((recall[i] - recall[i - 1]) * precision[i])
        return ap

    def _roc_curve(self, predicted, target, target_instances):
        ROC = []
        predicted, predicted_instances = self._filter_instances(predicted)

        # compute precision/recall curve points for various IoU values from a given range
        for min_iou in np.arange(self.iou_range[0], self.iou_range[1], 0.1):
            # initialize false negatives set
            false_negatives = set(target_instances)
            # initialize false positives set
            false_positives = set(predicted_instances)
            # initialize true positives set
            true_positives = set()

            for pred_label in predicted_instances:
                target_label = self._find_overlapping_target(pred_label, predicted, target, min_iou)
                if target_label is not None:
                    # update TP, FP and FN
                    if target_label == self.ignore_index:
                        # ignore if 'ignore_index' is the biggest overlapping
                        false_positives.discard(pred_label)
                    else:
                        true_positives.add(pred_label)
                        false_positives.discard(pred_label)
                        false_negatives.discard(target_label)

            tp = len(true_positives)
            fp = len(false_positives)
            fn = len(false_negatives)

            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            ROC.append((recall, precision))

        # sort points by recall
        ROC = np.array(sorted(ROC, key=lambda t: t[0]))
        # return recall and precision values
        return list(ROC[:, 0]), list(ROC[:, 1])

    def _find_overlapping_target(self, predicted_label, predicted, target, min_iou):
        """
        Return ground truth label which overlaps by at least 'min_iou' with a given input label 'p_label'
        or None if such ground truth label does not exist.
        """
        mask_predicted = predicted == predicted_label
        overlapping_labels = target[mask_predicted]
        labels, counts = np.unique(overlapping_labels, return_counts=True)
        # retrieve the biggest overlapping label
        target_label_ind = np.argmax(counts)
        target_label = labels[target_label_ind]
        # return target label if IoU greater than 'min_iou'; since we're starting from 0.5 IoU there might be
        # only one target label that fulfill this criterion
        mask_target = target == target_label
        # return target_label if IoU > min_iou
        if self._iou(mask_predicted, mask_target) > min_iou:
            return target_label
        return None

    @staticmethod
    def _iou(prediction, target):
        """
        Computes intersection over union
        """
        intersection = np.logical_and(prediction, target)
        union = np.logical_or(prediction, target)
        return np.sum(intersection) / np.sum(union)

    def _filter_instances(self, predicted):
        """
        Filters instances smaller than 'min_instance_size' by overriding them with 'ignore_index'
        :param predicted: input instance segmentation
        :return: tuple: (instance segmentation with small instances filtered, set of unique labels without the 'ignore_index')
        """
        if self.min_instance_size is not None:
            labels, counts = np.unique(predicted, return_counts=True)
            for label, count in zip(labels, counts):
                if count < self.min_instance_size:
                    mask = predicted == label
                    predicted[mask] = self.ignore_index

        labels = set(np.unique(predicted))
        labels.discard(self.ignore_index)
        return predicted, labels


# ----------------------------------From  nnunet--------------------------------
def normalized_surface_dice(a: np.ndarray, b: np.ndarray, threshold: float, spacing: tuple = None, connectivity=1):
    """
    This implementation differs from the official surface dice implementation! These two are not comparable!!!!!

    The normalized surface dice is symmetric, so it should not matter whether a or b is the reference image

    This implementation natively supports 2D and 3D images. Whether other dimensions are supported depends on the
    __surface_distances implementation in medpy

    :param a: image 1, must have the same shape as b
    :param b: image 2, must have the same shape as a
    :param threshold: distances below this threshold will be counted as true positives. Threshold is in mm, not voxels!
    (if spacing = (1, 1(, 1)) then one voxel=1mm so the threshold is effectively in voxels)
    must be a tuple of len dimension(a)
    :param spacing: how many mm is one voxel in reality? Can be left at None, we then assume an isotropic spacing of 1mm
    :param connectivity: see scipy.ndimage.generate_binary_structure for more information. I suggest you leave that
    one alone
    :return:
    """
    assert all([i == j for i, j in zip(a.shape, b.shape)]), "a and b must have the same shape. a.shape= %s, " \
                                                            "b.shape= %s" % (str(a.shape), str(b.shape))
    if spacing is None:
        spacing = tuple([1 for _ in range(len(a.shape))])
    a_to_b = __surface_distances(a, b, spacing, connectivity)
    b_to_a = __surface_distances(b, a, spacing, connectivity)

    numel_a = len(a_to_b)
    numel_b = len(b_to_a)

    tp_a = np.sum(a_to_b <= threshold) / numel_a
    tp_b = np.sum(b_to_a <= threshold) / numel_b

    fp = np.sum(a_to_b > threshold) / numel_a
    fn = np.sum(b_to_a > threshold) / numel_b

    dc = (tp_a + tp_b) / (tp_a + tp_b + fp + fn + 1e-8)  # 1e-8 just so that we don't get div by 0
    return dc
















