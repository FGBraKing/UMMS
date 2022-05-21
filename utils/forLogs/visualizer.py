import os
import sys
import ntpath
import time
import torch
import h5py
import datetime
import numpy as np

from . import html
from . import log
from tensorboardX import SummaryWriter
from subprocess import Popen, PIPE
from data.transforms.transformOnTensor import tensor2im
from utils.others.img_io import save_image
from utils.others.utils import mkdirs

try:
    import visdom
    has_visdom = True
except ImportError:
    has_visdom = False


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256, isvolum=False):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    if isvolum:
        short_path = ntpath.basename(image_path[0][0])
    else:
        short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


# name\isTrain\logs_dir
# visdom_server\visdom_port\visdom_env\visdom_id\visdom_ncols\
# html_winsize\no_html
# save_log
# tensorboard
class Visualizer:
    """
    This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display,
    and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    support: logging tensorboardX visdom html
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        Step 5: create a tensorboardX object
        """
        # cache the option
        self.opt = opt
        self.name = opt.name
        self.device = torch.device('cuda:{}'.format(opt.local_gpu)) if opt.local_gpu >= 0 else torch.device('cpu')

        self.use_html = opt.isTrain and opt.with_html and not opt.DEBUG
        self.use_tensorboard = opt.isTrain and opt.with_tensorboard and not opt.DEBUG
        self.use_visdom = opt.isTrain and opt.with_visdom and opt.visdom_id > 0 and has_visdom and not opt.DEBUG

        # connect to a visdom server given <display_port> and <display_server>
        if self.use_visdom:
            assert opt.visdom_id > 0, "visdom_id have to greater than 0"
            self.visdom_server = opt.visdom_server
            self.visdom_port = opt.visdom_port
            self.visdom_env = opt.visdom_env

            self.visdom_id = opt.visdom_id
            self.ncols = opt.visdom_ncols

            self.vis = visdom.Visdom(server=opt.visdom_server, port=opt.visdom_port, env=opt.visdom_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
        if self.use_html:
            self.win_size = opt.html_winsize  # html used
            self.web_dir = os.path.join(opt.logs_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            mkdirs([self.web_dir, self.img_dir])

        # create create a tensorboardX object
        if self.use_tensorboard:
            tensor_dir = os.path.join(opt.logs_dir, opt.name, 'tensorboard_log')
            mkdirs(tensor_dir)
            self.writer = SummaryWriter(logdir=tensor_dir, flush_secs=120,
                                        filename_suffix=opt.name, write_to_disk=True)

        # create a logging file to store training losses
        self.visuals_dir = os.path.join(opt.logs_dir, opt.name, 'visuals')
        mkdirs(self.visuals_dir)
        self.log_name = os.path.join(opt.logs_dir, opt.name, 'message.txt')
        self.loss_log_name = os.path.join(opt.logs_dir, opt.name, 'loss.txt')
        self.metrics_log_name = os.path.join(opt.logs_dir, opt.name, 'metrics.txt')

        if opt.DEBUG:
            self.opt.save_log = False

        # =================================一共定义了5个logger

        self.message_logger = log.LOG(logname='train_message_log', is_save=self.opt.save_log,
                                      save_name=self.log_name, fmt='%(message)s')
        self.loss_msg_logger = log.LOG(logname='train_loss_log', is_save=self.opt.save_log,
                                       save_name=self.loss_log_name, fmt='%(message)s')
        self.metrics_msg_logger = log.LOG(logname='train__metrics_log', is_save=self.opt.save_log,
                                          save_name=self.metrics_log_name, fmt='%(message)s')

        self.loss_msg_logger(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.metrics_msg_logger(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.loss_msg_logger('{:#^50s}'.format('  Training Loss  '))
        self.metrics_msg_logger('{:#^50s}'.format('  Training metrics  '))

        if opt.test_on_train:
            test_log_name = os.path.join(opt.logs_dir, opt.name, 'test_metrics.txt')
            self.test_msg_logger = log.LOG(logname='test_log', is_save=self.opt.save_log,
                                           save_name=test_log_name, fmt='%(message)s')
            self.test_msg_logger(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            if not opt.DDP:
                slide_test_log_name = os.path.join(opt.logs_dir, opt.name, 'slide_test_metrics.txt')
                self.slide_test_msg_logger = log.LOG(logname='slide_test_log', is_save=self.opt.save_log,
                                                     save_name=slide_test_log_name, fmt='%(message)s')
                self.slide_test_msg_logger(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def create_visdom_connections(self):
        """
        If the program could not connect to Visdom server,
        this function will start a new server at port < self.port >
         """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.visdom_port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        # if self.display_id > 0:  # show images in the browser using visdom
        if self.use_visdom:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                image_numpy = None
                print('visuals len:', len(visuals))
                for label, image in visuals.items():
                    image_numpy = tensor2im(image)
                    print('shape:', image_numpy.shape)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.visdom_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.visdom_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and save_result:  # save images to an HTML file if they haven't been saved.
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image in visuals.items():
                    image_numpy = tensor2im(image)   # [-1,1] ->[0,255]
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def show_current_images(self, visuals, cur_iter, suffix=''):
        if self.use_visdom:
            idx = 1
            try:
                for label, image in visuals.items():
                    image_numpy = tensor2im(image)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.visdom_id + idx)
                    idx += 1
            except VisdomExceptionBase:
                self.create_visdom_connections()
        if self.use_tensorboard:
            for name, image in visuals.items():
                if image.ndim == 2:
                    image = torch.unsqueeze(image, dim=0)
                self.writer.add_image(tag=name+suffix, img_tensor=image, global_step=cur_iter)

    def show_current_images_v2(self, tag, img_tensor, global_step=None):
        if self.use_tensorboard:
            self.writer.add_images(tag, img_tensor, global_step)

    def play_current_video(self, tensor, videofile, total_iters=None, tag=''):
        """tensor: shape should be L*H*W*C"""
        if self.use_visdom > 0:
            if tensor and tensor.ndim == 4:
                self.vis.video(tensor=tensor, win=self.visdom_id,
                               env=self.visdom_env, opts={'fps': 25})
            elif os.path.isfile(videofile):
                self.vis.video(videofile=videofile, win=self.visdom_id,
                               env=self.visdom_env, opts={'fps': 25})
        if self.use_tensorboard:
            if tensor.ndim == 4:        # NDHW
                tensor = torch.unsqueeze(tensor, dim=2)
            self.writer.add_video(tag=tag, vid_tensor=tensor,
                                  global_step=total_iters, fps=1)

    # TODO：不知道display_id在实际显示中影响哪部分，等后续实践之后再改进
    def play_current_audio(self, tensor=None, audiofile=None, total_iters=None):
        """tensor: shape like N*2"""
        if self.use_visdom > 0:
            if tensor and tensor.ndim == 2:
                self.vis.audio(tensor=tensor, win=self.visdom_id,
                               env=self.visdom_env, opts={'sample_frequency': 44100})
            elif os.path.isfile(audiofile):
                self.vis.audio(audiofile=audiofile, win=self.visdom_id,
                               env=self.visdom_env, opts={'sample_frequency': 44100})
        if self.use_tensorboard:
            self.writer.add_audio(tag='audio'+self.name,
                                  snd_tensor=tensor,
                                  global_step=total_iters,
                                  sample_rate=44100)

    def draw_model_graph(self, model, shape, verbose=False):
        if self.use_tensorboard:
            # device = model.get
            dummy_input = torch.zeros(shape).to(self.device)
            self.writer.add_graph(model=model, input_to_model=[dummy_input], verbose=verbose)

    def plot_one_scalar(self, value, step, name, tag='over step'):
        if self.use_tensorboard:
            self.writer.add_scalar(tag, value, step, display_name=name)

    def plot_current_losses(self, epoch, counter_ratio, losses, total_iters=None, tag='loss over time'):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
            :param total_iters:
            :tag:
        """
        if self.use_visdom:
            if not hasattr(self, 'plot_data'):
                self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
            self.plot_data['X'].append(epoch + counter_ratio)
            self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
            try:
                self.vis.line(
                    X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                    Y=np.array(self.plot_data['Y']),
                    opts={
                        'title': self.name + ' loss over time',
                        'legend': self.plot_data['legend'],
                        'xlabel': 'epoch',
                        'ylabel': 'loss'},
                    win=self.visdom_id)
            except VisdomExceptionBase:
                self.create_visdom_connections()
            except AttributeError:
                pass
        if self.use_tensorboard:
            self.writer.add_scalars(main_tag=self.name + tag,
                                    tag_scalar_dict=losses,
                                    global_step=total_iters)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        # fmt = "%(filename)s|%(funcName)s %(levelname)s %(asctime)-15s %(threadName)s"
        # log.LOG(logname='train_log', is_save=self.opt.save_log, save_name=self.log_name, fmt=fmt)('None')
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        self.loss_msg_logger(message)

    def print_current_metrics(self, metrics, epoch, iters):
        message = '(epoch: %d, iters: %d) ' % (epoch, iters)
        for k, v in metrics.items():
            try:
                message += '%s: %.4f ' % (k, v)
            except TypeError as e:
                # print('some wrong happened %s' % e)
                message += '%s: %r ' % (k, list(v))
        self.metrics_msg_logger(message)

    def print_current_test_metrics(self, metrics, epoch, batch_id, prefix=''):
        message = prefix
        message += '(epoch: %d, test_batch_id: %d) ' % (epoch, batch_id)
        for k, v in metrics.items():
            try:
                message += '%s: %.4f ' % (k, v)
            except TypeError as e:
                # print('some wrong happened %s' % e)
                message += '%s: %r ' % (k, list(v))
        self.test_msg_logger(message)

    def print_current_slide_test_metrics(self, metrics, epoch, batch_id, prefix=''):
        message = prefix
        message += '(epoch: %d, slide_batch_id: %d) ' % (epoch, batch_id)
        for k, v in metrics.items():
            try:
                message += '%s: %.4f ' % (k, v)
            except TypeError as e:
                # print('some wrong happened %s' % e)
                message += '%s: %r ' % (k, list(v))
        self.slide_test_msg_logger(message)

    def record_metrics_message(self, message):
        self.metrics_msg_logger(repr(message))

    def record_loss_message(self, message):
        self.loss_msg_logger(repr(message))

    def record_test_metrics_message(self, message):
        self.test_msg_logger(repr(message))

    def record_slide_test_metrics_message(self, message):
        self.slide_test_msg_logger(repr(message))

    def write_text(self, text):
        assert isinstance(text, str)
        self.message_logger(text)

    def add_hparams(self, hparam_dict=None, metric_dict=None, name=None, global_step=None):
        if self.use_tensorboard:
            self.writer.add_hparams(hparam_dict, metric_dict, name, global_step)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow'):
        if self.use_tensorboard:
            self.writer.add_histogram(tag, values, global_step, bins)

    def show_image_with_boxes(self, tag, img_tensor, box_tensor, global_step=None):
        if self.use_tensorboard:
            self.writer.add_image_with_boxes(tag, img_tensor, box_tensor, global_step)

    def add_text(self, tag, text_string, global_step=None):
        if self.use_tensorboard:
            self.writer.add_text(tag, text_string, global_step)

    def add_mesh(self, tag, vertices, colors=None, faces=None, config_dict=None, global_step=None):
        if self.use_tensorboard:
            self.writer.add_mesh(tag, vertices, colors, faces, config_dict, global_step)

    def add_pr_curve_raw(self, tag, true_positive_counts,
                         false_positive_counts,
                         true_negative_counts,
                         false_negative_counts,
                         precision,
                         recall,
                         global_step=None,
                         num_thresholds=127):
        if self.use_tensorboard:
            self.writer.add_pr_curve_raw(tag, true_positive_counts, false_positive_counts, true_negative_counts,
                                         false_negative_counts, precision, recall, global_step, num_thresholds)

    def save_visuals(self, visuals, name):
        save_name = os.path.join(self.visuals_dir, name+'visuals.h5')
        with h5py.File(save_name, mode='w') as fw:
            for name, image in visuals.items():
                if isinstance(image, np.ndarray):
                    fw.create_dataset(name=name, data=image)
                else:
                    fw.create_dataset(name=name, data=image.clone().detach().cpu().numpy())

    def close(self):
        if self.use_tensorboard:
            self.writer.close()
        if self.use_visdom:
            self.vis.save(self.opt.visdom_env)
            self.vis.close()


# add_image_with_boxes add_figure
