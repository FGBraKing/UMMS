from models.test_networks.test_base import TestGeneric
from models.modules.MultimodalSegmentation.DualStream import DualStreamUnetV1, DualStreamUnetV2, DualStreamUnetV3, DualStreamUnetV4, SingleUnet


def define_model(opt, device, domains=None):
    if opt.network_type == "V1":
        net = DualStreamUnetV1(in_channels=opt.input_nc,
                               out_channels=opt.output_nc,
                               domains=domains,
                               f_maps=opt.init_channel_number,
                               num_levels=5,
                               with_activation=True,
                               final_sigmoid=True,
                               interpolation=opt.up_interpolate,
                               norm_type="batch",
                               act_type="lrelu").to(device)
    elif opt.network_type == "V2":
        net = DualStreamUnetV2(in_channels=opt.input_nc,
                               out_channels=opt.output_nc,
                               domains=domains,
                               f_maps=opt.init_channel_number,
                               num_levels=5,
                               with_activation=True,
                               final_sigmoid=True,
                               interpolation=True,
                               norm_type="batch",
                               act_type="lrelu").to(device)
    elif opt.network_type == "V3":
        net = DualStreamUnetV3(in_channels=opt.input_nc,
                               out_channels=opt.output_nc,
                               domains=domains,
                               f_maps=opt.init_channel_number,
                               num_levels=5,
                               with_activation=True,
                               final_sigmoid=True,
                               interpolation=True,
                               norm_type="batch",
                               act_type="lrelu").to(device)
    elif opt.network_type == "V4":
        net = DualStreamUnetV4(in_channels=opt.input_nc,
                               out_channels=opt.output_nc,
                               domains=domains,
                               f_maps=opt.init_channel_number,
                               num_levels=5,
                               with_activation=True,
                               final_sigmoid=True,
                               interpolation=True,
                               norm_type="batch",
                               act_type="lrelu").to(device)
    else:
        net = SingleUnet(in_channels=opt.input_nc,
                         out_channels=opt.output_nc,
                         domains=domains,
                         f_maps=opt.init_channel_number,
                         num_levels=5,
                         with_activation=True,
                         final_sigmoid=True,
                         interpolation=True,
                         norm_type="batch",
                         act_type="lrelu").to(device)

    return net


class TestDualStream(TestGeneric):
    def __init__(self, opt):
        super(TestDualStream, self).__init__(opt)
        self.net_umms = define_model(opt, self.device, self.domains)

        self.print_networks(opt.verbose)













