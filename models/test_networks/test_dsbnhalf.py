from models.test_networks.test_base import TestGeneric
from models.modules.MultimodalSegmentation.DsbnHalf import DsbnHalfUnet


def define_model(opt, device, domains=None):
    net = DsbnHalfUnet(in_channels=opt.input_nc,
                       out_channels=opt.output_nc,
                       domains=domains,
                       f_maps=opt.init_channel_number,
                       num_levels=5,
                       with_activation=True,
                       final_sigmoid=True,
                       interpolation=opt.up_interpolate,
                       norm_type="batch",
                       act_type="lrelu").to(device)

    return net


class TestDSBNHalf(TestGeneric):
    def __init__(self, opt):
        super(TestDSBNHalf, self).__init__(opt)
        self.net_umms = define_model(opt, self.device, self.domains)

        self.print_networks(opt.verbose)













