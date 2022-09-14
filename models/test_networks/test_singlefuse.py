from models.test_networks.test_base import TestGeneric
from models.modules.MultimodalSegmentation.SingleUnetWithFuse import SingleUnetWithFuse


def define_model(opt, device, domains=None):
    net = SingleUnetWithFuse(in_channels=opt.input_nc,
                             out_channels=opt.output_nc,
                             f_maps=opt.init_channel_number,
                             num_levels=5,
                             with_activation=True,
                             final_sigmoid=True,
                             interpolation=opt.up_interpolate,
                             norm_type="batch",
                             act_type="lrelu").to(device)

    return net


class TestSingleFuse(TestGeneric):
    def __init__(self, opt):
        super(TestSingleFuse, self).__init__(opt)
        self.net_umms = define_model(opt, self.device, self.domains)

        self.print_networks(opt.verbose)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.source_predict, self.target_predict = self.net_umms(self.source_volume, self.target_volume)