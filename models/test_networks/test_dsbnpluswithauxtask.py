from models.test_networks.test_base import TestGeneric
from models.modules.MultimodalSegmentation.DSBNFusewithDeepSupervised import ChilopodUnetWithFuseAndDeepsupervised
from models.modules.MultimodalSegmentation.DSBNWithAuxTask import ChilopodUnetWithRegression, ChilopodUnetWithRegressionFinetuneV1, ChilopodUnetWithRegressionFinetuneV2


def define_model(opt, device, domains=None):
    if opt.network_type == 'finetunev1':
        net = ChilopodUnetWithRegressionFinetuneV1(
            in_channels=opt.input_nc,
            out_channels=opt.output_nc,
            f_maps=opt.init_channel_number,
            num_levels=5,
            with_activation=True,
            final_sigmoid=True,
            interpolation=opt.up_interpolate,
            norm_type="batch",
            act_type="lrelu",
        ).to(device)
    elif opt.network_type == 'finetunev2':
        net = ChilopodUnetWithRegressionFinetuneV2(
            in_channels=opt.input_nc,
            out_channels=opt.output_nc,
            f_maps=opt.init_channel_number,
            num_levels=5,
            with_activation=True,
            final_sigmoid=True,
            interpolation=opt.up_interpolate,
            norm_type="batch",
            act_type="lrelu",
        ).to(device)
    else:
        net = ChilopodUnetWithRegression(
            in_channels=opt.input_nc,
            out_channels=opt.output_nc,
            f_maps=opt.init_channel_number,
            num_levels=5,
            with_activation=True,
            final_sigmoid=True,
            interpolation=opt.up_interpolate,
            norm_type="batch",
            act_type="lrelu",
        ).to(device)
    return net


class TestDSBNPlusWithAuxTask(TestGeneric):
    def __init__(self, opt):
        super(TestDSBNPlusWithAuxTask, self).__init__(opt)
        self.model_names = ['umms']
        self.net_umms = define_model(opt, self.device, self.domains)
        self.print_networks(opt.verbose)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.network_type == 'finetunev1' or self.opt.network_type == 'finetunev2':
            self.source_aux_predict1, self.target_aux_predict1, self.ratio_predict, self.source_predict, self.target_predict = self.net_umms(
                self.source_volume, self.target_volume)
        else:
            self.source_predict, self.target_predict, self.ratio_predict = self.net_umms(self.source_volume,
                                                                                         self.target_volume)

    @staticmethod
    def get_tensors_detach(tensors):
        out_list = []
        for tensor in tensors:
            out_list.append(tensor.detach())
        return out_list
