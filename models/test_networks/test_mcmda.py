from models.test_networks.test_base import TestGeneric
from models.modules.MultimodalSegmentation.AdversarialDA import MCMDA, FeatureMapExtractor, FeatureDiscriminator, MaskDiscriminator, GANLoss


def define_model(opt, device, domains=None):
    net = MCMDA(in_channels=opt.input_nc,
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


def define_mask_discriminator(opt, device):
    net = MaskDiscriminator(input_nc=1,
                            ndf=16,
                            n_layers=4,
                            norm_type="batch",
                            act_type="lrelu").to(device)

    return net


def define_feature_discriminator(opt, device):
    net = FeatureDiscriminator(fmap=16,
                               mid_ch=16,
                               ndf=16,
                               n_layers=4,
                               norm_type="batch",
                               act_type="lrelu").to(device)

    return net


class TestMCMDA(TestGeneric):
    def __init__(self, opt):
        super(TestMCMDA, self).__init__(opt)
        if opt.use_adversarial:
            self.model_names = ['umms', 'df', 'dm']
            self.net_umms = define_model(opt, self.device, self.domains)
            self.net_df = define_feature_discriminator(opt, self.device)
            self.net_dm = define_mask_discriminator(opt, self.device)
            self.criterionGAN = GANLoss('lsgan').to(self.device)  # define GAN loss.
        else:
            self.model_names = ['umms']
            self.net_umms = define_model(opt, self.device, self.domains)

        self.feature_extractor = FeatureMapExtractor(self.net_umms, ['source_encoders.4',
                                                                     'target_encoders.4',
                                                                     'decoders.0',
                                                                     'decoders.1',
                                                                     'decoders.2',
                                                                     'decoders.3'])
        self.logit_extractor = FeatureMapExtractor(self.net_umms, ['outconv.conv3d'])

        self.print_networks(opt.verbose)

    def set_input(self, inputs):
        if self.opt.reverse_model:
            self.source_volume = inputs['us_volume'].to(self.device)  # bs C D H W, C=1
            self.source_label = inputs['us_label'].to(self.device)  # bs C D H W, C=1
            self.target_volume = inputs['mr_volume'].to(self.device)  # bs C D H W, C=1
            self.target_label = inputs['mr_label'].to(self.device)  # bs C D H W, C=1
            self.volume_path = {'source': inputs['us_volume_path'], 'target': inputs['mr_volume_path']}
            self.label_path = {'source': inputs['us_label_path'], 'target': inputs['mr_label_path']}
            self.spacing = {'source': inputs['us_spacing'].mean(0).tolist(),
                            'target': inputs['mr_spacing'].mean(0).tolist()}
            self.origin_shape = {'source': inputs['us_origin_shape'], 'target': inputs['mr_origin_shape']}
            self.now_shape = {'source': inputs['us_now_shape'], 'target': inputs['mr_now_shape']}
        else:
            self.source_volume = inputs['mr_volume'].to(self.device)  # bs C D H W, C=1
            self.source_label = inputs['mr_label'].to(self.device)  # bs C D H W, C=1
            self.target_volume = inputs['us_volume'].to(self.device)  # bs C D H W, C=1
            self.target_label = inputs['us_label'].to(self.device)  # bs C D H W, C=1
            self.volume_path = {'source': inputs['mr_volume_path'], 'target': inputs['us_volume_path']}
            self.label_path = {'source': inputs['mr_label_path'], 'target': inputs['us_label_path']}
            self.spacing = {'source': inputs['mr_spacing'].mean(0).tolist(),
                            'target': inputs['us_spacing'].mean(0).tolist()}
            self.origin_shape = {'source': inputs['mr_origin_shape'], 'target': inputs['us_origin_shape']}
            self.now_shape = {'source': inputs['mr_now_shape'], 'target': inputs['us_now_shape']}

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.source_predict = self.net_umms(self.source_volume, 'source')
        self.source_features = self.feature_extractor.get_out_feature()
        self.source_logits = self.logit_extractor.get_out_feature()
        self.target_predict = self.net_umms(self.target_volume, 'target')
        self.target_features = self.feature_extractor.get_out_feature()
        self.target_logits = self.logit_extractor.get_out_feature()

    @staticmethod
    def get_tensors_detach(tensors):
        out_list = []
        for tensor in tensors:
            out_list.append(tensor.detach())
        return out_list

    def get_current_metrics(self):
        metrics_ret = super(TestMCMDA, self).get_current_metrics()
        if self.opt.use_adversarial:
            metrics_ret['sdf'] = self.criterionGAN(self.net_df(*self.get_tensors_detach(self.source_features)), True)
            metrics_ret['sdm'] = self.criterionGAN(self.net_dm(*self.get_tensors_detach(self.source_logits)), True)
            metrics_ret['tdf'] = self.criterionGAN(self.net_df(*self.get_tensors_detach(self.target_features)), True)
            metrics_ret['tdm'] = self.criterionGAN(self.net_dm(*self.get_tensors_detach(self.target_logits)), True)
        return metrics_ret
