import re
import h5py
from utils.others.img_io import show_array_3d, show_volume_label, show_volume_label_predict
from data.utils_data import h5_loader

if __name__ == '__main__':
    data_path = r'/home/lf/raid_lf/PROJECT/DLForPytorch/traces/results/' \
                r'trus_unet3d_DDP_SynBN_crop128_bs3x4_ch32_dc_adam_1e-4/test/' \
                r'slide_test_pad_noaug/65_net_trus_unet3d_DDP_SynBN_crop128_bs3x4_ch32_dc_adam_1e-4id-3.h5'
    patient_id = re.match(r'^/(?:.+/)*((\d+).*)\.h5$', data_path).groups()[-1]
    fr = h5py.File(data_path, 'r')
    label = fr.get('label')[:]
    segment = fr.get('segment')[:]
    volume = fr.get('pad_volume')[:]
    fr.close()
    # volume, segment, label = h5_loader(data_path,'volume','segment','label')
    show_volume_label_predict(volume.transpose((2, 1, 0)),
                              label.transpose((2, 1, 0)),
                              segment.transpose((2, 1, 0)),
                              True,
                              row=3, col=2, title=f'test on patient: {patient_id} ')
    pass

