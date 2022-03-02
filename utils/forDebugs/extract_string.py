import re
import os
from glob import glob


def find_best_result(root_dir=r'./traces/results', root_key=None):
    best_dc = 0.8
    best_weight = None
    for root, dirs, files in os.walk(root_dir):
        if root_key is not None:
            if root_key not in root:
                continue
        for file in files:
            if file == 'option.txt':
                file_path = os.path.join(root, file)
                with open(file_path) as info_file:
                    lines = info_file.readlines()
                    cur_dc_info = lines[-2]
                    cur_weight_info = lines[-1]
                    if 'dice' not in cur_dc_info:
                        continue
                    dc = float(cur_dc_info.strip().split(':')[-1])
                    # weight = cur_weight_info.strip().split(':')[-1]
                    weight = file_path
                    if best_dc < dc:
                        best_dc = dc
                        best_weight = weight
                    elif best_dc == dc:
                        if best_weight is None:
                            best_weight = weight
                        elif isinstance(best_weight, list):
                            best_weight.append(weight)
                        else:
                            best_weight = [best_weight].append(weight)
    print(f'best_dc: {best_dc}\n'
          f'best_weight:{best_weight}')


def find_best_dice(logs_dir=None, pat=r"^number(?:.*\s)+(?:total.*\s).*?dice.*?(\d\.\d+).*"):
    # '^number(?:.*\s)+(?:total.*\s)dice.*?(\d\.\d+).*'

    result_files = glob(os.path.join(logs_dir, '*.txt'))
    print(f'there is nothing in {logs_dir}')

    pat = re.compile(pat)
    best_dice = 0
    best_weight = None
    for result_file in result_files:
        with open(result_file, mode='r') as fread:
            content = fread.read()

        dice_result = pat.match(content)
        if dice_result is not None:
            dice = float(dice_result.groups()[0])
            if dice > best_dice:
                best_dice = dice
                best_weight = os.path.basename(result_file).split('.')[0]
            elif dice == best_dice:
                if isinstance(best_weight, list):
                    best_weight.append(os.path.basename(result_file).split('.')[0])
                else:
                    best_weight = [best_weight]
                    best_weight.append(os.path.basename(result_file).split('.')[0])

    print(f'best_dice: {best_dice}\t'
          f'best_weight: {best_weight}')
    return {'best_dice': best_dice, 'best_weight': best_weight}


if __name__ == '__main__':
    test_dir = r'/home/lf/data_fong/CODE/PycharmProject/DLForPytorch/traces/results'
    exp_name = r'trus_unet3d_DDP_Sybn_crop128_bs2x4_ch32_kaiming_dc_adam_1e-4_step_0.2_warmup_10_5e-5'
    phase_name = r'val'
    process_name = r'crop128_slide24_nopad_noaug'

    find_best_result(test_dir, root_key='val')
    result = find_best_dice(os.path.join(test_dir, exp_name, phase_name, process_name))    #
    print(result)



