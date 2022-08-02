from yacs.config import CfgNode as CN


ex_config = CN()
# _C = CN(new_allowed=True)  # init_dict=None, key_list=None, new_allowed=False


# ---------------------------------------------------------------------------- #
# 数据参数：包括数据集参数和数据增强参数
# ---------------------------------------------------------------------------- #
# dataset
ex_config.current_epoch = None
