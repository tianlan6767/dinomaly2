"""Dinomaly2 训练配置：ad_fine_crop_800_5cls（5 点位 ROI 800×800）。

启动：python run_train.py -c configs/ad_fine_crop_800_5cls.py
"""

data_path = r"E:\AOI_EVT_B_NI\02_train\TOP\01_ader\ad_fine_crop_800_5cls"

item_list = [
    "x00768_y01536",
    "x00768_y03072",
    "x01536_y03072",
    "x06912_y00000",
    "x07663_y00000",
]

save_dir = "./saved_results"
save_name = "dinomaly2_fine_crop_800_5cls_in800"

# image_size == crop_size：不 CenterCrop 丢边；须为 patch_size=14 的整数倍（800 不行，用 798=57×14）
image_size = 798
crop_size = 798
batch_size = 4
num_workers = 0
total_iters = 40_000
save_every_iters = 5000
eval_resize_mask = 256

backbone = "dinov2reg_vit_small_14"
dropout = 0.4
la = 1
lc = 2
ll = 1
ll_ratio = 0.9
ll_factor = 0.1
cr = 1
lr_decay_ratio = 1.0
cuda = 0
seed = 1
