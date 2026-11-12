"""Dinomaly2 训练配置：MVTec 风格 ad_test（5 类 bottle/cable/...）。

启动：python run_train.py -c configs/ad_test_5cls.py
"""

data_path = r"E:\AOI_EVT_B_NI\02_train\TOP\01_ader\ad_test"

# None = 自动扫描 data_path 下子目录（排除非目录）
item_list = None

save_dir = "./saved_results"
save_name = "dinomaly2_ad_test_5cls"

image_size = 448
crop_size = 392
batch_size = 16
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
