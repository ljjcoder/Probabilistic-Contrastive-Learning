_base_ = [
    '../_base_/cls_datasets/uda_officehome/uda_officehome_A_C.py',
    '../_base_/cls_models/resnet_50_gvb.py'
]

log_interval = 100
val_interval = 500

control = dict(
    log_interval=log_interval,
    max_iters=10000,
    val_interval=val_interval,
    cudnn_deterministic=True,
    save_interval=1000,
    max_save_num=1,
    seed=2,
)

train = dict(
    log_interval=log_interval,
    using_NCE=True,
    NCE_weight=0.05,
)

test = dict(
)
