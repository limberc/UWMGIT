_base_ = [
    './uwmgit_dataset.py',
    '../_base_/default_runtime.py'
]
num_classes = 3
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
loss = [
    dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
]
model = dict(
    type='SMPUnet',
    backbone=dict(
        type='timm-efficientnet-b0',
        pretrained="imagenet"
    ),
    decode_head=dict(
        num_classes=num_classes,
        align_corners=False,
        loss_decode=loss
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole", multi_label=True)
)
total_iters = 1000
# optimizer
optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# learning policy
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# runtime settings
find_unused_parameters = True
runner = dict(type='IterBasedRunner', max_iters=int(total_iters))
checkpoint_config = dict(by_epoch=False, interval=int(total_iters), save_optimizer=False)
evaluation = dict(by_epoch=False, interval=min(5000, int(total_iters)), metric=['imDice', 'mDice'],
                  pre_eval=True)
fp16 = dict()
work_dir = f'./work_dirs/tract/baseline'
