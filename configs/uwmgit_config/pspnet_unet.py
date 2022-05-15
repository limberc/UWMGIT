_base_ = [
    './uwmgit_dataset.py',
    '../_base_/default_runtime.py',
    '../_base_/models/pspnet_unet_s5-d16.py'
]
# model settings
loss = [
    dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
]
model = dict(
    decode_head=dict(
        num_classes=3,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    ),
    auxiliary_head=dict(
        num_classes=3,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    )
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
