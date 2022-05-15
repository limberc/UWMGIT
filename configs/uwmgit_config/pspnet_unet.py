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
