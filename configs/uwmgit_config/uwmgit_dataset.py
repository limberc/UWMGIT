num_classes = 3
# dataset settings
dataset_type = 'UWMGITDataset'
data_root = './data/mmseg_train/'
classes = ['large_bowel', 'small_bowel', 'stomach']
palette = [[0, 0, 0], [128, 128, 128], [255, 255, 255]]
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
size = 256
albu_train_transforms = [
    dict(type='RandomBrightnessContrast', p=0.5),
]
train_pipeline = [
    dict(type='LoadUWMGITFromFile', to_float32=True, color_type='unchanged', max_value='max'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(size, size), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Albu', transforms=albu_train_transforms),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(size, size), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadUWMGITFromFile', to_float32=True, color_type='unchanged', max_value='max'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(size, size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(size, size), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        multi_label=True,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        img_suffix=".png",
        seg_map_suffix='.png',
        split="splits/fold_0.txt",
        classes=classes,
        palette=palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        multi_label=True,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        img_suffix=".png",
        seg_map_suffix='.png',
        split="splits/holdout_0.txt",
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        multi_label=True,
        data_root=data_root,
        test_mode=True,
        img_dir='test/images',
        ann_dir='test/labels',
        img_suffix=".jpg",
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))
