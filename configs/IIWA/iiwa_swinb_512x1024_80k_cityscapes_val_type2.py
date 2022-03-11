_base_ = [
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='GTR_SEG',
    pretrained='pretrain/swin_base_patch4_window12_384_22k.pth',
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 1, 1),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    neck=dict(
        type='Cascade_TRP',
        in_channels=[128, 256, 512, 1024],
        d_models=[128, 256, 512],
        n_head=[4, 8, 16],
        dim_feedforwards=[256, 512, 1024],
        patch_sizes=[[4, 8], [8, 16], [4, 8], [4, 8], [4, 8], [4, 8]],
        dropout_ratio=0.1,
        num_outs=3,
        norm_cfg=dict(type='BN', requires_grad=True),
        out_type='2',
        overlap=False,
        cascade_num=1,
        pos_type='sin',
        upsample_cfg=dict(mode='nearest'),
        adapt_size1=(16, 16),
        adapt_size2=(16, 16),
        extra_out=False
    ),
    # neck=None,
    decode_head=dict(
        type='GTR_DECODER3',
        embed_dim=256,
        query_dim=256,
        memory_dims=[128, 256, 512],
        depths=[1, 1, 1, ],
        num_heads=[8, 8, 8, ],
        window_size_q=(4, 8),
        window_size_k=(4, 8),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[
                0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                1.0865, 1.1529, 1.0507
            ]),
        ignore_index=255,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        num_classes=19,
        aux_classes=19,
        head_scale=4,),
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'),
    test_cfg=dict(mode='slide', crop_size=(512, 1024), stride=(256, 512)), 
    )

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle2'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
        ]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
        ]

train_a = dict(
        type='CityscapesDataset',
        data_root='/media/estar/Data/cityscapes/',
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        pipeline=train_pipeline)
train_b = dict(
        type='CityscapesDataset',
        data_root='/media/estar/Data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=val_pipeline)



data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=train_a,
    val=dict(pipeline=val_pipeline),
    test=dict(
        type='CityscapesDataset',
        data_root='/media/estar/Data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=val_pipeline)
    )

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.2),
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
        }))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-06,
    power=0.96,
    min_lr=0.,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU')
load_from = './work_dirs/iiwa_swinb_512x1024_80k_cityscapes_CF_type1/iter_80000.pth'
