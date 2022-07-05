model = dict(
    type='RecognizerGCNKinematic',
    backbone=dict(
        type='STGCN',
        in_channels = 1,
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='kinematic_3d', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = '/mnt/data0-nfs/hthieu/data/pypkl_preprocessed/nturgbd/ntu60_3danno.pkl'
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='NormalizeJointAngle'),
    dict(type='GenSkeKinematicFeat', dataset='nturgb+d', feats=['a']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2, key='kinematic'),
    dict(type='FormatGCNInput', num_person=2, key='keypoint'),
    dict(type='Collect', keys=['keypoint', 'kinematic', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['kinematic', 'keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='NormalizeJointAngle'),
    dict(type='GenSkeKinematicFeat', dataset='nturgb+d', feats=['a']),
    dict(type='UniformSample', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2, key='kinematic'),
    dict(type='FormatGCNInput', num_person=2, key='keypoint'),
    dict(type='Collect', keys=['keypoint', 'kinematic', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['kinematic', 'keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='NormalizeJointAngle'),
    dict(type='GenSkeKinematicFeat', dataset='nturgb+d', feats=['a']),
    dict(type='UniformSample', clip_len=100, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2, key='kinematic'),
    dict(type='FormatGCNInput', num_person=2, key='keypoint'),
    dict(type='Collect', keys=['keypoint', 'kinematic', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['kinematic', 'keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xsub_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xsub_val'))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/k_stgcn++/k_stgcn++_ntu60_xsub_3dkp/a'
