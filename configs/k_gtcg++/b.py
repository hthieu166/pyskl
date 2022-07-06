model = dict(
    type='RecognizerGCNKinematic',
    backbone=dict(
        type='STGCN',
        in_channels = 1,
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='kinematic', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = '/mnt/data0-nfs/hthieu/data/pypkl_preprocessed/nturgbd/ntu60_hrnet.pkl'
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='NormalizeJointAngle'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2, key = 'angle'),
    dict(type='FormatGCNInput', num_person=2, key = 'keypoint'),
    dict(type='Collect', keys=['keypoint','angle', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['angle', 'keypoint'])
]

val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='NormalizeJointAngle'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='UniformSample', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2, key = 'angle'),
    dict(type='Collect', keys=['keypoint', 'angle', 'label'], meta_keys=[]),    
    dict(type='ToTensor', keys=['angle'])
]

test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='NormalizeJointAngle'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='UniformSample', clip_len=100, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2, key = 'angle'),
    dict(type='Collect', keys=['keypoint', 'angle', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['angle', 'keypoint'])
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
work_dir  = "./work_dirs/debug"
# work_dir = './work_dirs/k_stgcn++/joint_angle_stgcn++_ntu60_xsub_hrnet/b'
