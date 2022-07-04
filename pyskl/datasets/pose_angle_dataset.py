from distutils.command.config import config
import os.path as osp

import mmcv

# from ..utils import get_root_logger
# from ..utils import joint_angle
from pyskl.datasets.pose_dataset import PoseDataset
from pyskl.datasets.builder import DATASETS
from pyskl.utils.kinematic import compute_joint_angle

@DATASETS.register_module()
class PoseAngleDataset(PoseDataset):
    """Pose dataset for action recognition.

    The dataset loads pose and apply specified transforms to return a
    dict containing pose information.

    The ann_file is a pickle file, the json file contains a list of
    annotations, the fields of an annotation include frame_dir(video_id),
    total_frames, label, kp, kpscore.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        split (str | None): The dataset split used. For UCF101 and HMDB51, allowed choices are 'train1', 'test1',
            'train2', 'test2', 'train3', 'test3'. For NTURGB+D, allowed choices are 'xsub_train', 'xsub_val',
            'xview_train', 'xview_val'. For NTURGB+D 120, allowed choices are 'xsub_train', 'xsub_val', 'xset_train',
            'xset_val'. For FineGYM, allowed choices are 'train', 'val'. Default: None.
        valid_ratio (float | None): The valid_ratio for videos in KineticsPose. For a video with n frames, it is a
            valid training sample only if n * valid_ratio frames have human pose. None means not applicable (only
            applicable to Kinetics Pose). Default: None.
        box_thr (float): The threshold for human proposals. Only boxes with confidence score larger than `box_thr` is
            kept. None means not applicable (only applicable to Kinetics). Allowed choices are 0.5, 0.6, 0.7, 0.8, 0.9.
            Default: 0.5.
        class_prob (list | None): The class-specific multiplier, which should be a list of length 'num_classes', each
            element >= 1. The goal is to resample some rare classes to improve the overall performance. None means no
            resampling performed. Default: None.
        memcached (bool): Whether keypoint is cached in memcached. If set as True, will use 'frame_dir' as the key to
            fetch 'keypoint' from memcached. Default: False.
        mc_cfg (tuple): The config for memcached client, only applicable if `memcached==True`.
            Default: ('localhost', 22077).
        **kwargs: Keyword arguments for 'BaseDataset'.
    """
    def load_annotations(self):
        data = super().load_annotations()
        # for idx in range(len(data)):
        #     agl = compute_joint_angle(data[idx]["keypoint"])
        #     data[idx]['angle'] = agl
        return data
    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        joints = self.prepare_test_frames(idx) if self.test_mode else self.prepare_train_frames(idx)
        return compute_joint_angle(joints['keypoint'])
        # return joints 

from mmcv import Config
from pyskl.datasets import build_dataset

if __name__ == "__main__":
    config = "/mnt/data0-nfs/hthieu/repo/pyskl/configs/k_gtcg++_joint_angle/j.py"
    cfg = Config.fromfile(config)
    dts = build_dataset(cfg.data.train)
    dt  = dts.__getitem__(0)
    # print(['keypoint'].shape)
    import ipdb; ipdb.set_trace()
    v   = next(dts)
    print(v)