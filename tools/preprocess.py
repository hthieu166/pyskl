import numpy as np
import sys
import argparse
import pickle as pkl
from pyskl.utils.kinematic import *
from mmcv import load, dump
from tqdm import tqdm

layout_dict={
    "2d": JOINTS_2D_10_ANGLES_COCO,
    "3d": JOINTS_3D_19_ANGLES_NTU
}

def main():
    ann_file = sys.argv[1]
    annotations = load(ann_file)
    
    for idx in tqdm(range(len(annotations['annotations']))):
        agl = compute_joint_angle(annotations['annotations'][idx]["keypoint"], layout=layout_dict[sys.argv[2]])
        if np.isnan(agl).any():
            raise Exception("Array contains NaN")
        annotations['annotations'][idx]['angle'] = agl[..., None]
    
    dump(annotations, ann_file)

if __name__ == "__main__":  
    main()