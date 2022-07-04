import numpy as np
import sys
import argparse
import pickle as pkl
from pyskl.utils.kinematic import compute_joint_angle
from mmcv import load, dump
from tqdm import tqdm
def main():
    ann_file = sys.argv[1]
    annotations = load(ann_file)
    
    for idx in tqdm(range(len(annotations['annotations']))):
        agl = compute_joint_angle(annotations['annotations'][idx]["keypoint"])
        if np.isnan(agl).any():
            raise Exception("Array contains NaN")
        annotations['annotations'][idx]['angle'] = agl[..., None]
    
    dump(annotations, ann_file)

if __name__ == "__main__":  
    main()