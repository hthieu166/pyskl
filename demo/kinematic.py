import glob
from pyskl.smp import *
from pyskl.utils.kinematic import compute_joint_angle
from pyskl.utils.visualize import Vis3DPose, Vis2DPose
from mmcv import load, dump
import moviepy.editor as mpy
import numpy as np
import cv2

edges = [
        (0, 1, 'f'), (0, 2, 'f'), (1, 3, 'f'), (2, 4, 'f'), (0, 5, 't'), (0, 6, 't'),
        (5, 7, 'ru'), (6, 8, 'lu'), (7, 9, 'ru'), (8, 10, 'lu'), (5, 11, 't'), (6, 12, 't'),
        (11, 13, 'ld'), (12, 14, 'rd'), (13, 15, 'ld'), (14, 16, 'rd')
        ]

color_map = {
    'ru': ((0, 0x96, 0xc7), (0x3, 0x4, 0x5e)),
    'rd': ((0xca, 0xf0, 0xf8), (0x48, 0xca, 0xe4)),
    'lu': ((0x9d, 0x2, 0x8), (0x3, 0x7, 0x1e)),
    'ld': ((0xff, 0xba, 0x8), (0xe8, 0x5d, 0x4)),
    't': ((0xee, 0x8b, 0x98), (0xd9, 0x4, 0x29)),
    'f': ((0x8d, 0x99, 0xae), (0x2b, 0x2d, 0x42))}

def main():
    index = 0
    annotations = load('ntu60_2d.pkl')
    anno = annotations[index]
    
    angle = compute_joint_angle(anno['keypoint'])
    # anno = annotations[index]
    # vid = Vis2DPose(anno, thre=0.2, out_shape=(540, 960), layout='coco', fps=12, video=None)
    # vid.ipython_display()
    # annotations = load('ntu60_2d.pkl')
    # index = 0
    # single_frame= True
    # thre = 0.2
    # item = annotations[index]
    
    # kp = item['keypoint']
    # out_shape=(540, 960)
    
    # if single_frame: 
    #     total_frames = 1
    # else: 
    #     total_frames == kp.shape[1]
    
    # if 'keypoint_score' in item:
    #     kpscore = item['keypoint_score']
    #     kp = np.concatenate([kp, kpscore[..., None]], -1)
    # assert kp.shape[-1] == 3
    
    # img_shape = item.get('img_shape', out_shape)
    # kp[..., 0] *= out_shape[1] / img_shape[1]
    # kp[..., 1] *= out_shape[0] / img_shape[0]
    
    # frames = [np.ones([out_shape[0], out_shape[1], 3], dtype=np.uint8) * 255 for i in range(total_frames)]
    # # Defining joint angles
    # angles = [
    #     (16,14,12),
    #     (14,12,6),
    #     (12,6,8),
    #     (6,8,10),
    #     (15,13,11),
    #     (13,11,5),
    #     (11,5,7),
    #     (5,7,9)
    # ]
    
    # for i in tqdm(range(total_frames)):
    #     for m in range(kp.shape[0]):
    #         ske = kp[m, i]
    #         for e in edges:
    #             st, ed, co = e
    #             co_tup = color_map[co]
    #             j1, j2 = ske[st], ske[ed]
    #             j1x, j1y, j2x, j2y = int(j1[0]), int(j1[1]), int(j2[0]), int(j2[1])
    #             conf = min(j1[2], j2[2])
    #             if conf > thre:
    #                 color = [x + (y - x) * (conf - thre) / 0.8 for x, y in zip(co_tup[0], co_tup[1])]
    #                 color = tuple([int(x) for x in color])
    #                 frames[i] = cv2.line(frames[i], (j1x, j1y), (j2x, j2y), color, thickness=2)
    
    #         for a in angles:
    #             j0, j1, j2 = a
    #             ske = kp[...,:2][m, i]
    #             ang = joint_angle(ske, j0, j1, j2)
    #             frames[i] = cv2.putText(
    #                 img = frames[i],
    #                 text = "%.00f" % (ang * 360.0/ (2*np.pi)),
    #                 org = (int(ske[j1][0]), int(ske[j1][1])),
    #                 fontFace = cv2.FONT_HERSHEY_DUPLEX,
    #                 fontScale = 0.5,
    #                 color = (0,0,0),
    #                 thickness = 1
    #             )
    # if single_frame:
    #     cv2.imwrite("test.png",frames[0])
    # else:
    #     return mpy.ImageSequenceClip(frames, fps=fps)
if __name__ == "__main__":
    main()