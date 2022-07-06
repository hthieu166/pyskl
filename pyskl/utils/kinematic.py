import numpy as np

JOINTS_2D_10_ANGLES_COCO = [
    (16,14,12), (14,12,6), (12,6,8), (6,8,10), 
    (15,13,11), (13,11,5), (11,5,7), (5,7,9),
    (6,0,4), (5,0,3),
    ]

JOINTS_3D_19_ANGLES_NTU = [
    (19,18,17),(18,17,16),(17,16,0),  # right leg: 0 - 1 - 2
    (15,14,13),(14,13,12),(13,12,0),  # left  leg: 3 - 4 - 5
    (12,0,1)  ,(4,20,2),  (20,2,3),   # body     : 6 - 7 - 8
    (20,8,9)  ,(8,9,10),  (9,10,11),(10,11,23),(23,11,24), # right arm: 9 - 10 - 11 - 12 - 13
    (20,4,5)  ,(4,5,6),   (5,6,7),   (6,7,21), (21,7,22)   # left  arm:14 - 15 - 16 - 17 - 18
    ]

def vector_angle(uo, vo):
    if len(uo.shape) == 2:
        u  = uo/(np.linalg.norm(uo, axis=1) + 1e-6)[:,None]
        v  = vo/(np.linalg.norm(vo, axis=1) + 1e-6)[:,None]
        dp = np.einsum('ij,ij->i',  u, v)
    else:
        u  = uo/np.linalg.norm(u)
        v  = vo/np.linalg.norm(v)
        dp = np.dot(u,v)
    dp = np.clip(dp, -1.0, 1.0)
    ag = np.arccos(dp)
    return ag

def joint_angle(loc, j0, j1, j2):
    if len(loc.shape) == 3:
        u = loc[:, j0] - loc[:, j1]
        v = loc[:, j2] - loc[:, j1]
    else:
        u = loc[j0] - loc[j1]
        v = loc[j2] - loc[j1]
    
    return(vector_angle(u, v))

def compute_bone_length(keypoint, layout = JOINTS_2D_10_ANGLES_COCO):
    ns, nf, nj, nd = keypoint.shape
    keypoint   = keypoint.reshape(ns*nf, nj, nd)
    bone_length= np.zeros((ns * nf, len(layout), 2))
    
    for i, (j0,j1,j2) in enumerate(layout):
        bone_length[:, i, 0] = np.linalg.norm(keypoint[:,j0,:] - keypoint[:,j1,:])
        bone_length[:, i, 1] = np.linalg.norm(keypoint[:,j2,:] - keypoint[:,j1,:])   
    bone_length = bone_length.reshape(ns, nf, len(layout), -1)
    return bone_length

def compute_joint_angle(keypoint, layout = JOINTS_2D_10_ANGLES_COCO):
    ns, nf, nj, nd = keypoint.shape
    keypoint = keypoint.reshape(ns*nf, nj, nd)
    angle    = np.zeros((ns*nf, len(layout), 1))
    for i, (j0,j1,j2) in enumerate(layout):
        angle[:,i,0] = joint_angle(keypoint, j0, j1, j2)
    angle = angle.reshape(ns, nf, len(layout), -1)
    return angle

# def compute_joint_angle(keypoint, layout = JOINTS_2D_10_ANGLES_COCO):
#     angle = []
#     ns, nf, nj, nd = keypoint.shape
#     keypoint = keypoint.reshape(ns*nf, nj, nd)
#     for j0,j1,j2 in layout:
#         angle.append(joint_angle(keypoint, j0, j1, j2)[:,None])
#     angle = np.concatenate(angle, axis=-1)
#     angle = angle.reshape(ns, nf, -1)
    
#     return angle