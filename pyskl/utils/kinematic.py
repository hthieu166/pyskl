import numpy as np

JOINTS_8_ANGLES = [
    (16,14,12), (14,12,6), (12,6,8), (6,8,10), 
    (15,13,11), (13,11,5), (11,5,7), (5,7,9)
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

def compute_joint_angle(keypoint):
    angle = []
    ns, nf, nj, nd = keypoint.shape
    keypoint = keypoint.reshape(ns*nf, nj, nd)
    for j0,j1,j2 in JOINTS_8_ANGLES:
        angle.append(joint_angle(keypoint, j0, j1, j2)[:,None])
    angle = np.concatenate(angle, axis=-1)
    angle = angle.reshape(ns, nf, -1)
    
    return angle