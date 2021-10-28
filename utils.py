import json
import numpy as np
from transform import global2label

## Algorithm to detect the change in gripper state
def detect_change(gripper_state):
    change_state = np.zeros((gripper_state.shape))

    for i, state in enumerate(gripper_state):
        if i == 0:
            continue
        else:
            change_state[i] = np.abs(gripper_state[i] - gripper_state[i-1])
    
    indices = []
    filtered_gripper = gripper_state.copy()
    for i, c_state in enumerate(change_state):
        if c_state > 0 :
            indices.append(i)
    
    return indices

## Algorithm to filter the noises with a threshold in the gripper state
def noise_filter(gripper_state, threshold=100):

    change_state = np.zeros((gripper_state.shape))

    for i, state in enumerate(gripper_state):
        if i == 0:
            continue
        else:
            change_state[i] = np.abs(gripper_state[i] - gripper_state[i-1])
    
    indices = []
    filtered_gripper = gripper_state.copy()
    for i, c_state in enumerate(change_state):
        if c_state > 0 :
            indices.append(i)
        
    for index in indices:
        #index = min(index, len(gripper_state))
        count = 0
        for thres in range(0,threshold):
            #t = np.clip(index+thres, 1, len(gripper_state))

            p = index+thres
            t = max(0, min(p, len(gripper_state)-1))
            #print(t)
            if (gripper_state[index]==gripper_state[t]):
                count += 1

        if count < threshold:
            filtered_gripper[index:index+threshold] = gripper_state[index-1]
    
    return filtered_gripper

### Algorithm to convert the end-effector position to 2D position in the image

def conv_endeff_2Dpos(end_effectors, cam_pos, cam_ori, output_size, cam_fov):

    state_2D = []
    for i, end_eff in enumerate(end_effectors):
        state_2D.append(global2label(end_eff, cam_pos, cam_ori, cam_fov))

    return state_2D 

_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition,
                             -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition,
                             -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition,
                             -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                             0.0)
    return euler       

