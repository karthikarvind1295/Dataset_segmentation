import h5py
import os
import cv2
from tqdm import tqdm
import imageio
from collections import defaultdict
import numpy as np
import json
from utils import noise_filter, detect_change, conv_endeff_2Dpos, mat2euler
from transform import global2label
import matplotlib.pyplot as plt
import mujoco_py as mujo
import argparse


## CLASS TO RENDER IMAGES FROM DEMONSTRATIONS

class extracting_demonstration:
    def __init__(self, model_location, hd_location, save_path):
        self.model_location = model_location
        self.hd_location = hd_location
        self.save_path = save_path
        #self.cam_pos = np.array((0, 0, 0))
        self.fov = 75
        self.output_size = (224, 224)

        
    def state_extraction(self):
        
        f = h5py.File(self.hd_location, "r")
        data = f['data']

        model_xml = []
        dump = {}

        count = 20
        i = 0
        for demo_id in tqdm(data.keys()): #The 'key' for each demonstration inside the file

            if i < count:    
                dump = []
                end_eff_2D = []
                
                user = data[demo_id]
                state = np.array(user['states'])
                joint_gripper= np.array(user['gripper_actuations'])
                end_eff = np.array(user['right_dpos'])

                model_xml.append(dict(user.attrs)['model_file'])
                # for index , values in enumerate(end_eff):
                #     end_eff_temp = global2label(values, self.cam_pos, self.cam_pos, self.output_size, self.fov)
                #     end_eff_2D.append(end_eff_temp)

                path = self.image_rendering(demo_id, model_xml, state, end_eff)

                time = state[:,0]
                #print(len(joint_gripper))
                filtered_gripper = noise_filter(joint_gripper,150)
                indices = detect_change(filtered_gripper)

                #end_eff_2D = global2label(end_eff, self.cam_pos, self.cam_pos, self.output_size, self.fov)
                

                dump.append({
                    'indices': indices
                })
                with open(os.path.join(path,'frame.json'),'w') as f:
                    json.dump(dump, f)

            else:
                pass

            i = i+1
        # with open(os.path.join(self.save_path,'frame.json'),'w') as f:
        #     json.dump(dump, f)
    
    
    def image_rendering(self, demo_id, model_xml, state, end_eff_2d):

        model_path = os.path.join(self.model_location, model_xml[0]) #Model is assumed to be the first one and it is verified with the other and it remains the same
        model = mujo.load_model_from_path(model_path)
        sim = mujo.MjSim(model)

        viewer = mujo.MjViewer(sim)
        # time = state[:,0]  #### FOR PEG EXPERIMENT
        # qpos = state[:, 1:25]
        # qvel = state[:, 25:47]
        time = state[:,0]   ##### FOR MILK CAN EXPERIMENT
        qpos = state[:, 1:39]
        qvel = state[:, 39:]

        path = os.path.join(self.save_path,'%s'%demo_id)
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            pass
        
        #img_array = []

        for i in range(len(time)):
            #Set the simulation environment
            state = mujo.MjSimState(time[i], qpos[i], qvel[i], None, None)
            sim.set_state(state)
            sim.forward()
            
            #self.cam_pos = sim.data.get_camera_xpos('birdview')
            #self.cam_orientation = mat2euler(sim.data.get_camera_xmat('birdview'))
            #end_eff_temp = global2label(end_eff_2d[i], self.cam_pos, self.cam_pos, self.output_size, self.fov)
            #print(self.cam_orientation)
            if i % 10 == 0:
                #Render image from 
                image = sim.render(width=self.output_size[0], height=self.output_size[1], camera_name='agentview')
                #image = cv2.circle(np.array(image[::-1, :, :]), (int(end_eff_temp[0]), int(end_eff_temp[1])), 10, -1)
                #cv2.imwrite(os.path.join(path,'image_%d.png'%i), image)
                cv2.imwrite(os.path.join(path,'image_%d.png'%i), image[::-1, :, :])
                # image = image[::-1, :, :]
                # height, width, _depth = image.shape
                # img_array.append(image)
        
        # out = cv2.VideoWriter(os.path.join(path,'%s.avi'%demo_id),cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))
        # for i in range(len(img_array)):
        #     out.write(img_array[i])
        
        # out.release

        return path





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parsing the frames from videos')
    parser.add_argument('-i', '--input_location', type=str, metavar='', required=True, help='Given the location of the h5py file')
    parser.add_argument('-m', '--model_file', type=str, metavar='', required=False, help='Given the model xml location')
    parser.add_argument('-s', '--save_path', type=str, metavar='', required=True, help='Given the save location path')
    args = parser.parse_args()

    extract = extracting_demonstration(args.model_file, args.input_location, args.save_path)
    extract.state_extraction()





