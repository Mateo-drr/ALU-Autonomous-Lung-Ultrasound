
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import numpy as np
import os
import cv2
import random
import sys
import copy
import matplotlib.pyplot as plt


# Get the current file's directory
current_dir = Path(__file__).resolve().parent
# Get the parent directory
parent_dir = current_dir.parent
# Append the imgProcessing directory to sys.path
img_processing_dir = parent_dir / 'imgProcessing'
sys.path.append(str(img_processing_dir))
# Now import the filtering module
import filtering as filt

'''
https://medium.com/@ym1942/create-a-gymnasium-custom-environment-part-2-1026b96dba69
https://blog.paperspace.com/creating-custom-environments-openai-gym/
'''

class LungUS(gym.Env):
    def __init__(self, path, rsize=512, angle=20, area=0.75, res=5, decay=0.9):
        super(LungUS, self).__init__()
        
        self.path = path
        datapath = Path(path)
        self.fileNames = [f.name for f in datapath.iterdir() if f.is_file()]
        self.current_index = 0
        self.numImgs = len(self.fileNames)-1
        
        self.angle=angle
        self.area=area
        self.rsize=rsize
        self.res=res
        self.decay=decay
        
        self.action_map = {
            '-x': 0,
            '+x': 1,
            '-y': 2,
            '+y': 3,
            '-ang': 4,
            '+ang': 5
        }
        
        self.state = []
        
        #self.action_space = spaces.Discrete(14)  # x,y,z,qw,qx,qy,qz twice for +-
        self.action_space = spaces.Discrete(6) #x,z,rot +-
        self.observation_space = spaces.Box(low=0, high=255, shape=(rsize, rsize), dtype=np.uint8)  # Example observation space

        self.current_image = self._load_image(self.current_index)
        
        self.counter = 0
        
    def _load_image(self, index):
        
        #Load numpy array
        image_path = os.path.join(self.path, self.fileNames[index])
        image = np.load(image_path)
        self.ogImg = image
        
        '''
        Give the image a random rotation and crop
        '''
        # Rotate 
        ang = random.randint(-self.angle, self.angle)
        rtimg = filt.rotate(image, ang)
        #store for plotting
        self.rotated = rtimg
        rotimg = filt.rotatClip(rtimg, image, ang)
        # Resize to original size to avoid size mismatches after different rotations
        try:
            rotimg = filt.rsize(rotimg.numpy(), x=image.shape[1], y = image.shape[0])[0][0]
        except:
            print('err', 
            rotimg.shape,
            image.shape,
            self.current_index,)
        # Move and crop
        movedimg,newx,endx,newy,endy,maxx,maxy = filt.moveClip(rotimg,
                                                               area=self.area,
                                                               cropidx=True)
        finalimg = filt.rsize(movedimg.numpy(),y=self.rsize,x=self.rsize)
        
        return finalimg.numpy()[0,0], [ang, newx, endx, newy, endy, maxx, maxy]
    
    def reset(self):
        self.current_index = random.randint(0, self.numImgs) # Pick next image randomly
        
        self.current_image, pos = self._load_image(self.current_index)
        #store targets for rewards
        self.state = {'initial':{'x':pos[1],'y':pos[3],'ang':pos[0],'maxx':pos[-2],'maxy':pos[-1]},
                      'currpos':{'x':pos[1],'y':pos[3],'ang':pos[0]}
                      }
        #Reset timeout counter
        self.counter=0
        
        return self.current_image
    
    def _new_state(self, index, state):
        #Load numpy array
        image_path = os.path.join(self.path, self.fileNames[index])
        image = np.load(image_path)
        self.ogImg = image
        #Rotate image with given angle
        rtimg = filt.rotate(image, state['currpos']['ang'])
        #store for plotting
        self.rotated = rtimg
        rotimg = filt.rotatClip(rtimg, image, state['currpos']['ang'])
        #Resize to original size to avoid size issues
        rotimg = filt.rsize(rotimg.numpy(), x=image.shape[1], y = image.shape[0])[0][0]
        
        # Move and crop with the given position
        movedimg,newx,endx,newy,endy,maxx,maxy = filt.moveClip(rotimg,
                                                               area=self.area,
                                                               newx=state['currpos']['x'],
                                                               newy=state['currpos']['y'],
                                                               cropidx=True)
        try:
            finalimg = filt.rsize(movedimg.numpy(),y=self.rsize,x=self.rsize)
        except:
            print('err2',
            movedimg.shape,
            self.rsize,
            self.current_index)
        
        #Store the new starting pixels
        state['currpos']['x'] = newx
        state['currpos']['y'] = newy
        
        return finalimg.numpy()[0,0], state
        
    def step(self, action):
        reward=0
        
        # Define how the environment changes in response to an action
        if action == 0: #-x
            self.state['currpos']['x'] -= self.res
        if action == 1: #+x
            self.state['currpos']['x'] += self.res
        if action == 2: #-y
            self.state['currpos']['y'] -= self.res
        if action == 3: #+y
            self.state['currpos']['y'] += self.res
        if action == 4: #-ang
            self.state['currpos']['ang'] -= self.res
        if action == 5: #+ang
            self.state['currpos']['ang'] += self.res
            
        #clip values and penalize
        if self.state['currpos']['ang'] >= self.angle:
            self.state['currpos']['ang'] = self.angle - self.res
            reward += -1
        elif self.state['currpos']['ang'] <= -self.angle:
            self.state['currpos']['ang'] = -self.angle + self.res
            reward += -1
        if self.state['currpos']['x'] >= self.state['initial']['maxx']:
            self.state['currpos']['x'] = self.state['initial']['maxx'] - self.res
            reward += -1
        elif self.state['currpos']['x'] < 0:
            self.state['currpos']['x'] = 0 + self.res
            reward += -1
        if self.state['currpos']['y'] >= self.state['initial']['maxy']:
            self.state['currpos']['y'] = self.state['initial']['maxy'] - self.res
            reward += -1
        elif self.state['currpos']['y'] < 0:
            self.state['currpos']['y'] = 0 + self.res
            reward += -1

        #get the real state ie after moving the image as planned
        self.current_image, self.state = self._new_state(self.current_index, self.state)
        
        '''
        Reward calculation
        '''
        #angle loss
        angloss = ((0 - self.state['currpos']['ang']) / self.angle) ** 2
        #x loss
        current_x = self.state['currpos']['x']
        initial_maxx = self.state['initial']['maxx'] // 2
        xloss = ((initial_maxx - current_x) / initial_maxx) ** 2
        #y loss
        current_y = self.state['currpos']['y']
        initial_maxy = self.state['initial']['maxy'] // 2
        yloss = ((initial_maxy - current_y) / initial_maxy) ** 2
        
        #Reward
        reward = -(angloss + xloss + yloss) #inverse of the loss
        
        #Reached target
        done = False
        if reward == 0:
            done = True
        
        #Extra info
        info = self.state
        
        timeout = False
        if self.counter >= 100:
            timeout=True
        info['timeout'] = timeout
        
        self.counter +=1
        
        return self.current_image, reward, done, info
    
    def render(self, mode='human'):
        # Optionally implement rendering
        if mode == 'human':
            # cv2.imshow('Image', self.current_image)
            # cv2.waitKey(1)
            fig, axs = plt.subplots(1, 2)  # Create a figure with 1 row and 2 columns
            
            axs[0].imshow(self.rotated, cmap='gray')  # Show self.current_image in the first subplot
            axs[0].axhline(self.state['currpos']['y'])
            axs[0].axvline(self.state['currpos']['x'])
            axs[0].axhline(self.rotated.shape[0] - self.state['currpos']['y'])
            axs[0].axvline(self.rotated.shape[1] - self.state['currpos']['x'])
            axs[0].axis('off')  # Turn off axis labels
            
            axs[1].imshow(self.ogImg, cmap='gray')  # Show self.ogImg in the second subplot
            axs[1].axis('off')  # Turn off axis labels
            plt.show()
    
    def close(self):
        cv2.destroyAllWindows()
