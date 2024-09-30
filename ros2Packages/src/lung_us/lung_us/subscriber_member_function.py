# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray 
from us_msg.msg import StampedArray

import matplotlib.pyplot as plt
import cv2
# import cv2.aruco as aruco
import numpy as np
import torch
from torchvision import transforms

from pathlib import Path
import sys
current_dir = Path(__file__).resolve()
basePath = current_dir.parent.parent.parent.parent.parent
modelPath = basePath / 'imgProcessing' / 'ml' 
print(modelPath)

sys.path.append(modelPath.as_posix())
from plExtractor import plExtractor
sys.path.append(modelPath.parent.as_posix())
import byble as byb
sys.path.append(basePath.as_posix())
from bayesianOp import ManualGPMinimize

from geometry_msgs.msg import PoseStamped
from skopt.space import Real
import time

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        print('Started listener...')
        
        self.subscription = self.create_subscription(StampedArray,
                                                     '/imgs',
                                                     self.callback,2)
        
        self.publisher = self.create_publisher(PoseStamped,
                                               '/target',
                                               10)  # Publish move commands
        
        self.subscription  
        
        #plExtractor load weights
        self.model = plExtractor('cpu')
        self.model.eval()
        self.model.load_state_dict(torch.load(modelPath / 'model.pth'))
        
        #Resize operation to prepare the data for the model
        self.rsize = transforms.Resize((512,128),antialias=True)
        
        self.first=True
        self.plotting=False
        
        self.callback(np.random.rand(129, 512))

    def costFunc(self, pos):
        
        img= self.img
        #TODO send to device
        #Send the image to the trained model
        print(img.shape)
        #TODO remove extra unsqueeze needed cause of cmap in dl
        prepImg = self.rsize(torch.tensor(img).unsqueeze(0).unsqueeze(0)) #add batch and channel dim
        min_val = torch.min(prepImg)
        max_val = torch.max(prepImg)
        prepImg = (prepImg - min_val) / (max_val - min_val)
        print(prepImg.shape)
        #TODO remove double output form model
        mask,_ = self.model(prepImg.to(torch.float32).unsqueeze(0)) 
        #Get the pixels in height where the pleura starts and ends
        print(mask.shape)
        
        try:
            one_mask = (mask[0,0,0,:] == 1)  # Boolean mask for 1s #first class
            print(one_mask.shape)
            top = one_mask.nonzero(as_tuple=True)[0][0]  # First index with 1
            btm = one_mask.nonzero(as_tuple=True)[0][-1]  # Last index with 1
        except:
            print(one_mask)
        
        print(top,btm, img.shape)
        #limit predictions
        top = 100#top.clamp(0,img.shape[0])
        btm = 200#btm.clamp(0,img.shape[0])
        
        #crop the pleura zone and calculate the features
        print(top,btm)
        pleura = img[top:btm,:]
        #variance
        yhist,xhist = byb.getHist(pleura)
        #TODO add other features
        
        
        cost = np.var(yhist)
        
        triangle_array = -np.concatenate((np.arange(1, 11), np.arange(11, 0, -1)))
        c = triangle_array[int(abs(pos[0]+10))]
        
        print(cost, c, pos[0]+10)
        return c


    def callback(self, msg):
        self.get_logger().info('image')
        
        #Receive and rebuild the image
        img = msg#np.array(msg.array.data)
        print(img.shape)
        img = img.reshape((129,512)).transpose()
        print(img.shape)
        
        self.img = img
        
        #Define the search space around the initial position
        if self.first:
            self.cpos = [0]#,0,0,0,0,0,0] #get the current position of the robot where the image was taken
            searchSpace = [
                                Real(-10.0, 10.0),  # x
                                # Real(-10.0, 10.0),  # y
                                # Real(-10.0, 10.0),  # z
                                # Real(-1.0, 1.0),    # q0 (quaternion)
                                # Real(-1.0, 1.0),    # q1 (quaternion)
                                # Real(-1.0, 1.0),    # q2 (quaternion)
                                # Real(-1.0, 1.0)     # q3 (quaternion)
                            ]
            
            #Start the optimization
            print('a')
            self.optimizer = ManualGPMinimize(self.costFunc, searchSpace, initial_point=self.cpos)
            print('b')
            # cost = self.costFunc(self.cpos) #check score of last suggested position
            # self.optimizer.update(self.cpos, cost)  # Update the optimizer
            # nextMove = self.optimizer.step()
            
            
            self.first=False
        
        # else:
        # nextMove = self.optimizer.step()
        print('a')
        cost = self.costFunc(self.cpos) #check score of last suggested position
        self.optimizer.update(self.cpos, cost)  # Update the optimizer
        nextMove = self.optimizer.step() #get new move
        print('b')
        
        print(nextMove)
        #TODO send the new move to the robot
        # postPose(self.publisher, nextMove)
        
        self.cpos = nextMove
        
        print('Best result:', self.optimizer.getResult())
        
        #Autoloop --> debugging
        time.sleep(2)
        self.callback(np.random.rand(129, 512))
        
        
        if self.plotting:
            # Convert image for display
            img = 20 * np.log10(abs(img) + 1)
            img = np.uint8((img - img.min()) / (img.max() - img.min()) * 255)
            
            # Apply the viridis colormap
            img_colored = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
            
            # Display the image in the OpenCV window
            cv2.imshow("Image", img_colored)
            cv2.waitKey(1)
        
        return
        
def postPose(publisher,targets):
    msg = PoseStamped()
    msg.header.frame_id = 'base_link'
    #Target coordinates
    msg.pose.position.x = targets[0]
    msg.pose.position.y = targets[1]
    msg.pose.position.z = targets[2]
    #Quaternion
    msg.pose.orientation.x = targets[3]
    msg.pose.orientation.y = targets[4]
    msg.pose.orientation.z = targets[5]
    msg.pose.orientation.w = targets[6]
    #Publish the target
    publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
