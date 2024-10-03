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

# from std_msgs.msg import String
# from sensor_msgs.msg import Image
# from std_msgs.msg import Float32MultiArray 
from us_msg.msg import StampedArray
from std_msgs.msg import Bool

# import matplotlib.pyplot as plt
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
import pathCalc as pc
from pprint import pprint
from confidenceMap import confidenceMap
from skimage.transform import resize
from scipy.ndimage import laplace
import pickle


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        print('Started listener...')
        
        self.subscription = self.create_subscription(StampedArray,
                                                     '/imgs',
                                                     self.callback,2)
        
        self.publisher = self.create_publisher(PoseStamped,
                                               '/target_frame',
                                               10)  # Publish move commands
        
        
        
        self.poseNode = self.create_subscription(PoseStamped,
                                                     '/cartesian_compliance_controller/current_pose',
                                                     self.storePose,2)  
        self.storeMsg=False
        self.pose=None
        self.cpos = None #get the current position of the robot where the image was taken
        self.livePose=None
        self.sentPose=None
        
        self.reqImage = self.create_publisher(Bool,
                                              '/req_img',
                                              2)
        
        #plExtractor load weights
        self.model = plExtractor('cpu')
        self.model.eval()
        self.model.load_state_dict(torch.load(modelPath / 'model.pth', map_location=torch.device('cpu'), weights_only=True))
        
        #Resize operation to prepare the data for the model
        self.rsize = transforms.Resize((512,128),antialias=True)
        
        self.first=True
        self.plotting=True
        
        #Preprocessing variables
        self.fs = 50e6
        self.fc = 6e6
        self.highcut=self.fc+1e6
        self.lowcut=self.fc-1e6
        
        self.counter=0
        
        #load scaler
        with open(basePath / 'minmax_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
            
        self.startTime = None
        
        
        #Wait for user input
        _=input('Enter to begin')
        #Ask for initial image
        self.askImg(True)
        

    def costFunc(self, pos):
        
        img= self.img
        #TODO send to device 
        #Send the image to the trained model
        print(img.shape, 'image shape')
        #TODO remove extra unsqueeze needed cause of cmap in dl
        prepImg = self.rsize(torch.tensor(img).unsqueeze(0).unsqueeze(0)) #add batch and channel dim
        min_val = torch.min(prepImg)
        max_val = torch.max(prepImg)
        prepImg = (prepImg - min_val) / (max_val - min_val)
        print(prepImg.shape, 'resized image')
        #TODO remove double output form model
        mask,_ = self.model(prepImg.to(torch.float32).unsqueeze(0)) 
        #Get the pixels in height where the pleura starts and ends
        print(mask.shape, 'mask shape')
        
        #TODO maybe this comment causes errors
        # if mask is not None:
        try:
            mask = mask.clamp(0,1)
            one_mask = (mask[0,0,0,:] == 1)  # Boolean mask for 1s #first class
            print(one_mask.shape, 'one_mask shape')
            top = one_mask.nonzero(as_tuple=True)[0][0]  # First index with 1
            btm = one_mask.nonzero(as_tuple=True)[0][-1]  # Last index with 1
        
            print(top,btm, img.shape, 'crop points')
            #limit predictions
            top = top.clamp(0,img.shape[0])
            btm = btm.clamp(0,img.shape[0])
    
            top,btm=50,-200 
            pleura = img[top:btm,:]
        except:
            top,btm=50,-200 #dont crop
            pleura = img[top:btm,:]
            print('segmentation failed')
        
        if self.plotting:
            self.plotUS(pleura, name='crop')
        
        #variance
        #TODO check width and height order in envelope
        hb = byb.envelope(pleura)
        hbyh, _ = byb.getHist(hb)
        #confidence map #TODO check if should send cropped or not cropped
        cmap = confidenceMap(img[10:,:],rsize=True) #cropping top pixels with noise
        cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)
        if self.plotting:
            self.plotUS(cmap,norm=False,name='cmap')
        cyhist,_ = byb.getHist(cmap[top:btm,:])
        vardif = np.var(np.diff(cyhist))
        #intensity
        lum = np.mean(np.abs(pleura))
        #laplace
        lap = laplace(pleura).var()
        
        features = [np.var(hbyh), vardif, lum, lap]
        
        #scale
        normFeat = self.scaler.transform([features])[0] #need to remove batch 
        #weights
        w = [-0.96480376, -0.67775069,  1.80283288,  0.47980818]
        
        print(normFeat,'normalize features')
        
        linmod = [x * y for x, y in zip(normFeat, w)]
        print('Features', linmod)
        
        cost = -(np.sum(linmod) + -0.027142835796609033)
        
        #Confidence maximization
        threshold = 0.85
        confC = np.sum(cmap < threshold)
        # Normalize the high confidence count by the total number of pixels
        confRT = confC / cmap.size
        print('\n',confRT, 'CONFIDENCE SCORE\n')

        # triangle_array = -np.concatenate((np.arange(1, 11), np.arange(11, 0, -1)))
        # c = triangle_array[int(abs(pos[0]+10))]
        
        print(cost, 'COST')#, c, pos[0]+10)
        
        if confRT <= 0.9:
            return cost
        else:
            return confRT*10 
    
    def plotUS(self,img,name='Image',save=False,norm=True):
        # Convert image for display
        if norm:
            img = 20 * np.log10(abs(img) + 1)
        img = np.uint8((img - img.min()) / (img.max() - img.min()) * 255)
        
        # Apply the viridis colormap
        img_colored = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
        
        # Display the image in the OpenCV window
        cv2.imshow(name, img_colored)
        cv2.waitKey(100)
        
        # Save the colored image
        if save:
            cv2.imwrite(f'/home/mateo-drr/Pictures/us/us{self.counter}.png', img_colored)
            self.counter+=1
        
    def plotScore(self):
        #TODO plot in log?
        
        # Assuming self.optimizer.get_tested_positions_and_scores() returns a tuple of (positions, scores)
        positions, scores = self.optimizer.get_tested_positions_and_scores()
    
        # Check if scores is empty
        if not scores:
            print("No scores to plot.")
            return
        
        # Define the canvas size for plotting
        width, height = 800, 600
        margin = 50  # margin around the plot area
        plot_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # white canvas
    
        # Define the plot area size
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin
    
        # Find min and max scores for scaling the plot to fit within the canvas
        min_score = min(scores)
        max_score = max(scores)
    
        # Check if all scores are the same
        if min_score == max_score:
            # If all scores are the same, set all y-coordinates to a fixed value (e.g., middle of the plot)
            fixed_y = height - margin - plot_height // 2
            # Draw the axes
            cv2.line(plot_img, (margin, margin), (margin, height - margin), (0, 0, 0), 2)  # y-axis
            cv2.line(plot_img, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 2)  # x-axis
            
            # Draw a horizontal line for the fixed score
            cv2.line(plot_img, (margin, fixed_y), (width - margin, fixed_y), (255, 0, 0), 2)
    
        else:
            # Draw the axes
            cv2.line(plot_img, (margin, margin), (margin, height - margin), (0, 0, 0), 2)  # y-axis
            cv2.line(plot_img, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 2)  # x-axis
            
            # Draw the first point
            y_first = height - margin - int((scores[0] - min_score) / (max_score - min_score) * plot_height)
            x_first = margin
            cv2.circle(plot_img, (x_first, y_first), 5, (255, 0, 0), -1)  # Draw first point
    
            # Draw the score values as a line plot
            for i in range(1, len(scores)):
                x1 = margin + int((i - 1) / (len(scores) - 1) * plot_width)
                y1 = height - margin - int((scores[i - 1] - min_score) / (max_score - min_score) * plot_height)
                x2 = margin + int(i / (len(scores) - 1) * plot_width)
                y2 = height - margin - int((scores[i] - min_score) / (max_score - min_score) * plot_height)
    
                # Draw the line between points
                cv2.line(plot_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Optionally draw the current point
                cv2.circle(plot_img, (x2, y2), 5, (255, 0, 0), -1)  # Draw the current point
    
        # Display the image
        cv2.imshow("Optimization Convergence", plot_img)
        cv2.waitKey(100)
        # cv2.destroyAllWindows()


    def storePose(self,msg):
        self.livePose=msg
        
        if self.sentPose is not None and self.has_reached_target(msg, self.sentPose):
            print('\n reached target!')
            self.askImg(True)
            self.sentPose = None
            self.startTime = None
            
        elif self.startTime is not None and (time.time() - self.startTime) > 10:
            print('\nTimeout reached, stopping the current attempt, assesing current pos')
            self.askImg(True)
            self.sentPose = None
            self.startTime = None

            
    def has_reached_target(self, current_pose, target_pose, position_tolerance=(0.001, 0.001, 0.001), orientation_tolerance=0.00001):
    
        # Extract positions
        current_position = np.array([current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z])
        target_position = np.array([target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z])
        
        # Calculate the absolute difference for each position axis
        position_error = np.abs(current_position - target_position)
        
        # Extract orientations
        current_orientation = current_pose.pose.orientation
        target_orientation = target_pose.pose.orientation
        
        # Calculate the difference between the quaternions using dot product
        orientation_diff = abs(current_orientation.x * target_orientation.x +
                               current_orientation.y * target_orientation.y +
                               current_orientation.z * target_orientation.z +
                               current_orientation.w * target_orientation.w)
    
        # A quaternion is a unit vector, hence the maximum dot product is 1
        # If the value is close to 1, it means orientations are nearly identical
        orientation_diff = 1 - orientation_diff  # Get the distance to 1 for easier comparison
        
        print(f"\rPosition error (x, y, z): {position_error[0]:.4f}, {position_error[1]:.4f}, {position_error[2]:.4f}, Orientation error: {orientation_diff:.5f}", end='')
        
        # Check if each position axis is within the corresponding tolerance
        position_reached = all(position_error < np.array(position_tolerance))
        
        # Check if orientation is within tolerance
        orientation_reached = orientation_diff < orientation_tolerance
        
        # If both position and orientation are within tolerance
        if position_reached and orientation_reached:
            #Formatting the pose
            self.cpos = self.formatPose(self.getPose())
            #Clipping it within bounds
            self.cpos = list(np.clip(self.cpos, self.lower_bounds, self.upper_bounds))

            return True
        return False

        
    def askImg(self, ask):
        print('Asking for an image', ask)
        msg = Bool()
        msg.data = ask
        self.reqImage.publish(msg)
        
    def getPose(self):
        if self.livePose is None:
            print('Didnt get position from robot, maybe its off?')
        return self.livePose
        
    def formatPose(self,receivedPose):
        #extract the quaternions
        q = pc.getQuat([receivedPose.pose.orientation.w,
                       receivedPose.pose.orientation.x,
                       receivedPose.pose.orientation.y,
                       receivedPose.pose.orientation.z
                       ], numpy=False)
        #transform to euler angles
        rot = q.SE3().rpy(unit='deg')
        #Join everything
        
        
        return [# Extract position data
                        receivedPose.pose.position.x*100,
                        receivedPose.pose.position.y*100,
                        receivedPose.pose.position.z*100,
                        
                        #rot
                        rot[0],
                        rot[1],
                        rot[2]
                        ]
    
    def callback(self, msg):
        self.get_logger().info('image')
        
        #Stop asking for an image
        self.askImg(False)
        
        #Receive and rebuild the image
        img = np.array(msg.array.data)
        print(img.shape, 'received data shape')
        img = img.reshape((128,512)).transpose()
        print(img.shape, 'resshaped data')
        
        # img = self.preproc(img)
        
        if self.plotting:
            self.plotUS(img,save=True)
        
        # _=input()
        
        self.img = img
        
        #Define the search space around the initial position
        if self.first:
            
            print(self.getPose())
            
            while (self.cpos is None):
                self.cpos = self.formatPose(self.getPose())
            
            # q = pc.getQuat([self.pose.pose.orientation.w,
            #                self.pose.pose.orientation.x,
            #                self.pose.pose.orientation.y,
            #                self.pose.pose.orientation.z
            #                ], numpy=False)
            # print(q, [self.pose.pose.orientation.w,
            #                self.pose.pose.orientation.x,
            #                self.pose.pose.orientation.y,
            #                self.pose.pose.orientation.z
            #                ], q.SE3(), q.SE3().rpy(unit='deg'))
            # rot = q.SE3().rpy(unit='deg')
            
            # self.cpos = [# Extract position data
            #                 self.pose.pose.position.x*100,
            #                 self.pose.pose.position.y*100,
            #                 self.pose.pose.position.z*100,
                            
            #                 #rot
            #                 rot[0],
            #                 rot[1],
            #                 rot[2]
            #                 ]
            
            print('current Pose:', self.cpos)
            self.searchSpace = [
                                Real(self.cpos[0] - 2.0, self.cpos[0] + 2.0),  # x
                                Real(self.cpos[1] - 0.5, self.cpos[1] + 2.0),  # y
                                Real(self.cpos[2] - 0.01, self.cpos[2] + (0.2+ 2.5)),  # z with deeper targets
                                Real(self.cpos[3] - 10.0, self.cpos[3] + 10.0),  # r1
                                Real(self.cpos[4] - 10.0, self.cpos[4] + 10.0),  # r2
                                Real(self.cpos[5] - 10.0, self.cpos[5] + 10.0)   # r3
                            # ]
                                # Real(-1.0, 1.0),    # q0 (quaternion)
                                # Real(-1.0, 1.0),    # q1 (quaternion)
                                # Real(-1.0, 1.0),    # q2 (quaternion)
                                # Real(-1.0, 1.0)     # q3 (quaternion)
                            ] #this is in cm!! robot uses m
            # Extract bounds from searchSpace
            self.lower_bounds = np.array([r.low for r in self.searchSpace])
            self.upper_bounds = np.array([r.high for r in self.searchSpace])

            
            #Start the optimization
            print('a')
            self.optimizer = ManualGPMinimize(self.costFunc, self.searchSpace, initial_point=self.cpos)
            print('b')
            # cost = self.costFunc(self.cpos) #check score of last suggested position
            # self.optimizer.update(self.cpos, cost)  # Update the optimizer
            # nextMove = self.optimizer.step()
            
            
            self.first=False
        
        # else:
        # nextMove = self.optimizer.step()
        print('='*80)
        pprint(self.cpos)
        pprint(self.pose)
        print('='*80)
        cost = self.costFunc(self.cpos) #check score of last suggested position
        self.optimizer.update(self.cpos, cost)  # Update the optimizer
        nextMove = self.optimizer.step() #get new move
        print(f'Updatinf cost {cost} at position {self.cpos}')
        
        
        print(nextMove)
        
        rotmat = pc.rot2SE3(nextMove[3],nextMove[4],nextMove[5],unit='deg')
        quat = pc.getQuat(rotmat)
        print(quat)
        
        print(type(nextMove))
        target = list(np.array(nextMove)*0.01)
        print('enter to move to [m] or f to finish ', target[0:3], nextMove[3:])
        user_input = input("").strip().lower()  # Get user input
        if user_input == 'f':
            #End the code
            self.get_logger().info("Finishing the node...")
            
            print("RESULTS")
            
            # Assuming self.optimizer.get_tested_positions_and_scores() returns a tuple of (positions, scores)
            positions, scores = self.optimizer.get_tested_positions_and_scores()
            
            # Print the header
            print(f"{'Index':<5} {'Position':<60} {'Score':<10}")
            print("="*80)
            
            # Iterate over positions and scores for pretty printing
            for index, (position, score) in enumerate(zip(positions, scores)):
                position_str = ', '.join(f"{val:.2f}" for val in position)  # Format position values to 2 decimal places
                print(f"{index:<5} {position_str:<60} {score:<10.2f}")

            print("="*80)
            position, score = self.optimizer.getResult()
            position_str = ', '.join(f"{val:.2f}" for val in position)  # Format position values to 2 decimal places
            print(f"Best {position_str:<60} {score:<10.2f}")
            
            rclpy.shutdown()  # shutdown ros node
            return
        
        #timeout
        self.startTime = time.time()
        
        self.sentPose = postPose(self.publisher, target, quat)
        
        self.pose = nextMove
        
        print('Best result:', self.optimizer.getResult())
        self.plotScore()
        
        #Autoloop --> debugging
        # time.sleep(20)
        # self.callback(np.random.rand(129, 512))
        
        

        
        return
        
def postPose(publisher,targets, quat):
    msg = PoseStamped()
    msg.header.frame_id = 'base_link'
    #Target coordinates
    msg.pose.position.x = targets[0]
    msg.pose.position.y = targets[1]
    msg.pose.position.z = targets[2]
    #Quaternion
    msg.pose.orientation.x = quat[1]
    msg.pose.orientation.y = quat[2]
    msg.pose.orientation.z = quat[3]
    msg.pose.orientation.w = quat[0]
    #Publish the target
    publisher.publish(msg)
    return msg

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
