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
# from skimage.transform import resize
from scipy.ndimage import laplace
import pickle
# from sklearn.preprocessing import MinMaxScaler
import threading
import copy
import os
import json

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
        self.desiredPose=None
        self.cpos = None #get the current position of the robot where the image was taken
        self.livePose=None
        self.sentPose=None
        
        self.reqImage = self.create_publisher(Bool,
                                              '/req_img',
                                              2)
        
        #plExtractor load weights
        self.model = plExtractor('cpu')
        self.model.eval()
        self.model.load_state_dict(torch.load(modelPath / 'model.pth',
                                              map_location=torch.device('cpu'),
                                              weights_only=True))
        
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
        self.maxit=20
        self.threshold=0.74
        self.zoffset = 0.017 #[m]
        
        #load scaler
        # with open(basePath / 'minmax_scaler_4f.pkl', 'rb') as f:
        # with open(basePath / 'minmax_scaler_3f.pkl', 'rb') as f:
        # with open(basePath / 'minmax_scaler_3fn.pkl', 'rb') as f:
        # with open(basePath / 'minmax_scaler_4fn.pkl', 'rb') as f:
        # with open(basePath / 'qtscalerM.pkl', 'rb') as f:
        # with open(basePath / 'powerScaler.pkl', 'rb') as f:
        with open(basePath / 'qtS.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        # self.scaler = MinMaxScaler()
            
        self.startTime = None
        
        self.smax=None
        self.smin=None
        self.cmax=None
        self.cmin=None
        
        self.cmapRes=None
        
        self.variables={'images':{'img shape':[],
                                  'ML model mask':[],
                                  'crop shape':[],
                                  'cmap shape':[]},
                        'features':{'raw':[],
                                    'log':[],
                                    'scaled':[],
                                    'weights':[],
                                    'bias':[],
                                    'weighted':[],
                                    'conf score cmap':[],
                                    'conf score lapv':[],
                                    'conf score join':[],
                                    'cost features':[],
                                    'cost penalty':[],
                                    'cost used':[]},
                        'positions':{'desired':[],
                                     'actual':[],
                                     'clipped':[],
                                     'next':[]},
                        'scores':[],
                        'stopcrit':[],
                        'results':{'all':[],
                                   'best':None}}
        
        
        #Wait for user input
        _=input('Enter to begin')
        #Ask for initial image
        self.askImg(True)
        
        #simulation
        self.debug = False
        if self.debug:
            self.callback(np.random.rand(512,128))
        
    def cmapThread(self, img):
        cmap = confidenceMap(img,rsize=False) #cropping top pixels with noise
        # cmap = resize(cmap, (img.shape[0], img.shape[1]), anti_aliasing=True)
        self.cmapRes = cmap
        
    def checkDif(self):
        _, scores = self.optimizer.get_tested_positions_and_scores()
        
        if len(scores) >= 5:
            last = np.array(scores[-5:])
            grad = np.diff(last)
            self.variables['stopcrit'].append([grad,grad.mean(),last.mean()])
            print(self.variables['stopcrit'][-1])
            if grad.mean() <= 0.02 and last.mean() < -0.5:
                return True
        return False

    def costFunc(self, pos):
        
        img= self.img
        #TODO send to device 
        #Send the image to the trained model
        print(img.shape, 'image shape')
        self.variables['images']['img shape'].append(img.shape)
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
        self.variables['images']['ML model mask'].append(mask.cpu().detach().numpy().tolist())
        
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
    
            top,btm=30,-300 
            pleura = img[top:btm,:]
        except:
            top,btm=30,-300 #dont crop
            pleura = img[top:btm,:]
            print('segmentation failed')
            
        self.variables['images']['crop shape'].append(pleura.shape)
            
        # Launch cmap calculation in a separate thread
        self.cmap_thread = threading.Thread(target=self.cmapThread, args=(img[10:-250,:],))
        self.cmap_thread.start()
        
        if self.plotting:
            self.plotUS(pleura, name='crop', save=True)
        
        #######################################################################
        #FEATURES
        #######################################################################
        '''
        Variance
        '''
        #envelope of whole image
        hb = byb.envelope(img)
        #normalization of each line
        himgCopy = hb.copy()
        for k in range(hb.shape[1]):
            line = hb[:,k]
            
            min_val = np.min(line)
            max_val = np.max(line)
            line = (line - min_val) / (max_val - min_val)
            
            himgCopy[:,k] = line
        #crop the image
        imgCrop = himgCopy[top:btm,:]
        #Join lines
        yhist = np.mean(imgCrop,axis=1)
        #variance
        varyhist = np.var(yhist)
        '''
        Luminosity
        '''
        hbCrop = hb[top:btm,:]
        lum = np.mean(abs(hbCrop))
        '''
        Laplace var
        '''
        logHimgCrop = 20*np.log1p(abs(hbCrop))
        lap = laplace(logHimgCrop).var()
        '''
        Confidence var
        '''
        # Wait for the cmap calculation to finish before using it
        self.cmap_thread.join()
        
        if self.cmapRes is not None:
            cmap = self.cmapRes[20:-50,:]
            print(cmap.shape, 'cmap shape')
            cyhist,_ = byb.getHist(cmap)
            print(cyhist.shape, 'cmap hist')
            # vardif = np.var(np.diff(cyhist))
            vardif = np.mean(np.abs(np.diff(cyhist)))
            if self.plotting:
                self.plotUS(cmap,norm=False,name='cmap', save=True)
            self.cmapRes = None
            #variance of laplacian of confidence map
            confl = laplace(cmap).var()
            
            self.variables['images']['cmap shape'].append(cmap.shape)
        
        #######################################################################
        features = [varyhist, vardif, lum, lap] #keep order as yhistvar, cmap, lum lap!
        self.variables['features']['raw'].append(features)
        #######################################################################
        
        print('\nRAW feat', np.round(features, 6))
        
        logfeat = np.log1p(features)
        self.variables['features']['log'].append(logfeat)
        print('log feat', np.round(logfeat, 6))
        
        normFeat = self.scaler.transform([features])[0]
        self.variables['features']['scaled'].append(normFeat)
        
        #qt scaler
        w = [ 1.50142378, -0.50806179,  0.18003294, -0.0502186 ]
        
        #intercept
        bias = -0.15638810337191103
        
        self.variables['features']['weights'].append(w)
        self.variables['features']['bias'].append(bias)
        
        print('norm feat', np.round(normFeat,6))
        
        linmod = [x * y for x, y in zip(normFeat, w)]
        print('weig feat', np.round(linmod,6))
        self.variables['features']['weighted'].append(linmod)
        
        cost = -(np.sum(linmod) + bias)
        
        #Confidence maximization
        threshold = 0.85
        confC = np.sum(cmap < threshold)
        # Normalize the high confidence count by the total number of pixels
        confRT = confC / cmap.size
        #lapv score
        confLap=-np.log(confl)/10
        
        confRT2 = abs(confRT+ confLap)/2
        print('Inv Confidence:', round(confRT,3))
        print('Lap Confidence:', round(confLap,3))
        print('L+C Confidence:', round(confRT2,3))
        self.variables['features']['conf score cmap'].append(confRT)
        self.variables['features']['conf score lapv'].append(confLap)
        self.variables['features']['conf score join'].append(confRT2)
        
        print('COST',round(cost,4),'\n')
        
        self.variables['features']['cost features'].append(cost)
        
        if confRT2 <= self.threshold:
            print('Using features')
            cost = cost
        else:
            print('Penalty cmap')
            cost = confRT2 - self.threshold#- 2.2    
        
        self.variables['features']['cost penalty'].append(confRT2 - self.threshold)
        self.variables['features']['cost used'].append(cost)
        
        return cost
    
    def plotUS(self,raw,name='Image',save=False,norm=True):
        img = copy.deepcopy(raw)
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
            os.makedirs(basePath.parent / 'runData/', exist_ok=True)

            cv2.imwrite(basePath.parent / 'runData' / f'{name}{self.counter}.png', img_colored)
            np.save(basePath.parent / 'runData' / f'{name}Raw{self.counter}.npy', raw)
            
        
    def plotScore(self,save=False):
        # Assuming self.optimizer.get_tested_positions_and_scores() returns a tuple of (positions, scores)
        positions, scores = self.optimizer.get_tested_positions_and_scores()
    
        # Check if scores is empty
        if not scores:
            print("No scores to plot.")
            return
    
        self.variables['scores'].append([positions[-1],scores[-1]])
    
        # Define the canvas size for plotting
        width, height = 800, 600
        margin = 50  # margin around the plot area
        plot_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # white canvas
    
        # Define the plot area size
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin
    
        # Set fixed minimum and maximum values for the y-axis
        min_score = min(scores) if scores else 0 # Fixed minimum y-axis value
        max_score = max(scores) if scores else 1  # Maximum y-axis value is the max score or at least 1
    
        # Draw the axes
        cv2.line(plot_img, (margin, margin), (margin, height - margin), (0, 0, 0), 2)  # y-axis
        cv2.line(plot_img, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 2)  # x-axis
    
        # Draw the axes labels
        num_ticks = 5
        # Y-axis labels
        for i in range(num_ticks + 1):
            y_pos = height - margin - int(i / num_ticks * plot_height)
            score_value = min_score + (max_score - min_score) * i / num_ticks
            cv2.putText(plot_img, f"{score_value:.2f}", (margin - 40, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.line(plot_img, (margin - 5, y_pos), (margin + 5, y_pos), (0, 0, 0), 1)
    
        # X-axis labels
        for i in range(num_ticks + 1):
            x_pos = margin + int(i / num_ticks * plot_width)
            position_value = int(i / num_ticks * (len(scores) - 1))
            cv2.putText(plot_img, f"{position_value}", (x_pos - 10, height - margin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.line(plot_img, (x_pos, height - margin - 5), (x_pos, height - margin + 5), (0, 0, 0), 1)
    
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
        
        if save:
            os.makedirs(basePath.parent / 'runData/', exist_ok=True)
            cv2.imwrite(basePath.parent / 'runData' / f'score{self.counter}.png', plot_img)
        # cv2.destroyAllWindows()

    def storePose(self,msg):
        self.livePose=msg
        
        if self.sentPose is not None: #check if a target was sent
            if self.has_reached_target(msg, self.sentPose) or (self.startTime is not None and (time.time() - self.startTime) > 5): 
                print(f'\nTime: {time.time() - self.startTime} [s]')
                
                self.variables['positions']['desired'].append(self.formatPose(self.sentPose))
                
                #Formatting the pose
                self.cpos = self.formatPose(self.getPose())
                self.variables['positions']['actual'].append(self.cpos)
                #Clipping it within bounds
                self.cpos = list(np.clip(self.cpos, self.lower_bounds, self.upper_bounds))
                self.variables['positions']['clipped'].append(self.cpos)
                
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
        
        print(f"\rPosition error (x, y, z): {position_error[0]:.4f}, {position_error[1]:.4f}, {position_error[2]:.4f}, Orientation error: {orientation_diff:.5f}", end=' ')
        
        # Check if each position axis is within the corresponding tolerance
        position_reached = all(position_error < np.array(position_tolerance))
        
        # Check if orientation is within tolerance
        orientation_reached = orientation_diff < orientation_tolerance
        
        # If both position and orientation are within tolerance
        if position_reached and orientation_reached:
            print('\nReached Target!')

            return True
        return False

        
    def askImg(self, ask):
        print('Asking for an image', ask)
        msg = Bool()
        msg.data = ask
        self.reqImage.publish(msg)
        
    def getPose(self):
        
        if self.debug:
            msg = PoseStamped()
            msg.header.frame_id = 'base_link'
            #Target coordinates
            msg.pose.position.x = 0.0
            msg.pose.position.y = 0.0
            msg.pose.position.z = 0.0
            #Quaternion
            msg.pose.orientation.x = 0.5
            msg.pose.orientation.y = 0.5
            msg.pose.orientation.z = 0.5
            msg.pose.orientation.w = 0.5
            self.livePose = msg
        
        elif self.livePose is None:
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
    def rpy2quat(self,rpy):
        rotmat = pc.rot2SE3(rpy[0],rpy[1],rpy[2],unit='deg')
        quat = pc.getQuat(rotmat)
        return quat
    
    def callback(self, msg):
        self.get_logger().info('image')
        
        #Stop asking for an image
        self.askImg(False)
        
        #Receive and rebuild the image
        if self.debug:
            img = msg
        else:
            img = np.array(msg.array.data)
        print(img.shape, 'received data shape')
        img = img.reshape((128,512)).transpose()
        print(img.shape, 'reshaped data')
        
        # img = self.preproc(img)
        
        if self.plotting:
            self.plotUS(img,save=True, name='Us')
        
        # _=input()
        
        self.img = img
        
        #Define the search space around the initial position
        if self.first:
            
            # print(self.getPose())
            
            while (self.cpos is None):
                self.cpos = self.formatPose(self.getPose())
            
            print('current Pose:', self.cpos)
            self.searchSpace = [
                                Real(self.cpos[0] - 2.0, self.cpos[0] + 2.0),  # x
                                Real(self.cpos[1] - 0.5, self.cpos[1] + 2.0),  # y
                                Real(self.cpos[2] - 0.01, self.cpos[2] + (0.7)),  # z with deeper targets
                                Real(self.cpos[3] - 15.0, self.cpos[3] + 15.0),  # r1
                                Real(self.cpos[4] - 15.0, self.cpos[4] + 15.0),  # r2
                                Real(self.cpos[5] - 20.0, self.cpos[5] + 20.0)   # r3
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
            self.optimizer = ManualGPMinimize(self.costFunc,
                                              self.searchSpace,
                                              # initial_point=self.cpos,
                                              n_initial_points=3, #points before optimizing
                                              n_restarts_optimizer=2,
                                              n_jobs=-1, #num cores to run optim
                                              #acq_func='EI',
                                              # xi=0.01+0.04,
                                              # kappa=1.96-0.5
                                              verbose=True,
                                              n_points = 10000, # number of predicted points, from which one is taken 
                                              #n_calls number of iterations (cost func calls) (done manually here)
                                              )
            
            
            self.first=False
        
        # else:
        # nextMove = self.optimizer.step()
        print('='*80)
        print('desired pose')
        pprint(self.desiredPose)
        print('Reached pose')
        pprint(self.cpos)
        print('bounds')
        print(self.lower_bounds)
        print(self.upper_bounds)
        print('='*80)
        
        cost = self.costFunc(self.cpos) #check score of last suggested position
        self.optimizer.update(self.cpos, cost)  # Update the optimizer
        nextMove = self.optimizer.step() #get new move
        self.variables['positions']['next'].append(nextMove)
        print(f'\nUpdating cost {cost:.3f} at position {self.cpos}')
        
        
        # print('Next move [cm]:')
        # pprint(nextMove)
        
        quat = self.rpy2quat(nextMove[-3:])
        print('Quaternions', quat)
        
        target = list(np.array(nextMove)*0.01)
        
        self.plotScore()
        
        print('enter to move or f to finish ')#, target[0:3], nextMove[3:])
        user_input = None#input("").strip().lower()  # Get user input
        
        if (self.counter is not None and self.counter > self.maxit) or user_input == 'f' or self.checkDif():
            
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
                position_str = ', '.join(f"{val:.4f}" for val in position)  # Format position values to 2 decimal places
                print(f"{index:<5} {position_str:<60} {score:<10.4f}")
                
                self.variables['results']['all'].append({
                                                        'index': index,
                                                        'position': position, 
                                                        'score': score
                                                        })  

            print("="*80)
            position, score = self.optimizer.getResult()
            position_str = ', '.join(f"{val:.4f}" for val in position)  # Format position values to 2 decimal places
            print(f"Best {position_str:<60} {score:<10.4f}")
            
            self.variables['results']['best'] = {
                                                'position': position, 
                                                'score': score
                                                }
            
            
            print('Optimization Done, moving to best found point:')
            position, score = self.optimizer.getResult()
            
            quat = self.rpy2quat(position[-3:])
            print('Quaternions', quat)
            
            target = list(np.array(position[:3])*0.01)
            
            #Plot the final score plot and save it
            self.plotScore(save=True)
            
            # Writing to a JSON file
            # print(self.variables)
            self.variables = byb.convert_to_native_types(self.variables)
            os.makedirs(basePath.parent / 'runData/', exist_ok=True)
            with open(basePath.parent / 'runData' / 'variables.json', 'w') as f:
                json.dump(self.variables, f)
            
            #stop receiving live pose
            self.destroy_subscription(self.poseNode)
            
            #send data
            self.sentPose = postPose(self.publisher, target, quat, zoffset=self.zoffset)
            
            time.sleep(2)
            sys.exit(0)
            
            return
        
        #Positions counter
        self.counter += 1
        
        #timeout
        self.startTime = time.time()
        
        self.sentPose = postPose(self.publisher, target, quat, zoffset=self.zoffset)
        
        self.desiredPose = nextMove
        
        # print('Best result:', self.optimizer.getResult())
        
        
        #Autoloop --> debugging
        if self.debug:
            time.sleep(0)
            self.callback(np.random.rand(128, 512))
        
        return
    
def postPose(publisher,targets, quat, zoffset=0):
    msg = PoseStamped()
    msg.header.frame_id = 'base_link'
    #Target coordinates
    msg.pose.position.x = targets[0]
    msg.pose.position.y = targets[1]
    msg.pose.position.z = targets[2]+ zoffset #[m]
    #Quaternion
    msg.pose.orientation.x = quat[1]
    msg.pose.orientation.y = quat[2]
    msg.pose.orientation.z = quat[3]
    msg.pose.orientation.w = quat[0]
    #Publish the target
    publisher.publish(msg)
    
    #remove offset
    msg.pose.position.z -= zoffset
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
