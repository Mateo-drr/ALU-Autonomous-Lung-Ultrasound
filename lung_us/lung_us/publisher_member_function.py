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
import sys
sys.path.append('/home/mateo-drr/Documents/ALU---Autonomous-Lung-Ultrasound')
import pathCalc as pc
import ask
import spatialmath as sm
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from pprint import pprint
import numpy as np

debug=True
freq = 500 #hz
speed = 0.5 #m/s

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        #Create a publisher that sends PoseStamped messages to topic with queue 
        self.publisher = self.create_publisher(PoseStamped, 'target_frame', 5)
        #Create publisher to send all targets
        self.publisherAll = self.create_publisher(PoseArray, 'target_all', 5)
        #Create publisher to send probe targets
        self.publisherMoved = self.create_publisher(PoseArray, 'target_probe', 5)
        #Create publisher to send only targets not interpolation
        self.publisherClean = self.create_publisher(PoseStamped, 'target_clean', 5)
        #Create publisher to send all interpolated targets
        self.publisherInt = self.create_publisher(PoseArray, 'target_int', 5)
        
        timer_period = 1/freq  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.ip = 0
        self.tot = 0
        
        flip=False #True if phantom's top points to the base of the robot
        
        config,scene = ask.askConfig()
        #rad, maxRot, stops, alpha = pc.askConfig(ask)
        
        if scene == 'curved':
            self.tcoord,self.trot = pc.curvedScene(config, flip)
        elif scene == 'linear':
            self.tcoord,self.trot = pc.linearScene(config, flip)
        elif scene == 'rotation':
            self.tcoord,self.trot = pc.rotationScene(config,flip)
        
        self.tmoved,self.targets = pc.encodeStops(self.tcoord, self.trot, config['flangeOffset'])
        self.stops = config['stopsL']+config['stopsW']
        self.config = config
        
        pprint(config)
        
        #Calculate interpolation of position
        # self.tcoordInt,self.numint = pc.interpolateCoord(self.tcoord, speed, timer_period)
        # #Calculate quaternions
        # self.quat = pc.getQuat(self.targets)
        # #Calculate slerp
        # self.quatInt = pc.interpolateRot(self.quat, self.numint)
        
        # #flatten the list
        # self.tcoordInt = list(chain.from_iterable(self.tcoordInt))
        # #transform coordinates to se3 without rotation (not necessary)
        # _,self.targetsInt = pc.encodeStops(self.tcoordInt, None, config['flangeOffset'], rotation=False)
        
        #Calculate quaternion
        #Robot end-effector targets
        self.quat = pc.getQuat(self.targets)
        #Probe targets
        self.quatmoved = pc.getQuat(self.tmoved)
        
        if scene != 'rotation':
            self.targetsInt, self.quatInt, self.numint = pc.interpolateTargets(self.tcoord,
                                                                               speed,
                                                                               timer_period,
                                                                               self.targets,
                                                                               config['flangeOffset'])
        else:
            self.targetsInt, self.quatInt = self.targets, self.quat
            self.numint = np.ones(self.stops) #needed to make stop logic work
        
        #Publish Pose Array
        postPoseArray(self.publisherAll, self.targets, self.quat)
        #Publish probe pose array
        postPoseArray(self.publisherMoved, self.tmoved, self.quatmoved)
        #publish interpolation pose array
        postPoseArray(self.publisherInt, self.targetsInt, self.quatInt)
        
        #Subscriber
        self.subscription = self.create_subscription(
            PoseStamped,
            'target_frame',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        
        #Confirmation flag
        self.roger = True
        
        _= input("Enter to start")
        
        #Move to a initial pose (won't do anything if final pose was applied before)
        self.initPose = [pc.encodeStop(config['initCoord'], config['initRot'])]
        self.initQuat = [pc.getQuat(self.initPose)]
        postPose(self.publisher, self.initPose, self.initQuat, 0)
        
        _= input("Moving to intial position... Enter to continue")

    def timer_callback(self):
        
        if debug:
            print(self.roger, self.stops, self.i, self.ip, self.tot, self.numint, sum(self.numint))
        
        #if a target is reached
        if self.i == 0 or self.tot == sum(self.numint)-1 or self.ip == self.numint[self.i-1] : #self.ip == self.numint[self.i]-1:
            
            #post next checkpoint
            postPose(self.publisher, self.targetsInt, self.quatInt, self.tot)
            
            #post the next target information
            postPose(self.publisherClean, self.targets, self.quat, self.i)
            
            self.i += 1 #increase target counter
            self.ip = 0 #reset interpolation counter
            
            _ = input('Enter to continue')
        
        else:
            
            #post next checkpoint
            postPose(self.publisher, self.targetsInt, self.quatInt, self.tot)
        
        #Logging progress
        if debug:
            try:
                print('-'*10,'\nTarget is at:\n', self.tcoord[self.i],'\n',
                      self.trot[self.i],'\n',
                      self.quat[self.i],'\n',
                      self.targets[self.i],
                      '-'*10,'\n')
            except:
                pass
            
        # #Stop counter
        # self.i += 1 
        #Interpolation counter
        self.ip += 1
        #Loop counter
        self.tot += 1
        
        if True:#(self.i-1)%(self.config['numInt']+1) == 0: #skip interpolated targets
    
            if self.i >= self.stops:
                print('Moving to the final stop')
                
                _=input('Enter to finish')
                postPose(self.publisher, self.initPose, self.initQuat, 0)
                print('Moving to resting position')
                sys.exit(0)
            # else:
            #     _ = input('Enter to continue')
            #     self.roger = False

    def listener_callback(self, msg):
        # self.get_logger().info('I heard: {str(msg)}')
        self.roger = True

def postPoseArray(publisher,targets,quat):
    #Publish Pose Array
    msg = PoseArray()
    msg.header.frame_id = 'base_link'
    for i in range(len(targets)):
        pose = Pose()
        pose.position.x = targets[i].t[0]
        pose.position.y = targets[i].t[1]
        pose.position.z = targets[i].t[2]
        pose.orientation.x = quat[i][1]
        pose.orientation.y = quat[i][2]
        pose.orientation.z = quat[i][3]
        pose.orientation.w = quat[i][0]
        msg.poses.append( pose )
    publisher.publish(msg)
    
def postPose(publisher,targets,quat,i):
    msg = PoseStamped()
    msg.header.frame_id = 'base_link'
    #Target coordinates
    msg.pose.position.x = targets[i].t[0]
    msg.pose.position.y = targets[i].t[1]
    msg.pose.position.z = targets[i].t[2]
    #Quaternion
    msg.pose.orientation.x = quat[i][1]
    msg.pose.orientation.y = quat[i][2]
    msg.pose.orientation.z = quat[i][3]
    msg.pose.orientation.w = quat[i][0]
    #Publish the target
    publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
