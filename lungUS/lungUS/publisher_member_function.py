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
sys.path.append('/home/mateo-drr/Documents/Trento/ALU---Autonomous-Lung-Ultrasound')
import pathCalc as pc
import spatialmath as sm
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        #Create a publisher that sends PoseStamped messages to topic with queue 
        self.publisher_ = self.create_publisher(PoseStamped, 'target_frame', 5)
        #Create publisher to send all targets
        self.publisherAll = self.create_publisher(PoseArray, 'target_all', 5)
        #Create publisher to send probe targets
        self.publisherMoved = self.create_publisher(PoseArray, 'target_probe', 5)
        
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        
        flip=False #True if phantom's top points to the base of the robot
        
        config,scene = pc.askConfig()
        #rad, maxRot, stops, alpha = pc.askConfig(ask)
        
        if scene == 'curved':
            self.tcoord,self.trot = pc.curvedScene(config, flip)
            
        elif scene == 'linear':
            self.tcoord,self.trot = pc.linearScene(config, flip)
        
        self.targets,self.tmoved = pc.encodeStops(self.tcoord, self.trot, config['flangeOffset'])
        #self.tmoved = pc.encodeStops(self.moved, self.trot)
        self.stops = config['stopsL']+config['stopsW']
        self.config = config
        print(config)
        
        #Calculate Quaternions
        #Robot end-effector targets
        self.quat = []
        for targ in self.targets:
            self.quat.append(pc.getQuat(targ))
        #Probe targets
        self.quatmoved = []
        for targ in self.tmoved:
            self.quatmoved.append(pc.getQuat(targ))
        
        #Publish Pose Array
        msg = PoseArray()
        msg.header.frame_id = 'base_link'
        for i in range(len(self.targets)):
            pose = Pose()
            pose.position.x = self.targets[i].t[0]
            pose.position.y = self.targets[i].t[1]
            pose.position.z = self.targets[i].t[2]
            pose.orientation.x = self.quat[i][1]
            pose.orientation.y = self.quat[i][2]
            pose.orientation.z = self.quat[i][3]
            pose.orientation.w = self.quat[i][0]
            msg.poses.append( pose )
            
        self.publisherAll.publish(msg)
        
        #Publish probe pose array
        msg = PoseArray()
        msg.header.frame_id = 'base_link'
        for i in range(len(self.targets)):
            pose = Pose()
            pose.position.x = self.tmoved[i].t[0]
            pose.position.y = self.tmoved[i].t[1]
            pose.position.z = self.tmoved[i].t[2]
            pose.orientation.x = self.quatmoved[i][1]
            pose.orientation.y = self.quatmoved[i][2]
            pose.orientation.z = self.quatmoved[i][3]
            pose.orientation.w = self.quatmoved[i][0]
            msg.poses.append( pose )
            
        self.publisherMoved.publish(msg)
        
        #Subscriber
        self.subscription = self.create_subscription(
            PoseStamped,
            'target_frame',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        
        #Confirmation flag
        self.roger = True

    def timer_callback(self):
        
        print(self.roger, self.stops, self.i)
        # if not self.roger or self.i >= self.stops: #only send data after receiving confirmation
        #     return
        
        msg = PoseStamped()
        msg.header.frame_id = 'base_link'
        #Target coordinates
        msg.pose.position.x = self.targets[self.i].t[0]
        msg.pose.position.y = self.targets[self.i].t[1]
        msg.pose.position.z = self.targets[self.i].t[2]
        #Quaternion
        msg.pose.orientation.x = self.quat[self.i][1]
        msg.pose.orientation.y = self.quat[self.i][2]
        msg.pose.orientation.z = self.quat[self.i][3]
        msg.pose.orientation.w = self.quat[self.i][0]
        
        #Logging progress
        print('-'*10,'\nTarget is at:\n', self.tcoord[self.i],'\n',
              self.trot[self.i],'\n',
              self.quat[self.i],'\n',
              self.targets[self.i],
              '-'*10,'\n')
        
        #Publish the target
        self.publisher_.publish(msg)
        #self.get_logger().info(f'Publishing: {str(msg)}')
        
        #Stop counter
        self.i += 1
        if self.i >= self.stops:
            print('Scan completed')
            sys.exit(0)
        else:
            _ = input('Enter to continue')
            self.roger = False

    def listener_callback(self, msg):
        # self.get_logger().info('I heard: {str(msg)}')
        self.roger = True


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
