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

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        #Create a publisher that sends PoseStamped messages to topic with queue 
        self.publisher_ = self.create_publisher(PoseStamped, 'target_frame', 5)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        
        ask=False
        middlepoint = (-25,50,10) #position of the point of interest [cm] x,z,y
        flip=False #True if phantom's top points to the base of the robot
        rad, maxRot, stops, alpha = pc.askConfig(ask)
        
        if alpha ==0: #ie needs to be calculated
            alpha = pc.calcAlpha(stops,maxRot)

        pitsA,stops = pc.pitStopsAng(alpha,maxRot,rad)
        c1,r1 = pc.projPath3dAng(pitsA,middlepoint,rad,path='length',flip=flip)
        #pitsA,stops = pc.pitStopsAng(alpha,maxRot,rad) 
        c2,r2 = pc.projPath3dAng(pitsA,middlepoint,rad,path='width',flip=flip)
        
        self.tcoord,self.trot = c1+c2,r1+r2
        self.targets = pc.encodeStops(self.tcoord, self.trot)
        self.stops = stops
        
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
        if not self.roger or self.i >= self.stops: #only send data after receiving confirmation
            return
        
        msg = PoseStamped()
        msg.header.frame_id = 'base_link'
        #Target coordinates
        msg.pose.position.x = self.tcoord[self.i][0]
        msg.pose.position.y = self.tcoord[self.i][1]
        msg.pose.position.z = self.tcoord[self.i][2]
        #Quaternion
        quat = pc.getQuat(self.targets[self.i])
        msg.pose.orientation.x = quat[1]
        msg.pose.orientation.y = quat[2]
        msg.pose.orientation.z = quat[3]
        msg.pose.orientation.w = quat[0]
        print('Target is at:', self.tcoord[self.i],'\n',
              self.trot[self.i],'\n',
              quat,'\n',
              self.target[self.i])
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {str(msg)}')
        self.i += 1
        _ = input('Anything to continue')
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
