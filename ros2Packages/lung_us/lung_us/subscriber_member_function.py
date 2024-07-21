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
from us_img.msg import StampedArray

# #import matplotlib.pyplot as plt
# import cv2
# import cv2.aruco as aruco
import numpy as np

# #aruco is at 0.45,0.4,0.89 m from robot base

# sim = False

# def image_msg_to_numpy(data):
#     fmtString = data.encoding
#     if fmtString in ['mono8', '8UC1', 'bgr8', 'rgb8', 'bgra8', 'rgba8']:
#         img = np.frombuffer(data.data, np.uint8)
#     elif fmtString in ['mono16', '16UC1', '16SC1']:
#         img = np.frombuffer(data.data, np.uint16)
#     elif fmtString == '32FC1':
#         img = np.frombuffer(data.data, np.float32)
#     else:
#         print('image format not supported:' + fmtString)
#         return None

#     depth = data.step / (data.width * img.dtype.itemsize)
#     if depth > 1:
#         img = img.reshape(data.height, data.width, int(depth))
#     else:
#         img = img.reshape(data.height, data.width)
#     return img

# class MinimalSubscriber(Node):

#     def __init__(self):
#         super().__init__('minimal_subscriber')
#         print('Started listener...')
        
#         if sim:
#             self.subscription = self.create_subscription(Image,'/image',self.rgb_callback,2)
#             self.dsub = self.create_subscription(Float32MultiArray,'/depth',self.d_callback,2)
#         else:
#             self.subscription = self.create_subscription(Image,'/camera/camera/color/image_raw',self.rgb_callback,2)
#             self.dsub = self.create_subscription(Image,'/camera/camera/depth/image_rect_raw',self.d_callback,2)
        
#         # prevent unused variable warning
#         self.subscription  
#         self.dsub
        
#         self.calcPos=True

#     def rgb_callback(self, msg):
#         self.get_logger().info('I heard')
        
#         if sim:
#             img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.width, msg.height, 3)
#         else:
#             img = image_msg_to_numpy(msg)
            
#         print('saving rgb')
#         np.save('rgbimg.npy',img)
#         print('saved rgb')
#         #img = np.rot90(img)
#         bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         cv2.imshow('Received Image', bgr_image)
#         cv2.waitKey(1)
        
#         if self.calcPos:
#             arucoEst(bgr_image)
#             self.calcPos=False

#     def d_callback(self, msg):
#         self.get_logger().info('depth')
#         if sim:
#             img = np.array(msg.data, dtype=np.float32).reshape((msg.layout.dim[0].size, msg.layout.dim[1].size))
#         else:
#             img = image_msg_to_numpy(msg)
#         print(img.shape)
#         #img = np.rot90(dpt)
#         print(img.min(),img.max())
        
#         print('saving depth')
#         np.save('dimg.npy',img)
#         print('saved depth')
#         _=input('aa')
#         #img = (img - img.max())*-1
#         #print(img)
        
#         #scale to plot
#         simg = (img - img.min()) / (img.max() - img.min())
#         simg = cv2.applyColorMap(cv2.convertScaleAbs(simg,  alpha=255.0), cv2.COLORMAP_JET)
#         cv2.imshow('Received Depth', simg)
#         cv2.waitKey(1)

# def invert_pose(rvec, tvec):
#     R, _ = cv2.Rodrigues(rvec)
#     print("R:\n", R)
#     R_inv = R.T
#     print("R_inv:\n", R_inv)
    
#     # Ensure tvec is a column vector
#     tvec = tvec.reshape(-1, 1)
#     print("tvec (reshaped):\n", tvec)
    
#     tvec_inv = -R_inv @ tvec
#     rvec_inv, _ = cv2.Rodrigues(R_inv)
#     print("tvec_inv:\n", tvec_inv)
#     return rvec_inv, tvec_inv

# def arucoEst(img):
#     #  Convert the image to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Define the dictionary used to detect the markers
#     dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
#     # Create DetectorParameters object
#     parameters = cv2.aruco.DetectorParameters()
    
#     # Create ArucoDetector object
#     detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    
#     # Detect the markers
#     markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(gray)
#     # print('Detected corners:', markerCorners)
#     # print('Detected IDs:', markerIds)
#     # print('Rejected candidates:', rejectedCandidates)
    
#     # Draw detected markers on the image
#     if markerIds is not None:
#         for corners, markerId in zip(markerCorners, markerIds):
#             corners = corners.reshape((4, 2)).astype(int)
#             for corner in corners:
#                 cv2.circle(img, tuple(corner), 5, (0, 255, 0), -1)
#         cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIds)
#     else:
#         print("No markers detected or recognized.")
    
#     # Draw rejected candidates
#     # if rejectedCandidates is not None:
#     #     for candidate in rejectedCandidates:
#     #         candidate = candidate.reshape((4, 2)).astype(int)
#     #         for corner in candidate:
#     #             cv2.circle(img, tuple(corner), 5, (0, 0, 255), -1)
#     #     cv2.aruco.drawDetectedMarkers(img, rejectedCandidates, borderColor=(0, 0, 255))

#     # Display the image
#     cv2.imshow("Detected ArUco Markers", img)
#     cv2.waitKey(1)
#     cv2.destroyAllWindows()
    
#     #Calculate camera params
#     fov = np.deg2rad(60)
#     #(w,h,c)
#     width,height=img.shape[:-1]
#     # Calculate the focal lengths
#     fx = width / (2 * np.tan(fov / 2))
#     fy = height / (2 * np.tan(fov / 2))
#     # Calculate the optical center (assuming the center of the image)
#     cx = width / 2
#     cy = height / 2
#     # Camera matrix
#     camera_matrix = np.array([
#         [fx, 0,  cx],
#         [0,  fy, cy],
#         [0,  0,  1]
#     ])
#     print("Camera matrix:\n", camera_matrix)
#     #Distorsion simulated is 0 so
#     dist_coeffs = np.zeros(5)  # Assuming no distortion
    
    
#     #CAMERA POSE ESTIMATION
#     if markerIds is not None:
#         rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 0.1, camera_matrix, dist_coeffs)  # Assuming marker size is 0.05m (5cm)
#         for i in range(len(markerIds)):
#             rvec_marker = rvecs[i]
#             tvec_marker = tvecs[i]
#             print(rvec_marker,tvec_marker)
#             rvec_camera, tvec_camera = invert_pose(rvec_marker, tvec_marker)
#             print(f"Camera pose relative to marker ID {markerIds[i]}: rvec: {rvec_camera.flatten()}, tvec: {tvec_camera.flatten()}")

#     return

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        print('Started listener...')
        
        self.subscription = self.create_subscription(StampedArray,
                                                     '/imgs',
                                                     self.callback,2)
        
        self.subscription  

    def callback(self, msg):
        self.get_logger().info('image')
        
        img = np.array(msg.data)
        
        return
        


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
