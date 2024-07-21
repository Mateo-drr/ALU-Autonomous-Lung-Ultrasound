function ros2msg = us_msg_msg_StampedArray_1To2_Converter(message,ros2msg)
%us_msg_msg_StampedArray_1To2_Converter passes data of ROS message to ROS 2 message.
% Copyright 2019 The MathWorks, Inc.
ros2msg.stamp.sec = message.Stamp.Sec;
ros2msg.stamp.nanosec = message.Stamp.Nsec;
ros2msg.array.layout.dim.label = message.Array.Layout.Dim.Label;
ros2msg.array.layout.dim.size = message.Array.Layout.Dim.Size;
ros2msg.array.layout.dim.stride = message.Array.Layout.Dim.Stride;
ros2msg.array.layout.data_offset = message.Array.Layout.DataOffset;
end