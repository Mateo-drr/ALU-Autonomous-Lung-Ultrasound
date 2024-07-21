function ros1msg = us_msg_msg_StampedArray_2To1_Converter(message,ros1msg)
%us_msg_msg_StampedArray_2To1_Converter passes data of ROS 2 message to ROS message.
% Copyright 2019 The MathWorks, Inc.    
ros1msg.Stamp.Sec = message.stamp.sec;
ros1msg.Stamp.Nsec = message.stamp.nanosec;
ros1msg.Array.Layout.Dim.Label = message.array.layout.dim.label{1};
ros1msg.Array.Layout.Dim.Size = message.array.layout.dim.size;
ros1msg.Array.Layout.Dim.Stride = message.array.layout.dim.stride;
ros1msg.Array.Layout.DataOffset = message.array.layout.data_offset;
end