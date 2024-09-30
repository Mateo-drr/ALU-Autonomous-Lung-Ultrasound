function [data, info] = stampedArray
%StampedArray gives an empty data for us_msg/StampedArray
% Copyright 2019-2021 The MathWorks, Inc.
data = struct();
data.MessageType = 'us_msg/StampedArray';
[data.stamp, info.stamp] = ros.internal.ros2.messages.builtin_interfaces.time;
info.stamp.MLdataType = 'struct';
[data.array, info.array] = ros.internal.ros2.messages.std_msgs.float32MultiArray;
info.array.MLdataType = 'struct';
info.MessageType = 'us_msg/StampedArray';
info.constant = 0;
info.default = 0;
info.maxstrlen = NaN;
info.MaxLen = 1;
info.MinLen = 1;
info.MatPath = cell(1,11);
info.MatPath{1} = 'stamp';
info.MatPath{2} = 'stamp.sec';
info.MatPath{3} = 'stamp.nanosec';
info.MatPath{4} = 'array';
info.MatPath{5} = 'array.layout';
info.MatPath{6} = 'array.layout.dim';
info.MatPath{7} = 'array.layout.dim.label';
info.MatPath{8} = 'array.layout.dim.size';
info.MatPath{9} = 'array.layout.dim.stride';
info.MatPath{10} = 'array.layout.data_offset';
info.MatPath{11} = 'array.data';
