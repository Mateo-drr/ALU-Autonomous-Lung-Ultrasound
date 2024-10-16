close all
clc 
clear all

% Add path to the ULA-OP program files
addpath("C:\Program Files (x86)\ULA-OP\MATLAB\MatLink\")

nlines = 129;
ysize = 6292;
fs = 50e6;
lcut = 5e6;
hcut = 7e6;

global imageRequested;
imageRequested=false;

simulate=false;

% ROS2 Initialization
% rosinit;
% Create ROS2 node
node = ros2node("/matlab");
msgType = 'us_msg/StampedArray'; % Custom message type
% Create publisher
publisher = ros2publisher(node, '/imgs', msgType);
% Create subscriber to listen for image request
boolSub = ros2subscriber(node, '/req_img', 'std_msgs/Bool', @imageRequestCallback);

% Initialize ROS2 timestamp reference
t0 = datetime(1970, 1, 1, 0, 0, 0, 'TimeZone', 'UTC+2');

% US Acquisition Configuration
exePath = 'C:\Program Files (x86)\ULA-OP\Applicazione';
svPath = 'C:\Users\Medical Robotics\Documents\imgs';
pathPrefix = 'Matlink';
Link = UOLink(exePath, svPath, pathPrefix);
config = 'C:\Users\Medical Robotics\Documents\ALU-Autonomous-Lung-Ultrasound\matlab\B-Mode Matlink\B_Config.cfg';
probe = 'C:\ProgramData\ULA-OP\Probe\probeLA533.wks';
DSN = 'SliceRf';
toggle = 0;

if ~simulate
    r = Link.Open(config, probe);

    % AutoSave and Freeze
    
    r = Link.AutoSave(1, toggle);
    r = Link.Freeze(0);
end

toggle = 1 - toggle;

% Waiting for image request
disp('Waiting for image request...');

% Main loop
while true

    %continiously acquire new images and save them
    if ~simulate
        % Acquire the image
        r = Link.WaitSave(5000);
    
        if r ~= 0
            disp('Error while waiting for image save. Exiting...');
            break;
        end
    
        r = Link.AutoSave(1, toggle);
        r = Link.Freeze(0);
    end
    toggle = 1 - toggle;

    % Check if a new image is requested
    if imageRequested

        % Get the acquired data
        FilePath = [svPath, '\' pathPrefix, sprintf('_%03d_', toggle), DSN];
        UosStrings = importdata([FilePath, '.uos']);
        disp(FilePath)
        y = Link.GetAcq(toggle, DSN, nlines, UosStrings);
        
        % Plot the US image
        dims = size(y);
        if length(dims) == 3
            y = y(:, :, 1);
        end
        y = y(1:ysize, :);

        % Filtering
        nyquist = fs / 2;
        low_cutoff_norm = lcut / nyquist;
        high_cutoff_norm = hcut / nyquist;
        [b, a] = butter(10, [low_cutoff_norm, high_cutoff_norm], 'bandpass');
        
        filtered_img = zeros(size(y));
        for col = 1:size(y, 2)
            column_data = double(y(:, col));
            filtered_img(:, col) = filtfilt(b, a, column_data);
        end
        
        % Resize for visualization and sending
        rimg = imresize(filtered_img, [512, 128]);
        
        % Send image through ROS2
        msg = ros2message(publisher);
        msg.array.data = single(rimg);
        
        % Add timestamp
        currentTime = datetime('now', 'TimeZone', 'UTC+2');
        sec = seconds(currentTime - t0);
        rtime = ros2time(sec);
        msg.stamp = rtime;
        
        % Send the message
        send(publisher, msg);
        disp('Image sent.');

        imageRequested = 0;
    end
    pause(0.1); % Adjust pause duration if necessary
end

% Close the link after loop
Link.Close()

% Callback function for image request
function imageRequestCallback( msg)
    global imageRequested; % Use a global variable to store request state
    disp(msg.data)
    imageRequested = msg.data; % Update the global variable with the request state
end