close all
clc 
clear all

%% ROS2
%Create cstm ros msg type
%ros2genmsg(p)
%call C:\dev\ros2_humble\local_setup.bat
%Create node
node = ros2node("/matlab");
msgType = 'us_msg/StampedArray';%'std_msgs/Float32MultiArray';
%Create publisher
publisher = ros2publisher(node,'/imgs',msgType);
%Initial date
t0 = datetime(1970, 1, 1, 0, 0, 0, 'TimeZone', 'UTC+2');

%% US Aqcuisition

%Define executable path
exePath = 'C:\Program Files (x86)\ULA-OP\Applicazione';

%Define path to save the images
svPath = 'C:\Users\Medical Robotics\Documents\imgs';

%File names used to store
pathPrefix = 'Matlink';

%Create the link
Link = UOLink(exePath, svPath, pathPrefix);

%Path to config file
%config = 'C:\ProgramData\ULA-OP\Mode\LA533 Modes (MSDLab Default)\B-Mode\B_Config.cfg';
config = 'C:\Users\Medical Robotics\Documents\B-Mode Matlink\B_Config.cfg';

%Path to probe file
probe = 'C:\ProgramData\ULA-OP\Probe\probeLA533.wks';

%Open the ulaop app
r = Link.Open(config, probe);

%Name of the slice, also used for reading the file
%DNS = 'SliceRf';
DSN = 'SliceIQ';
%Number of lines 
nlines = 129;

%File number
count = 0;


num_iterations = 1000000;  % Number of iterations in your loop

toggle = 0;

r = Link.AutoSave(1, toggle);
r = Link.Freeze(0);

toggle = 1-toggle;

% h = figure(1);
while(r == 0)
                
        r = Link.WaitSave(5000);
        
        if(r ~= 0)
            return;
        end
        
        r = Link.AutoSave(1, toggle);
        r = Link.Freeze(0);
        
        toggle = 1-toggle;

        if(count == 0)
            tog = sprintf('_%03d_', toggle);
            FilePath = [svPath, '\' pathPrefix , tog, DSN];
            UosStrings = importdata([FilePath , '.uos']);
        end

        y = Link.GetAcq(toggle, DSN, nlines, UosStrings);
        
        %% ROS2 send message
        %Create message var
        msg = ros2message(publisher);
        %msg.layout.dim.label = 'test';
        %msg.data = single(y);
        msg.array.data = single(real(y));
        
        % %Add timestamp
        currentTime = datetime('now', 'TimeZone', 'UTC+2');
        % % Convert MATLAB time to ROS 2 time
        sec = seconds(currentTime - t0); % seconds since Unix epoch
        rtime = ros2time(sec);
        % % Assign the header to the message
        msg.stamp = rtime;
        %Send message
        send(publisher,msg)

        count= count+1;
        
        if(count==num_iterations)
            break;
        end

end

Link.Close;


%% ROS2 testing
%Create node
node = ros2node("/matlab");
msgType = 'us_msg/StampedArray';%'std_msgs/Float32MultiArray';
%Create publisher
publisher = ros2publisher(node,'/imgs',msgType);
%Initial date
t0 = datetime(1970, 1, 1, 0, 0, 0, 'TimeZone', 'UTC+2');

while (r == 0)
%Create message var
        msg = ros2message(publisher);
        %msg.layout.dim.label = 'test';
        %msg.data = single(y);
        msg.array.data = single(real(y));
        
        % %Add timestamp
        currentTime = datetime('now', 'TimeZone', 'UTC+2');
        % % Convert MATLAB time to ROS 2 time
        sec = seconds(currentTime - t0); % seconds since Unix epoch
        rtime = ros2time(sec);
        % % Assign the header to the message
        msg.stamp = rtime;
        %Send message
        send(publisher,msg)
        pause(5)
end