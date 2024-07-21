close all
clc 
clear all

%% ROS2
%call C:\dev\ros2_humble\local_setup.bat
%Create node
node = ros2node("/matlab");
msgType = 'std_msgs/Float32MultiArray';
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


num_iterations = 50;  % Number of iterations in your loop
execution_times = zeros(1, num_iterations);

toggle = 0;

r = Link.AutoSave(1, toggle);
r = Link.Freeze(0);

toggle = 1-toggle;

% h = figure(1);
while(r == 0)
        
        tic
        
        r = Link.WaitSave(5000);
        
        if(r ~= 0)
            return;
        end
        
        r = Link.AutoSave(1, toggle);
        r = Link.Freeze(0);
        
        toggle = 1-toggle;
        
        y = Link.GetAcq(toggle, DSN, nlines);
        
        %% ROS2 send message
        %Create message var
        % msg = ros2message(publisher);
        % msg.layout.dim.label = 'test';
        % %msg.data = single(y);
        % msg.data = single(real(y));
        % %Send message
        % send(publisher,msg)

        count= count+1;
        execution_times(count) = toc;
        
        if(count==50)
            break;
        end

end

Link.Close;
