
close all
clc 
clear all

%% ROS2
%call C:\dev\ros2_humble\local_setup.bat
%Create node
test1 = ros2node("/test1");
msgType = 'sensor_msgs/Image';
msgType = 'std_msgs/Float32MultiArray';
%Create publisher
publisher = ros2publisher(test1,'/imgs',msgType);

%% US Aqcuisition

%Define executable path
exePath = 'C:\Program Files (x86)\ULA-OP\Applicazione';

%Define path to save the images
svPath = 'C:\Users\Medical Robotics\Documents\imgs';

%?
pathPrefix = 'f';

%Create the link
Link = UOLink(exePath, svPath, pathPrefix);

%Path to config file
%config = 'C:\ProgramData\ULA-OP\Mode\LA533 Modes (MSDLab Default)\B-Mode\B_Config.cfg';
config = 'C:\Users\Medical Robotics\Documents\cstmConfig.cfg';

%Path to probe file
probe = 'C:\ProgramData\ULA-OP\Probe\probeLA533.wks';

%Open the ulaop app
r = Link.Open(config, probe);

%Name of the slice, also used for reading the file
%DNS = 'SliceRf';
DNS = 'SliceIQ';
%Number of lines 
nlines = 129;

%h = figure(1);

%File number
count = 0;

t0 = datetime(1970, 1, 1, 0, 0, 0, 'TimeZone', 'UTC+2');

num_iterations = 50;  % Number of iterations in your loop
execution_times = zeros(1, num_iterations);

while(r == 0)

    tic

    fprintf('1\n')

    r = Link.AutoSave(1, count);
    fprintf('2\n')

    r = Link.Freeze(0); %1 means freeze
    fprintf('3\n')
    r = Link.WaitSave(1000000000); %timeout for waiting the app to save the data
    fprintf('4\n')

    %r not 0
    if(r ~= 0)
        return;
    end

    execution_times(count+1) = toc;
        
    %Get the US data
    %This thing uses the count number and formats it as _00{count}_

    y = Link.GetAcq(count, DNS, nlines); %0 is the file enumeration integer
    fprintf('5\n')

    % if(~ishandle(h))
    %     break;
    % end

    %increase counter
    count= count+1;

    %% ROS2 send message
    %Create message var
    msg = ros2message(publisher);
    msg.layout.dim.label = 'test';
    %msg.data = single(y);
    msg.data = single(real(y));

    % %Add timestamp
    % currentTime = datetime('now', 'TimeZone', 'UTC+2');
    % 
    % % Convert MATLAB time to ROS 2 time
    % sec = seconds(currentTime - t0); % seconds since Unix epoch
    % rtime = ros2time(sec);
    % % Assign the header to the message
    % header = ros2message('std_msgs/Header');
    % header.stamp = rtime;
    % msg.Header = header;
    % 
    % msg.header = rtime;

    %Send message
    send(publisher,msg)

    %
    execution_times(count) = toc;

    fprintf('6\n')
    
    % pippo=abs( y(:,:,1) );
    % imagesc( 20*log10(pippo/max(max(pippo))) ,[-50 0]);
    % drawnow;
    
    fprintf('Loop done\n')

    if(count==50)
        break;
    end
    
end

Link.Close;

