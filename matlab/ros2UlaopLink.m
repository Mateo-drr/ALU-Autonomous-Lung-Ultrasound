close all
clc 
clear all

%Add path to the ulaop program files
addpath("C:\Program Files (x86)\ULA-OP\MATLAB\MatLink\")

nlines=129;
ysize=6292;
numit=0;
fs=50e6;
lcut=5e6;
hcut=7e6;

%IF RUNNING FIRST TIME YOU NEED TO BUILD THE CSTM MSG TYPE
firstTime=false;
if firstTime
    %Get current directory
    % dir = fileparts(mfilename('fullpath'));
    ros2genmsg("C:\Users\Medical Robotics\Documents\ALU-Autonomous-Lung-Ultrasound\ros2Packages\src\")
end



%% ROS2
%Create cstm ros msg type
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
%config = 'C:\Users\Medical Robotics\Documents\B-Mode Matlink\B_Config.cfg';
config = 'C:\Users\Medical Robotics\Documents\ALU-Autonomous-Lung-Ultrasound\matlab\B-Mode Matlink\B_Config.cfg';

%Path to probe file
probe = 'C:\ProgramData\ULA-OP\Probe\probeLA533.wks';

%Open the ulaop app
r = Link.Open(config, probe);

%Name of the slice, also used for reading the file
DSN = 'SliceRf';
%DSN = 'SliceIQ';
%Number of lines 
nlines = 129;

%File number
count = 0;


num_iterations = 1;  % Number of iterations in your loop

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

        %% Plot the US image
        dims = size(y);
        if length(dims) == 3
            y = y(:,:,1);
        end
        y = y(1:ysize,:);
    
        % Filtering
        % Convert the cutoff frequencies to normalized form
        nyquist = fs / 2;
        low_cutoff_norm = lcut / nyquist;
        high_cutoff_norm = hcut / nyquist;
    
        % Design Butterworth bandpass filter
        [b, a] = butter(10, [low_cutoff_norm, high_cutoff_norm], 'bandpass');
    
        % Get the size of the image
        [num_rows, num_cols] = size(y);
    
        % Initialize the filtered image
        filtered_img = zeros(size(y));
    
        % Apply the filter column by column
        for col = 1:num_cols
            column_data = double(y(:, col));  % Get the column data and convert to double
            filtered_column = filtfilt(b, a, column_data);  % Apply zero-phase filtering
            filtered_img(:, col) = filtered_column;  % Store the filtered column
        end
    
        %plot
        figure;
        
        subplot(1, 2, 1);
        imagesc(20*log10(abs(y)));
    
        subplot(1, 2, 2);
        imagesc(20*log10(abs(hilbert(filtered_img))));

        figure(2);
        rimg = imresize(filtered_img, [512,128]);
        imagesc(20*log10(abs(hilbert(rimg))))
        
        %% ROS2 send message
        %Create message var
        msg = ros2message(publisher);
        %msg.layout.dim.label = 'test';

        if strcmp(DSN,'SliceRf')
            % rimg = imresize(y,[512,128]);
            msg.array.data = single(rimg);
        else
            msg.array.data = single(real(y));
        end

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
% %Create node
% node = ros2node("/matlab");
% msgType = 'us_msg/StampedArray';%'std_msgs/Float32MultiArray';
% %Create publisher
% publisher = ros2publisher(node,'/imgs',msgType);
% %Initial date
% t0 = datetime(1970, 1, 1, 0, 0, 0, 'TimeZone', 'UTC+2');
% 
% while (r == 0)
% %Create message var
%         msg = ros2message(publisher);
%         %msg.layout.dim.label = 'test';
%         %msg.data = single(y);
%         msg.array.data = single(real(y));
% 
%         % %Add timestamp
%         currentTime = datetime('now', 'TimeZone', 'UTC+2');
%         % % Convert MATLAB time to ROS 2 time
%         sec = seconds(currentTime - t0); % seconds since Unix epoch
%         rtime = ros2time(sec);
%         % % Assign the header to the message
%         msg.stamp = rtime;
%         %Send message
%         send(publisher,msg)
%         pause(5)
% end