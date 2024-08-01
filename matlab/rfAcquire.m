
close all
clc 
clear all

%PARAMS
nlines=129;
numit=0;
fs=50e6;
lcut=5e6;
hcut=7e6;

%Define executable path
exePath = 'C:\Program Files (x86)\ULA-OP\Applicazione';

%Define path to save the images
%svPath = 'C:\Users\Mateo-drr\Documents\ALU---Autonomous-Lung-Ultrasound\data\acquired\Jul\raw';
svPath = 'C:\Users\Medical Robotics\Documents\ALU---Autonomous-Lung-Ultrasound\data\acquired\Jul\raw';
sv = 'C:\Users\Medical Robotics\Documents\ALU---Autonomous-Lung-Ultrasound\data\acquired\Jul\pydata';

%Files name prefix
pathPrefix = 'Matlink';

%Create the link
Link = UOLink(exePath, svPath, pathPrefix);

%config = 'C:\Users\Mateo-drr\Documents\ALU---Autonomous-Lung-Ultrasound\matlab\B-Mode RF\B_Config.cfg';
config = 'C:\Users\Medical Robotics\Documents\ALU---Autonomous-Lung-Ultrasound\matlab\B-Mode RF\B_Config.cfg';

%Path to probe file
probe = 'C:\ProgramData\ULA-OP\Probe\probeLA533.wks';

%Open the ulaop app
r = Link.Open(config, probe);


%Save files loop
while true

    %unfreeze
    r = Link.Freeze(0);

    %% Wait for next acquisition or exit
    userInput = input('Press Enter to continue or "f" to finish: ','s');
    % Check if the user input is 'f'
    if strcmp(userInput, 'f')
        disp('Link is closed');
        break;  % Exit the loop
    end

    %Save the US imgs
    r = Link.Save(numit);

    if(r ~= 0)
        disp('ERROR');
        return;
    end

    %small pause to let the files store
    pause(2)
    
    %% Read IQ file and save it
    DSN = 'SliceIQ';
    %if(numit == 0)
        tog = sprintf('_%03d_', numit);
        FilePath = [svPath, '\' pathPrefix , tog, DSN];
        UosStrings = importdata([FilePath , '.uos']);
    %end
    y = Link.GetAcq(numit, DSN, nlines, UosStrings);
    
    saveFileName = fullfile(sv, sprintf('iq_cf_%03d.mat', numit));
    save(saveFileName, 'y');
    
    %% Read RF file and save it
    DSN = 'SliceRf';
    %if(numit == 0)
        tog = sprintf('_%03d_', numit);
        FilePath = [svPath, '\' pathPrefix , tog, DSN];
        UosStrings = importdata([FilePath , '.uos']);
    %end
    y = Link.GetAcq(numit, DSN, nlines, UosStrings);
    
    saveFileName = fullfile(sv, sprintf('rf_cf_%03d.mat', numit));
    save(saveFileName, 'y');
    
    %% Plot the US image
    dims = size(y);
    if length(dims) == 3
        y = y(:,:,1);
    end
    y = y(1:6292,:);

    %% Filtering
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

    %increase the counter 
    numit = numit + 1
    
end

Link.Close;
close all