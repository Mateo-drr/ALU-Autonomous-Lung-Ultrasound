
clear all
close all
clc

%% PARAMS
store=false
plot=false
singleImg=false
id=0

% sampling frequency
fs=50e6;
% propagating medium S.O.S.
c0=1540;
% number of lines in each frame/image
Nlines=129;
% Start of frame
idx=120;

%% Reading RF Data
path = mfilename('fullpath');
path = fileparts(path)

%load ulaop lib
addpath(strcat(path,'\ULA-OP\ULAOP'));
addpath(strcat(path,'\ULA-OP\Class'));

%Path to files
datapath = strcat(path,'\data');
sv = strcat(path,'\pydata')

% List all files in the datapath directory that match the naming pattern
filePattern = fullfile(datapath, 'cf*_SliceRf.rfb');
fileList = dir(filePattern);
fileNames = {fileList.name};
fileList = natsortfiles(fileNames);

% Read and save the files for python processing
for n = 1:length(fileList)
    data = fullfile(datapath, fileList{n});
    disp(['Processing file: ', fileList{n}]);

    DataObj=DataUlaopPostBeamforming(data); 
    Read(DataObj,'firstPri',1,'npri',GetTotalPri(DataObj)); %Read all the avaialble PRIs
    [nGate,nPri]=size(DataObj.LastReadData); %extract the number of gate and the number of pris
    time=DataObj.LastReadTime(1)+(0:nGate-1)/DataObj.fs; %time axis
    rf=DataObj.LastReadData;
    
    

    %plot the frames
    if plot
        figure, imagesc(20*log10(abs(rf)));
    end

    %save
    if store
        saveFileName = fullfile(sv, sprintf('cf_%03d.mat', n));
        save(saveFileName, 'rf');
    end
end

% Read file
% data = strcat(path,files(id));
% DataObj=DataUlaopPostBeamforming([char(data)]); 
% Read(DataObj,'firstPri',1,'npri',GetTotalPri(DataObj)); %Read all the avaialble PRIs
% [nGate,nPri]=size(DataObj.LastReadData); %extract the number of gate and the number of pris
% time=DataObj.LastReadTime(1)+(0:nGate-1)/DataObj.fs; %time axis
% rf=DataObj.LastReadData;
% 
% %plot the frames
% figure, imagesc(20*log10(abs(rf)));
% 
% %save
% if save
%     saveas(grf, strcat(sv,'frames',int2str(id),'.png'))
% end

% rfIMAGE=rf(:,idx:idx+129);
% rfIMAGE=rfIMAGE./max(max(rfIMAGE));
% 
% LateralDimension=[1:Nlines]*Pitch;
% LateralDimension=LateralDimension-(Pitch*floor(Nlines/2));
% Depth=OffSet+[1:6144]*c0*1/(2*fs);
% 
% %plot a single frame (?)
% figure
% imagesc(LateralDimension,Depth*1000,20*log10(abs(rfIMAGE)))
% caxis([-40 0])
% xlabel('Lateral Dimension [mm]')
% ylabel('Depth [mm]')
% axis image
% saveas(gcf, strcat(sv,'img',int2str(id),'.png'))
% 
% % Axis
% dt=1/fs;
% yy = size(rfIMAGE);
% yy = yy(1)-1;
% timeaxis=[0:yy]*dt;
% frequenxyaxis=[0:yy]*(fs/yy);
% frequenxyaxis=frequenxyaxis;
% 
% %%Filtering the image by removing frequencies 3e6<x<6.5e6 line by line
% for LineIndex=1:129
% % Select Signal
% signal=rfIMAGE(:,LineIndex);
% 
% % filtering
% [A,B] = butter(10,[3e6 6.5e6]/25e6); %filter order, cutoff freqs
% % h = fvtool(A,B)
% filteredsignal=filter(A,B,signal);
% 
% ImageFiltered(:,LineIndex)=abs(hilbert(filteredsignal));
% % 
% figure(3)
% subplot(2,1,1)
% plot(timeaxis*1000,signal./max(signal))
% axis([0 0.12 -1.2 1.2])
% xlabel('Time [ms]')
% ylabel('Normalized amplitude')
% subplot(2,1,2)
% plot(frequenxyaxis./1e6,20*log10(abs(abs(fft(signal))))-max(20*log10(abs(abs(fft(signal))))))
% 
% axis([0 25 -40 0])
% xlabel('Frequency [MHz]')
% ylabel('Normalized amplitude [dB]')
% %saveas(gcf, strcat(sv,'noFilter',int2str(id),'.png'))
% 
% % 
% figure(4)
% subplot(2,1,1)
% plot(timeaxis*1000,filteredsignal./max(filteredsignal))
% hold on
% plot(timeaxis*1000,ImageFiltered(:,LineIndex)./max(filteredsignal))
% hold off
% axis([0 0.12 -1.2 1.2])
% xlabel('Time [ms]')
% ylabel('Normalized amplitude')
% subplot(2,1,2)
% plot(frequenxyaxis./1e6,20*log10(abs(abs(fft(filteredsignal))))-max(20*log10(abs(abs(fft(filteredsignal))))))
% axis([0 25 -40 0])
% xlabel('Frequency [MHz]')
% ylabel('Normalized amplitude [dB]')
% 
% end
% saveas(gcf, strcat(sv,'filtered',int2str(id),'.png'))
% 
% figure(5)
% imagesc(LateralDimension,Depth*1000,20*log10(abs(ImageFiltered)))
% caxis([-40 0])
% xlabel('Lateral Dimension [mm]')
% ylabel('Depth [mm]')
% axis image
% colormap('gray')
% 
% saveas(gcf, strcat(sv,'grey',int2str(id),'.png'))


