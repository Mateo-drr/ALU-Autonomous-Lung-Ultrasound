
%% Reading RF Data
path = mfilename('fullpath');
path = fileparts(path)

%load ulaop lib
addpath(strcat(path,'\ULA-OP\ULAOP'));
addpath(strcat(path,'\ULA-OP\Class'));

dpath = 'C:\Users\Medical Robotics\Documents\imgs\f_000_SliceIQ.uob'
DataObj=DataUlaopPostBeamforming(data);