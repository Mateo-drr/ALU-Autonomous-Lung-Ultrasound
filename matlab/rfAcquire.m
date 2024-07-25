close all
clc 
clear all

%Define executable path
exePath = 'C:\Program Files (x86)\ULA-OP\Applicazione';

%Define path to save the images
svPath = 'C:\Users\Mateo-drr\Documents\ALU---Autonomous-Lung-Ultrasound\data\acquired\Jul\raw';

%?
pathPrefix = 'Matlink';

%Create the link
Link = UOLink(exePath, svPath, pathPrefix);

config = 'C:\Users\Mateo-drr\Documents\ALU---Autonomous-Lung-Ultrasound\matlab\B-Mode RF\B_Config.cfg';

%Path to probe file
probe = 'C:\ProgramData\ULA-OP\Probe\probeLA533.wks';

%Open the ulaop app
r = Link.Open(config, probe);

%Read Rf file and save it
DSN = 'SliceRf';
y = Link.GetAcq(toggle, DSN, nlines);

saveFileName = fullfile(sv, sprintf('cf_%03d.mat', n));
save(saveFileName, 'y');


%Read iq file and save it
DSN = 'SliceIQ';
y = Link.GetAcq(toggle, DSN, nlines);

saveFileName = fullfile(sv, sprintf('iq_cf_%03d.mat', n));
save(saveFileName, 'y');


