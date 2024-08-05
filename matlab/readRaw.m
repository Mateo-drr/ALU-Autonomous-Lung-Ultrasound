
clear all
clc
close all

%PARAMS
day='01Aug5';
ptype='rl';
FilePathPrefix='Matlink'; 
slicename='SliceRf';
nlines=129;
plot=false;
store=true;

% Specify the folder containing the files
folder = ['C:\Users\Mateo-drr\Documents\data\acquired\',day,'\raw\',ptype,'\'];  
sv=['C:\Users\Mateo-drr\Documents\data\acquired\',day,'\pydata\',ptype,'\'];

% Get a list of all files in the folder
fileList = dir(fullfile(folder, '*.rfb'));  % Replace '*.mat' with the file extension of your files if different

% Loop through each file in the folder
for k = 1:length(fileList)
    % Get the full path of the file
    filePath = fullfile(folder, fileList(k).name)
    
    % Load the file
    data = GetAcq(k-1,folder,FilePathPrefix,slicename,nlines);
    
    % Process the loaded data
    dims = size(data);

    if plot
        if length(dims) == 3
            rf=data(:,:,1); %get the first to plot
        end
        imagesc(20*log10(abs(rf)))
    end

    if store
        name = [ptype,sprintf('_%03d', k-1)]
        save(fullfile(sv, [name, '.mat']), 'data');
    end

end

function [S] = parseUosStrings(UosStrings)
    labels = UosStrings.textdata(:, 1);
    values = UosStrings.data;

    % Create a struct to hold the parsed values
    S = struct();
    
    % Assuming the values are in the same order as the labels after the '[Info]' entry
    for i = 2:length(labels)
        label = labels{i};
        label = strrep(label, '[', ''); % Remove '['
        label = strrep(label, ']', ''); % Remove ']'
        S.(label) = values(i-1); % The first label '[Info]' does not have a corresponding value
    end
end

function y = GetAcq(count,path, FilePathPrefix, slicename, nlinesperframe)
    % Define the count string
    count = sprintf('_%03d_', count);
    % Create the file path
    FilePath = [path,FilePathPrefix , count , slicename];

    % Import UosStrings and parse it
    UosStrings = importdata([FilePath , '.uos']);
    S = parseUosStrings(UosStrings);

    % Assign variables from the parsed struct
    FirstBlock = S.FirstBlock;
    NBlocks = S.NBlocks;
    BlockSize = S.BlockSize;
    BlockLength = S.BlockLength;
    Type = S.Type;

    % Check if nlinesperframe exists, otherwise set it to 1
    if nargin < 4
        nlinesperframe = 1;
    end

    % Calculate BlocksToDiscard and BlocksToLoad
    BlocksToDiscard = mod(FirstBlock, nlinesperframe);
    if BlocksToDiscard ~= 0
        BlocksToDiscard = nlinesperframe - BlocksToDiscard;
    end
    BlocksToLoad = NBlocks - BlocksToDiscard;
    BlocksToLoad = BlocksToLoad - mod(BlocksToLoad, nlinesperframe);

    % Handle different file types
    switch Type
        case 0
            fid = fopen([FilePath , '.uob'], 'rb');
            fseek(fid, BlockSize * BlocksToDiscard, -1);
            src = fread(fid, BlocksToLoad * BlockLength * 2, 'int32');
            fclose(fid);
            src = src(1:2:end) + 1i .* src(2:2:end);
            src = squeeze(reshape(src, BlockLength, nlinesperframe, []));
            y = src;
        case 3
            fid = fopen([FilePath , '.rfb'], 'rb');
            fseek(fid, BlockSize * BlocksToDiscard, -1);
            src = fread(fid, BlocksToLoad * BlockLength * 8, 'int16=>int16');
            fclose(fid);
            l = length(src);
            src = reshape(src, [4 l/4]);
            src = permute(src, [2 1]);
            src = reshape(src, [4 2 l/4/4/2 4]);
            src = permute(src, [1 3 2 4]);
            src = reshape(src, BlockLength, nlinesperframe, [], 8);
            y = squeeze(sum(src, 4));
        otherwise
            error('Unsupported file type');
    end
end
