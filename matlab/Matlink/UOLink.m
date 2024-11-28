%% UOLink
% Construct a UOLink object
%
% UOLink activates a link between Matlab and ULA-OP.
% 
%% Syntax
% 
% |obj=UOLink(AppFolder,SaveFolder,FilePathPrefix,'PropertyName',PropertyValue,...)|
% 
%% Description
% 
% |obj=UOLink(AppFolder,SaveFolder,FileName,'PropertyName',PropertyValue,...)| constructs an
% object of the UOLink class and sets the properties of that 
% object that are named in the argument list to the given values. 
% All property name arguments must be quoted strings. Any properties
% that are not specified are given their default values.
% 
classdef UOLink < hgsetget
    properties (Access=public)
        %% Properties
        %
        % |AppFolder| - Application Folder (where the executable resides)
        % 
        % |SaveFolder| - Save Folder (where the app saves the acquisitions)
        %
        % |FilePathPrefix| - Complete File-path prefix
        % 
        % |Admin| - Administrative Level
        AppFolder='';               
        SaveFolder=tempdir;         
        FilePathPrefix=tempname;    
    end
    properties
        Admin=2;
    end
    methods
        %% Methods
        function obj=UOLink(AppFolder,SaveFolder,FileName,varargin)
            %UOLink Constructor. AppFolder, SaveFolder and FileName are required 
            %input, the user can define Admin in varargin.
            if nargin~=0
                if nargin<3
                    ME=MException('MATLAB:UOLink','Not enough input arguments.');
                    throw(ME)
                end
                obj.AppFolder=AppFolder;
                obj.SaveFolder=SaveFolder;
                obj.FilePathPrefix=[SaveFolder , '\' , FileName];
                if ~isempty(varargin)
                    set(obj,varargin{:});    
                end
                [y]=ML2UOGate(0, 1, 0); % Clear any pending flag
            end
        end
        function [ y ] = GetAcq(obj, count, slicename, nlinesperframe, UosStrings)
            %% 
            % *GetAcq*
            % 
            % |[ y ] = GetAcq(obj, count, slicename, nlinesperframe)|
            % Get acquisition data (PRE8, POST & IQ supported).
            % 
            % Input:
            % 
            % |-   count| = File enumeration integer
            % 
            % |-   slicename| = Name of the slice
            % 
            % |-   nlinesperframe| = (optional) Number of lines per frame (eg.
            %      192 for a standard B-mode image)
            % 
            % Output:
            % 
            % |-   y| = N-Dimensional Matrix with acquisition data
            % 
            %      IQ   -> #Gates, #LinesPerFrame, #Frames
            % 
            %      PRE8 -> #Channels (64), #Gates, #LinesPerFrame, #Frames
            % 
            %      POST -> #Gates, #LinesPerFrame, #Frames
            % 
            count = sprintf('_%03d_', count);
            FilePath = [obj.FilePathPrefix , count , slicename];

            % UosStrings = importdata([FilePath , '.uos']);
            % UosStrings= [Info]
            % Type = 0;
            % Slice = 0;
            % TotalSize = 264192;
            % NBlocks = 129;
            % BlockLength = 256;
            % BlockSize = 2048;
            % FirstBlock = 0;
            obj.assignvars(UosStrings);
            
            if( ~exist('nlinesperframe', 'var') )
                nlinesperframe = 1;
            end

            BlocksToDiscard = mod(FirstBlock, nlinesperframe);
            if(BlocksToDiscard ~= 0)
                BlocksToDiscard = nlinesperframe-BlocksToDiscard;
            end
            BlocksToLoad = NBlocks - BlocksToDiscard;
            BlocksToLoad = BlocksToLoad - mod(BlocksToLoad, nlinesperframe);

            if(Type == 0)
                fid = fopen([FilePath , '.uob'], 'rb');
                fseek(fid, BlockSize*BlocksToDiscard, -1);
                src = fread(fid, BlocksToLoad*BlockLength*2, 'int32');
                fclose(fid);
                src = src(1:2:end) + 1i .* src(2:2:end);
                src = squeeze( reshape(src, BlockLength, nlinesperframe, []) );
                y = src;
            end

            if(Type == 1)
                fid = fopen([FilePath , '.rff'], 'rb');
                fseek(fid, BlockSize*BlocksToDiscard, -1);
                src = fread(fid, BlocksToLoad*BlockLength*64, 'int8=>int8');
                fclose(fid);
                src = reshape(src, 2, 4, 8, []);
                src = permute(src, [1 3 2 4]);
                y = reshape(src, 64, BlockLength, nlinesperframe, []);
            end
            
            if(Type == 2)
                fid = fopen([FilePath , '.rff12'], 'rb');
                fseek(fid, BlockSize*BlocksToDiscard, -1);
                src = fread(fid, BlocksToLoad*BlockLength*64*1.5, 'uint8=>uint8');
                fclose(fid);
                src=reshape(src,2,4,8,3,[]);
                src=uint16(src); 
                dim=size(src);
                src=int16(cat(4,...
                    bitor(bitshift(bitand(src(:,:,:,3,:),uint16(15)),8),src(:,:,:,1,:)),...
                    bitor(bitshift(bitand(src(:,:,:,3,:),uint16(240)),4),src(:,:,:,2,:))));
                src=mod(src-2^11,2^12)-2^11;
                src=reshape(src,[dim(1),dim(2),dim(3),2*dim(5)]);
                src=permute(src,[1 3 2 4]);
                src=reshape(src,64,[]);
                y=reshape(src,64,BlockLength, nlinesperframe, []);
            end
            
            if(Type == 3)
                fid = fopen([FilePath , '.rfb'], 'rb');
                fseek(fid, BlockSize*BlocksToDiscard, -1);
                src = fread(fid, BlocksToLoad*BlockLength*8, 'int16=>int16');
                fclose(fid);
                l = length(src);
                src = reshape(src, [4 l/4]);
                src = permute(src, [2 1]);
                src = reshape(src, [4 2 l/4/4/2 4]);
                src = permute(src, [1 3 2 4]);
                src = reshape(src, BlockLength, nlinesperframe, [], 8);
                y = squeeze( sum(src, 4) );
            end
        end
        function [ y ] = Open(obj, config, probe)
            %% 
            % *Open*
            % 
            % |[ y ] = Open(obj, config, probe)|
            % Opens the 'ULA-OP Modula' application for subsequent data acquisition.
            % 
            % Input:
            % 
            % |-   config| = Configuration file (full path required)
            % 
            % |-   probe| = Probe file (full path required)
            % 
            % Output:
            % 
            % |-   y| 
            % 
            %      0 -> Ok. Application correctly initialized
            % 
            %      1 -> Warning. The application is already open
            % 
            %     -1 -> Error.
            % 
            y = ML2UOGate(0, 0, 0);
            if(y < 0)
                return; %Error
            end
            if(y == 0)
                y = 1; %Already open
                return;
            end
            AppLine = [obj.AppFolder , '\UlaOpModula.exe'];
            CmdLine = ['-C "' , config , '" -P "' , probe , '" -S "', obj.SaveFolder , '" -A ', num2str(obj.Admin)];
            DirLine = [obj.AppFolder];
            y = ML2UOGate(2, 1, 0, ['?' , AppLine , '?' , CmdLine , '?' , DirLine, '?']);
            if(y < 0)
                return; %Error
            end
            y = ML2UOGate(0, 0, 10000);
            if(y ~= 0)
                y = -1;
                return; %Error
            end
            y = ML2UOGate(0, 1, 0); %Clear any pending 'SAVE' flag
            if(y >= 0)
                y = 0;
            end
        end
    end
    methods (Static)
        function [ y ] = AutoSave(state, count)
            %% 
            % *AutoSave*
            % 
            % |[ y ] = AutoSave(state, count)|
            % Sets the autosave state. When the autosave feature is
            % enabled, the application stores data immediately after the
            % end of the acquisition.
            % 
            % Input:
            % 
            % |-   state|
            % 
            %      0 -> autosave OFF
            % 
            %      1 -> autosave ON
            % 
            % |-   count| File enumeration integer
            % 
            % Output:
            % 
            % |-   y|
            % 
            %      0 -> Ok
            % 
            %     -1 -> Error
            % 
            if(state ~= 0)
                state = 65536;
            end
            y = ML2UOGate(1, 2, state+count);
        end
        function [ y ] = Close
            %% 
            % *Close*
            % 
            % |[ y ] = Close| Closes the 'ULA-OP Modula' application
            % 
            % Output:
            % 
            % |-   y|
            % 
            %      0 -> Ok
            % 
            %     -1 -> Error
            % 
            y = ML2UOGate(1, 10, 0);
        end
        function [ y ] = Freeze(state)
            %% 
            % *Freeze*
            % 
            % |[ y ] = Freeze(state)|
            % Sets the PRF state (Freezes/Unfreezes the acquisition)
            % 
            % Input:
            % 
            % |-   state|
            % 
            %      0 -> PRF On (acq. un-frozen)
            % 
            %      1 -> PRF Off (acq. frozen)
            % 
            % Output:
            % 
            % |-   y|
            % 
            %      0 -> Ok
            % 
            %     -1 -> Error
            % 
            if(state ~= 0)
                state = 1;
            end
            y = ML2UOGate(1, 0, state);
        end
        function [ y ] = Key(keycode)
            %% 
            % *Key*
            % 
            % |[ y ] = Key(keycode)|
            % Send a key-code to the application for parameter setting
            % 
            % Input:
            % 
            % |-   keycode| = Windows Key-Code
            % 
            % Output:
            % 
            % |-   y|
            % 
            %      0 -> Ok
            % 
            %     -1 -> Error
            % 
            y = ML2UOGate(1, 3, keycode);
        end
        function [ y ] = Save(count)
            %% 
            % *Save*
            % 
            % |[ y ] = Save(count)|
            % Stores the acquisition immediately
            % 
            % Input:
            % 
            % |-   count| = File enumeration integer
            % 
            % Output:
            % 
            % |-   y|
            % 
            %      0 -> Ok
            % 
            %     -1 -> Error
            % 
            y = ML2UOGate(1, 1, count);
        end
        function [ y ] = WaitSave(time)
            %% 
            % *WaitSave*
            % 
            % |[ y ] = WaitSave(time)|
            % Waits until the application has stored all the data
            % 
            % Input:
            % 
            % |-   time| = timeout value (in milliseconds)
            % 
            % Output:
            % 
            % |-   y|
            % 
            %      0 -> Ok. Data are ready
            % 
            %      1 -> Timeout
            % 
            %     -1 -> Error
            % 
            y = ML2UOGate(0, 1, time);
        end
        function [ y ] = ShowWindow(param)
            %%
            % *ShowWindow*
            %
            % |[ y ] = ShowWindow(param)|
            % Sets how the main window appears. 
            % 
            % Input:
            % |- param| a string among 'hide' 'maximize' 'minimize' 'restore' 'show'
            % 
            % Output:
            %
            % |-   y|
            % 
            %      0 -> Ok
            % 
            %     -1 -> Error
            %
            param=lower(param);
            switch param
                case 'hide'
                    param=0;
                case 'maximize'
                    param=3;
                case 'minimize'
                    param=6;
                case 'restore'
                    param=9;
                case 'show'
                    param=4;
                otherwise
                    param=4;
            end
            y = ML2UOGate(1, 4, param);
        end
    end
    methods (Static, Access=private)
        function assignvars(newData1)
            for i = 2:length(newData1.textdata)
                assignin('caller', newData1.textdata{i,1},newData1.data(i-1));
            end
        end  
    end
end