classdef DataSimPostBeamforming < DataSim
    properties
        lastWrittenPri=0;
        type=0;
        headerOffset=0;
        size;
    end
    methods
        function obj=DataSimPostBeamforming(filename,varargin)
            if nargin~=0
                obj=DataSimPostBeamforming;
                if ~isempty(varargin)
                    set(obj,varargin{:});
                end
            else
                filename=[tempname '.spb'];
            end
            obj.filename=filename;
        end
        function npri=GetTotalPri(obj)
            npri=obj.lastWrittenPri;
        end
        function Read(obj,varargin)
            firstpri=obj.LastReadPri+1;
            npri=1;
            for i = 1 : 2 : length(varargin)
                propname=lower(varargin{i});
                propvalue=varargin{i+1};
                switch propname
                    case 'firstpri'
                        if ~isnumeric(propvalue)
                            if propvalue<1 || propvalue>numel(obj.fileObj.time) || floor(propvalue)==propvalue
                                MException('SIMAG:DataError',[propname ' is not valid.']);
                                return
                            end
                            MException('SIMAG:DataError',[propname ' is not valid.']);
                            return
                        end
                        firstpri=propvalue;
                    case 'npri'
                        if ~isnumeric(propvalue)
                            if propvalue<1 || propvalue>numel(obj.fileObj.time) || floor(propvalue)==propvalue
                                MException('SIMAG:DataError',[propname ' is not valid.']);
                                return
                            end
                            MException('SIMAG:DataError',[propname ' is not valid.']);
                            return
                        end
                        npri=propvalue;
                    otherwise
                        MException('SIMAG:DataError',[propname ' is not an expected property.']);
                        return
                end
            end
            index=firstpri:firstpri+npri-1;
            index=index+obj.offsetPri;
            index(index>GetTotalPri(obj))=[];
            if ~isempty(index)
                ind=(index(1)-1)*4*(obj.size(1)+1);
                fseek(obj.fileObj,ind+obj.headerOffset,'bof');
                temp=fread(obj.fileObj,[(obj.size(1)+1),length(index)],'single');
                obj.LastReadData=temp(2:end,:);
                obj.LastReadTime=temp(1,:);
                obj.LastReadPri=index(end)-obj.offsetPri;
            else
                obj.LastReadData=[];
                obj.LastReadTime=[];
                obj.LastReadPri=[];
            end
        end
        function ReserveMemory(obj,size,varargin)
            waitbarFlag=true;
            for i = 1 : 2 : length(varargin)
                propname=lower(varargin{i});
                propvalue=varargin{i+1};
                
                switch propname
                    case 'waitbarflag'
                        if ~islogical(propvalue)
                            MException('SIMAG:DataError',[propname ' is not valid.']);
                            return
                        end
                        waitbarFlag=propvalue;
                    otherwise
                        MException('SIMAG:DataError',[propname ' is not an expected property.']);
                        return
                end
            end
            filename=obj.filename;
            fileID=fopen(filename,'w+');
            totSize=prod(size)+size(2);
            packSize=25*1024^2; %scrive 100MB per volta
            packData=zeros(1,packSize,'single');
            nTot=ceil(totSize/packSize);
            if waitbarFlag
                h=waitbar(0,'Creating file...');
                drawnow;
            end
            perc=0;
            fwrite(fileID,obj.type,'uint32');
            fwrite(fileID,size(1),'uint32');
            fwrite(fileID,size(2),'uint32');
            fwrite(fileID,obj.fs,'single');
            obj.headerOffset=ftell(fileID);
            while totSize>0
                if totSize<packSize
                    packSize=totSize;
                end
                totSize=totSize-packSize;
                fwrite(fileID,packData(1:packSize),'single');
                if waitbarFlag
                    perc=perc+1/nTot;
                    waitbar(perc,h);
                end
                drawnow;
            end
            fclose(fileID);
            GenerateFileObj(obj,true);
            if waitbarFlag
                close(h);
            end
        end
        function GenerateFileObj(obj,writable)
            if writable
                obj.fileObj=fopen(obj.filename,'r+');
            else
                obj.fileObj=fopen(obj.filename,'r');
            end
            obj.type=fread(obj.fileObj,1,'uint32');
            obj.size=fread(obj.fileObj,2,'uint32');
            obj.fs=fread(obj.fileObj,1,'single');
            obj.headerOffset=ftell(obj.fileObj);
            if ~writable
                obj.lastWrittenPri=obj.size(2);
            end
        end
        function Write(obj,data,time,index,varargin)
            temp=cat(1,reshape(time,1,[]),data);
            ind=(index-1)*4*(obj.size(1)+1);
            fseek(obj.fileObj,ind+obj.headerOffset,'bof');
            fwrite(obj.fileObj,temp,'single');
            obj.lastWrittenPri=index+length(time)-1;
            bufferflag=false;
            for i = 1 : 2 : length(varargin)
                propname=lower(varargin{i});
                propvalue=varargin{i+1};
                switch propname
                    case 'bufferflag'
                        if ~islogical(propvalue)
                            MException('SIMAG:DataError',[propname ' is not valid.']);
                            return
                        end
                        bufferflag=propvalue;
                        bufferFlagInd=i;
                    otherwise
                        MException('SIMAG:DataError',[propname ' is not an expected property.']);
                        return
                end
            end
            if bufferflag
                varargin(bufferFlagInd:bufferFlagInd+1)=[];
                keyboard
                Buffer(obj,data,time,index,varargin);
            end 
        end
        function delete(obj)
            if exist(obj.filename,'file')==2
                [~,permission,~,~]=fopen(obj.fileObj);
                fclose(obj.fileObj);
                if strcmp(permission,'rb+')
                    delete(obj.filename);
                end
            end
        end
    end
end