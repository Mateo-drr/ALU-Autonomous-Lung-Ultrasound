classdef DataSimUlated < DataSim
    methods
        function obj=DataSimUlated(varargin)
                obj.filename=[tempname,'.mat'];
                if nargin~=0
                    obj=DataSimUlated;
                    if ~isempty(varargin)
                        set(obj,varargin{:});
                    end
                end
                obj.fileObj=matfile(obj.filename,'Writable',true);
        end
        function Write(obj,data,time,index,varargin)
%             ind=index:index+length(time)-1;
            ind=index;
            obj.fileObj.data(ind,1)={data};
            obj.fileObj.time(ind,1)=time;
            obj.fileObj.depth(ind,1)=size(data,1);
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
                Buffer(obj,data,time,index,varargin);
            end      
        end
        function DataOut=Convert2PreBeamforming(obj,varargin)
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
            Read(obj,'firstpri',1,'npri',1);
            nEle=size(obj.LastReadData{:},2);
            DataOut=DataSimPreBeamforming;
            DataOut.fs=obj.fs;
            s_time=obj.fileObj.time;
            depth=obj.fileObj.depth;
            s_time=round(s_time*obj.fs);
            t_post=min(s_time(:));
            lastDepth=max(s_time+depth-t_post);
            nPri=length(s_time);
            try
                [~, system]=memory;
                nPritoRead=max(1,round(system.PhysicalMemory.Available/4/lastDepth/nEle/4)); % scelgo di allocare 1/4 della memoria disponibile sperando che sia tutta contigua
            catch
                nPritoRead=1;
            end
            nPritoRead=min(nPritoRead,nPri);
            ReserveMemory(DataOut,[lastDepth,nEle,nPritoRead],'waitbarFlag',waitbarFlag);
            pre=zeros(lastDepth,nEle,nPritoRead,'single');
            ind=1:nPritoRead:nPri;
            if ind(end)+nPritoRead-1<nPri
                ind(end+1)=nPri;
            end
            for i=1:length(ind)
                Read(obj,'firstpri',ind(i),'npri',nPritoRead);
                pre(:)=0;
                if length(obj.LastReadData)<nPritoRead
                    pre=pre(:,:,1:length(obj.LastReadData));
                end
                for j=1:length(obj.LastReadData)
                    temp=obj.LastReadData{j};
                    if t_post<0
                        d=s_time((i-1)*nPritoRead+j)-t_post;
                        temp(1:-t_post-d,:)=[];
                    else
                        d=s_time((i-1)*nPritoRead+j)-t_post;
                    end
                    pre(1:size(temp,1)+d,:,j)=padarray(temp,d,0,'pre');
                end
                Write(DataOut,pre,t_post(ones(1,size(pre,3)))/obj.fs,ind(i));
            end
        end
        function npri=GetTotalPri(obj)
            npri=size(obj.fileObj,'time',1);
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
            index(index>GetTotalPri(obj))=[];
            if ~isempty(index)
                obj.LastReadData=obj.fileObj.data(index,1);
                obj.LastReadTime=obj.fileObj.time(index,1);
                obj.LastReadPri=index(end);
            else
                obj.LastReadData=[];
                obj.LastReadTime=[];
                obj.LastReadPri=[];
            end
        end
        function delete(obj)
            if exist(obj.filename,'file')==2
                delete(obj.filename);
            end
        end
    end
end