classdef DataSim < Data
    properties (Access = protected)
        fileObj;
    end
    properties
        filename;
        LastReadTime;
        LastReadData;
    end
    properties(Access=public)
        LastReadPri=0;      
    end
    methods
        function Buffer(obj,data,time,index,varargin)
            ind=index:index+length(time)-1;
            obj.LastReadData=data;
            obj.LastReadTime=time;
            obj.LastReadPri=ind(end);
        end
        function firstBlock=GetFirstBlock(~)
            firstBlock=0;
        end
        function SaveOnFile(obj,newFile)
            FreeLast(obj);
            oldfile=obj.filename;
            copyfile(oldfile,newFile,'f');
        end
        function FreeLast(obj)
            obj.LastReadData=[];
            obj.LastReadTime=[];
            obj.LastReadPri=0;
        end
    end
end