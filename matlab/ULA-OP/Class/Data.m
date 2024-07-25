classdef Data < hgsetget
    properties
        fs;
        offsetPri=0;
    end
    methods (Abstract)
    end
    methods
        function SetOffsetPri(obj,nPri)
            %% 
            % *GetOffsetPri*
            % 
            % Gets the offset PRI to align PRI with correct absolute index
            % in Data.
            firstBlock=GetFirstBlock(obj);
            if mod(firstBlock,nPri)==0 
                obj.offsetPri=0; 
            else
                obj.offsetPri=nPri-mod(firstBlock,nPri); 
            end
        end
    end
end