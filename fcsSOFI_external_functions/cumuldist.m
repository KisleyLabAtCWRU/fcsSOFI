function [DwellFinal2, IndexFinal2]=cumuldist(Dwell,TimeArray)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LK031213 Cumulative distribution code

% function to make cumulative distributions
% INPUT: Dwell - vector of data to be arranged in distribution, the data that would
%                be organized in a histogram and account for occurance in a
%                histogram (i.e. hist(Dwell) for probability distribution)
%        TimeArray - vector of x data (what values would be plotted as possible
%                    bins) For cumulative distributions this can contain all possible
%                    bins as defined by integration time, sensitivity, increments on
%                    the collected data, etc.
% OUTPUT: DwellFinal2 - final x data
%         IndexFinal2 - resulting, non-repetative probabilities on y axis
% plot(DwellFinal,IndexFinal) will give resulting cumulative distribution
% visualized
% Note: probabilities goes from 100% to 0%, this is based on the citation
%       below, but this varies based on field
% Read Walder, et al. High throughput single molecule tracking for analysis
% of rare populations and events. Analyst 2012, 137, 2987-2996. for details
% on the benefits of cumulative distributions.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

index = 1:1:numel(Dwell);
Varsort = sort(Dwell);
index=index/max(index);
index=1-index;

VarsortKeep=Varsort(1);
indexKeep=index(1);
for i=2:size(Varsort,2)
    if Varsort(i)~=Varsort(i-1)
        VarsortKeep=[VarsortKeep Varsort(i)];
        indexKeep=[indexKeep index(i)];
    end
end
Varsort=VarsortKeep;
index=indexKeep;

DwellFinal=TimeArray;
IndexFinal=zeros(1,numel(TimeArray));

for i=1:numel(Varsort)
    for j=1:numel(DwellFinal);
        if Varsort(i)==DwellFinal(j)
            IndexFinal(j)=index(i);
        end
    end
end
placeholder=IndexFinal(1);
for i=1:numel(IndexFinal)
    if IndexFinal(i)~=0
        placeholder=IndexFinal(i);
    else
        IndexFinal(i)=placeholder;
    end
end

% Remove repeats of 'linear' parts in cumulative distribution
IndexFinal2=IndexFinal(1);
DwellFinal2=DwellFinal(1);
for j=2:numel(IndexFinal)
    if IndexFinal(j)~=IndexFinal(j-1)
        IndexFinal2=[IndexFinal2; IndexFinal(j)];
        DwellFinal2=[DwellFinal2; DwellFinal(j)];
    end
end

end
