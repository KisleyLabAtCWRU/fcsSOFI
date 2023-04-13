%{ 
 Code by Jixin Chen @ Rice University 2013/10/16
 Get the grouped final locations from the super-resolution image. Set a threshold sigma for the desperse of events.
 generate a Gaussian standard with this sigma and compare the standard to the super-resolution (SR) image. 
Set a correlation factor threshhold e.SRCorrFactor for the shape and a number of event threshold e.nevent.
After select the final location by cross correlation, group the events with the following rule: screen the 
events in a single frame and find the nearest final location; if the distance is less then e.FinalLocatThresh*the sigma, group it, otherwise
mark the events as nonspecific binding events. Nonspecific binding events will not show up in the grouped events. 
----------------Input-------------------------
 LocatStore,superdata, e.nzoom, e.FinalLocatSigma (default 0.5),e.nevent (default 5), e.SRCorrFactor (default 0.6) e.FinalLocatThresh (default 3) 
     PSFDataReduced is used. modify this as needed.
----------------Output-------------------------
 GroupLocat(1 : number of sites), LocatStore
     structure GroupLocat: .Centroid(y,x, sigmay, sigmax)  sigma being the dispersion of the group members
                .RawSites(y,x,frame, intensity;) sort column by frame number
     LocatStore column 7 is set to a flag for specific (1) and nonspecific binding (0).  
%}

%% 
function [GroupLocat, LocatStore] = LRG_SuperRes_GroupsfromSR(LocatStore, superdata, e)
tic
GroupLocat = struct('Centroid',[],'RawSites',[]);
if ~isfield(e, 'FinalLocatSigma')     % standard deviation of a binding site representing the localization accuracy.
    e.FinalLocatSigma = 0.5; disp('e.FinalLocatSigma = 0.5 pixel created'); %default 0.5 pixel ~32 nm ~ 75 nm resolution
end
if ~isfield(e,'nevent')
    e.nevent = 5; disp('e.nevent = 5 created'); % threshold for minimum SR intensity of specific binding
end
if ~isfield(e, 'SRCorrFactor')
    e.SRCorrFactor = 0.6; disp('e.SRCorrFactor = 0.6 created');  % shape of the SR spot, correlation factor comparing to Gaussian standard
end
if ~isfield(e, 'FinalLocatThresh')   % searching distance of a event to a final location is FinalLocatThresh*FinalLocatSigma
    e.FinalLocatThresh = 3; disp('e.FinalLocatThresh = 3 sigma created');
end

%% generate the 2D gaussian function 
% exp((-x^2-y^2)/(2c^2))
Func=[];
n = e.nzoom; c = e.FinalLocatSigma*e.nzoom;
for i=-n:n
    for j=-n:n
        Func=[Func;exp((-i^2-j^2)/2/c^2)];
    end
end
Func=Func/std(Func);  


%% find the final locations 
% use the eye pick function written by Bo Shuang@Rice Univ. The locations ordered with decrease of number of event at the location.
superdatacopy = superdata;
a=size(superdata,1);
digr=e.nzoom;
meanbg=0;
marker=[];
while max(superdatacopy(:))>= e.nevent
    [~,q]=max(superdatacopy(:));
    row=ceil(q/a);row1=max(row-digr,1);row2=min(row+digr,a);
    clm=q-(row-1)*a;clm1=max(clm-digr,1);clm2=min(clm+digr,a);
    digpart=superdata(clm1:clm2,row1:row2);
    digpart=digpart(:);
    jdg=cov(Func,digpart)/std(digpart); % std(Func) = 1. no need to /std(Func))
    if jdg(2)> e.SRCorrFactor
        marker=[marker; clm,row];
    end
    superdatacopy(clm1:clm2,row1:row2)=meanbg;
end

if isempty(marker)==1
    disp('no binding cite found for the threshold.'); return;
else
    disp('Final locations found');
end
  % the final value is the threshold of frames observed at the location.
  % eg. 5 means this location has at least 5 frames have event.
  % samelocation(col, row).
%figure; imagesc(superdata);hold on; plot(marker(:,2), marker(:,1),'gx','MarkerSize',10,'LineWidth',2);
sameLocat = marker./e.nzoom;  % transfer to normal matrix
disp('find final locations from super-resolution image:');
toc


%% find the group members (threshold used)
% screen events ro find the closest final location. If the distance is
% <e.FinalLocatThresh*e.FinalLocatSigma, put the event in the group of the final location.
% Otherwise, mark the events as nonspecific binding by adding a column in LocatStore.
tic
numFL = size(sameLocat,1);
for i = 1 : numFL
    GroupLocat(i).Centroid = sameLocat(i,:);
end

% here set the threshold
disThreshSq = (e.FinalLocatThresh*e.FinalLocatSigma)^2; % 3 sigma 99.73% confidence
for f = 1:e.nframes
    if ~isempty(LocatStore(f).PSFfinal)
    for i = 1 : size(LocatStore(f).PSFfinal,1) %screening each event
        event = ones(numFL,2);
        event(:,1) = event(:,1).*LocatStore(f).PSFfinal(i,1);
        event(:,2) = event(:,2).*LocatStore(f).PSFfinal(i,2);
        EFdistance = sameLocat - event;
        EFdistanceSq = sum(EFdistance.^2,2);
        [mindis, index] = min(EFdistanceSq);  %find the closest final location for the event
        if mindis <= disThreshSq  % if the event belongs to a final location 
            LocatStore(f).PSFfinal(i,7) = 1; %put a marker on the event says specific
            GroupLocat(index).RawSites = [GroupLocat(index).RawSites; LocatStore(f).PSFfinal(i,1:2), f, LocatStore(f).PSFfinal(i,5)];
            % asign the event to the group 
        else
             LocatStore(f).final(i,7) = 0; %put a marker on the event says nonspecific
        end
    end
    end
end

for i = 1 : numFL
    GroupLocat(i).Centroid(1,3) = std(GroupLocat(i).RawSites(:,1));  
    GroupLocat(i).Centroid(1,4) = std(GroupLocat(i).RawSites(:,2));
end
disp('find the raw locations for the groups:');
toc

%% self test function run in commend window
%{
fl = 1;  % The fl th final location
for i = 1 : size(GroupLocat(fl).RawSites,1)
   figure; imagesc(Data0(:, :, GroupLocat(fl).RawSites(i,3)));
   hold on; plot(GroupLocat(fl).Centroid(1,2), GroupLocat(fl).Centroid(1,1), 'gx','MarkerSize',10,'LineWidth',2);
   plot(GroupLocat(fl).RawSites(i,2), GroupLocat(fl).RawSites(i,1), 'ro')
end
%}