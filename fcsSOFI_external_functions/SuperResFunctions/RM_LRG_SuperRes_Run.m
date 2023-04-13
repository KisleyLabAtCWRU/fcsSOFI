
function [Data1, LocatStore, Bkg] = RM_LRG_SuperRes_Run(e, thrData)
%modified function by Ricardo MN, CWRU, 2019-2022, Kisley Lab

% Not Needed for function version
%{
addpath(genpath(e.codedir))
cd(e.path)
%Added local threshold storing 2/7/21

if strcmp(e.loadsif,'true')
    disp('Converting .tif File')
    LRG_SuperRes_Read_Data(e) %moded to ROI cutting for loops
    disp('Converted')
    %e.loadsif = 'false';
    disp('Loading Data')
    load(strcat(e.path,e.filename,'.mat'),'Data1');
end
%Data1=Data1(e.ymin:e.ymax,e.xmin:e.xmax,:);

%Data1 = makeSquare(Data1); %make into square matrix using zeroes (RMN 6/15/21)

%Optional - take ROI of data
if strcmp(e.selectROI, 'true')
%     if strcmp(e.wasCut, 'true') %Data was already cut to ROI and saved
%         disp('Loading Data')
%         load(strcat(e.path,e.filename,'.mat'),'Data1');
%         %e.filename = strcat(e.filename,'_ROICut');
%     else %load data and cut to ROI
        disp('Loading Data')
        load(strcat(e.path,e.filename,'.mat'),'Data1');
        disp('Cutting ROI of Data')
        [Data1,ROItype] = takeROI_RM(Data1);
        e.xmin=1;
        e.xmax=size(Data1,1);
        e.ymin=1;
        e.ymax=size(Data1,2);
        save(strcat(e.filename,'_ROICut.mat'),'Data1')
%         e.filename = strcat(e.filename,'_ROICut');
        e.wasCut ='true';
%     end
else %not cutting an ROI
    if strcmp(e.wasCut,'true')
        disp('Loading Data')
        load(strcat(e.path,e.filename,'_ROIcut.mat'),'Data1');
        Data1 = Data1(:,:,e.startframe:1:e.stopframe);
        disp('Loaded')
        
    else
        if strcmp(e.loadsif,'false')
        disp('Loading Data')
        load(strcat(e.path,e.filename),'thrData');
        Data1 = thrData(:,:,e.startframe:1:e.stopframe);
        disp('Loaded')
        end
    end
end
%}

Data1 = thrData;

%figure(2)
for iframe=1:e.nframes
    Bkg = zeros(size(thrData));
    [LocatStore(iframe), Bkg(:, :, iframe)] = LRG_SuperRes_Particle_Identify_org(Data1(:, :, iframe), e);
    switch iframe
        case 1
            disp('Identifying and Fitting Particles Started')
        case round(e.nframes / 4)
            disp('Identifying and Fitting Particles 25% Complete')
        case round(2 * e.nframes / 4)
            disp('Identifying and Fitting Particles 50% Complete')
        case round(3 * e.nframes / 4)
            disp('Identifying and Fitting Particles 75% Complete')
    end
end

switch e.runsuperres
    case 'false'
        return %CAUTION - edited for specific script -RMN 3/25/2022
        %disp('Identifying Final Sites of Interest')
        %GroupLocat=L RG_SuperRes_FindFinalLocation(LocatStore, e);
        
        
    case 'true'
        disp('Calculating SuperResolution Image')
        superdata=LRG_SuperRes_GenerateSR(LocatStore,e);
        disp('Identifying Final Sites of Interest')
        [GroupLocat, LocatStore]=LRG_SuperRes_GroupsfromSR(LocatStore,superdata,e);
end

disp('Identifying and Fitting Particles 100% Complete')

% Kinetics not Used in fcsSOFI
%{
if strcmp(e.kinetiC, 'true')
    disp('Now Calculating Kinetics')
    if isempty(GroupLocat(1).Centroid)==0
        [GroupLocat, Ensemble]=LRG_SuperRes_Kinetics_CP(GroupLocat,e,Data1);
    else
        disp('No Sites Found!')
        disp('No kinetics to report')
        GroupLocat=[];
        Ensemble.Dwell =[];
        Ensemble.Assoc = []; 
    end
    %disp(strcat('Saving Data to file: ',e.filename,'_analyzed.mat'))
    %save(strcat(e.filename,'_analyzed.mat'),'LocatStore','e','Data1','GroupLocat','Ensemble') %Added back 'Ensemble',
else
    disp('Skip Calculating Kinetics')
    %[GroupLocat, Ensemble]=LRG_SuperRes_Kinetics_CP(GroupLocat,e,Data1);
    %disp(strcat('Saving Data to file: ',e.filename,'_analyzed.mat'))
    %save(strcat(e.filename,'_analyzed.mat'),'LocatStore','e','Data1')%,'GroupLocat')%,'Ensemble') %Added back 'Ensemble',
end
%}

return