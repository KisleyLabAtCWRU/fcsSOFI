function LRG_SuperRes_Run(e)
addpath(genpath(e.codedir))
cd(e.path)
if strcmp(e.loadsif,'true')
    disp('Converting .tif File')
    LRG_SuperRes_Read_Data(e)
end
disp('Loading Data')
load(strcat(e.path,e.filename,'.mat'),'Data1');
Data1=Data1(e.ymin:e.ymax,e.xmin:e.xmax,:);
for iframe=1:e.nframes
    disp(strcat('Identifying and Fitting Particles in Frame #',num2str(iframe)))
    LocatStore(iframe)=LRG_SuperRes_Particle_Identify(Data1(:,:,iframe),e);
end

switch e.runsuperres
    case 'false'
        disp('Identifying Final Sites of Interest')
        GroupLocat=LRG_SuperRes_FindFinalLocation(LocatStore,e);
        
    case 'true'
        disp('Calculating SuperResolution Image')
        superdata=LRG_SuperRes_GenerateSR(LocatStore,e);
        disp('Identifying Final Sites of Interest')
        [GroupLocat, LocatStore]=LRG_SuperRes_GroupsfromSR(LocatStore,superdata,e);
end
%disp('Skip Calculating Kinetics')
[GroupLocat, Ensemble]=LRG_SuperRes_Kinetics_CP(GroupLocat,e,Data1);
disp(strcat('Saving Data to file: ',e.filename,'_analyzed.mat'))
save(strcat(e.filename,'_analyzed.mat'),'LocatStore','GroupLocat','Ensemble','e','Data1') %Removed 'Ensemble',
return
