%% RM - LRG Super Res Main Run
%Use LRG_SuperRes to identify particles over many
%frames (bining)
function [thrData, locatStore] = RM_Main_LRGSuperRes_Function(thrData)

% Not needed for function version
%{ 
% numFiles = 1;
% 
% 
% fileNames = "";
% filePaths = "";
% for i=1:numFiles
%     [file, path] = uigetfile('.tif','Select a TIF image file')
%     fileNames = [fileNames, file];
%     filePaths = [filePaths, path];
% end


startloc = '\\129.22.135.181\Test\BenjaminWellnitz\';
numFiles = 1;
multiSelect = 'false';

fileNames = "";
filePaths = "";

if strcmp(multiSelect,'true')==0
    for i=1:numFiles
        [file, path] = uigetfile(startloc,...
            'Select a TIF image file')
        fileNames = [fileNames, file];
        filePaths = [filePaths, path];
    end
else
   [file,path] = uigetfile(startloc,...
   'Select One or More Files', ...
   'MultiSelect', 'on');
    numFiles = size(file,2);
    for i=1:numFiles
        fileNames = [fileNames,file(i)];
        filePaths = [filePaths, path];
    end
end


disp('All Files Selected')
%}

%% User Defined Parameters - Run section for 2D and 3D
% General Parameters
e.codedir='C:\Users\Kisleylab\Desktop\UsersTemp\RicardoMN\2022_06_02\'; 
e.runsuperres='true'; % (true or false) false = find kinetics using diffraction limited data. true = find kinetics using superlocalized data
e.startframe=1; % The first frame to analyze 
e.stopframe=size(thrData, 3);% The last frame to analyze
e.nframes=e.stopframe-e.startframe+1;
e.xmin=1;
e.xmax=size(thrData, 1);
e.ymin=1;
e.ymax=size(thrData, 2);
e.pixelSize=109; % nm

% Data Parameters
e.path='Your Path'; % Not needed for function version
e.filename='yaxis'; % Name of the file to read in if reading in a single file. Do not include extension. 
e.loadsif='false'; % true = load data from a .sif file (if this is the first analysis for instance) false = data will be loaded from a .mat file

% Particle Identification Parameters
e.BackgroundThreshold = 1; %multiplicative minimum treshold over local background for particle ID (before adding sigma)
e.selectROI = 'false'; %prompts uuser to draw an ROI for each file
e.wasCut = 'false';

e.SNR_enhance=true; % Boost the signal to noise ratio. ({true} or false)
e.local_thd= true; % Use local background. ({true} or false)
e.sigma=3; % how many std to add up as a threshold (Default = 3)-  higher is less lenient
e.wide2=3; % pixels Local maximum cutoff distance (Real # between 1 and 5. Default = 3) - higher is less lenient
e.Gauss_width=5; % pixels Width of the PSF Gaussian. (real # beween 1 and 3. Default is 2) - higher is less lenient
e.fitting='rc'; % Fitting method to use. ({'rc'} 'gs' 'el') 'rc' = radial symmetry, 'gs' = gaussian fitting, 'el' = Euler fitting
e.test=false; % Generate a figure showing identified partilce locations in the first frame ('true' or {'false'}) Careful, this will be done for each frame. 

% Diffraction Limited Event Grouping Parameters
e.sameLocation=2; % pixels Distance threshhold for grouping particles in diffraction limited data (Real # Default is 2)

% SuperResolution Parameters
e.nzoom=20; % Super resolution zoom value
e.sigmarad=25; %nm sigma radius used to generate superresolution data. 

% SuperResolution Event Grouping Parameters
e.FinalLocatSigma=0.5;%the standard deviation of the single binding site - used for model 2D peak (0.5)
e.nevent=5;% minimum peak intensity of the spot - to distinguish specific and non-specific intrs (5)
e.SRCorrFactor=0.6; %the minimum cross correlation factor between a spot and the standard Gaussian peak (0.6)
e.FinalLocatThresh=3;%how many FinalLocatSigma are used to rule out spfc vs non-spfc (3)

% Parameters for kinetics analysis
e.kinetiC = 'false'; %whether or not to calculate kinetics
e.dataSpace=0; %ms Dead time of the detector
e.dataTime=10; %ms Integration time
e.datafreq=e.dataSpace+e.dataTime; %ms frame rate
e.chngpt=0; %calculate kinetics by change point (=1) or counting molec. ID (=0)?
e.CPstate=0; % program ID change point states (0) or # for user defined states (>0)

%% Run LRG_SuperRes 

addpath(strcat(pwd,'\LRG_SuperRes_Kinetics_Final\RMN Codes'));

[thrData, locatStore] = RM_LRG_SuperRes_Run(e, thrData);  

return