clear; close all hidden; clc;warning off
%% User Input
% Start Location of file prompter
startloc = '\\129.22.135.181\Test\Surajit\Microscope\01172024';%'\\129.22.135.181\Test\StephanieKramer\LSM Data\20240415\Area 3';
FinalFileName = 'A8B1_R2_0';

% %% Input Data Settings %% %

% Microscope Set Up
pixelsize = 0.102; % In micro meters (0.102 IX83; 0.160 IX73; 0.180 LSM); needed to accurately calculate D
dT = 0.05; % Time between frames in s; needed to accurately calculate D
zstep = 0.1; %Z-step between frames (um)

% File format
useTiffFile = 1; % Whether you are using a .tiff file (other option is a .mat file) (1 = yes, 0 = no)
tiffReadVol = 1; % If using tiff file and using matlab 2021b or newer, tiffReadVolume is faster (1 = yes)

% Files to load
multiSelect = 'false'; %set to true if selecting multiple files in the exact same folder (no subfolders)
numberFCSSOFIRuns = 7; % Number of fcsSOFI runs (how many different files for fcsSOFI)
numberFiles = 1;% Number of files to cancatnate together for each fcsSOFI image (Will combine files if more than one)

% File info
framesLength = 1000; % Total length of each file

% Region of interest in pixels (of all files added together)
ymin = 1;
ymax = 512;
xmin = 1;
xmax = 512;
tmin = 1; % Start frame
tmax = 1000; % End frame

% R Squared Cut off for cumulative distribution of D, beads=0.95, 76kDa=0.8, 2000kDa=0.9, BSA=0.88
R2cutoff = 0;

% SOFI scaling, Normalized (0-1) data will be cut off and re-nomarlized at these values
satMin = 0;
satMax = 1;%normally 1, 0.02 for nano
crossSatMax = satMax + 0;

% %% Sample and Anylisis Settings %% %

% Choose type of diffusion (1 = Brownian, 2 = 2-Comp Brownian, 3 = Anomalous, ...
        ... 4 = Brownian 1 Comp with tau, 5 = 1-comp Brownian with tau and A, ...
        ... 6 = Anomalous with Tau and alpha)
type = 4;

% Initial Condition For Fitting
D_stp = 10;     % Diffusion Coeficient (in micro m^2 / s)
D2_stp = 1e6;   % 2d Diffusion Coeficient (For two compnent)
A_stp = 1;      % A Value (for type 1 and 4)
alpha_stp = .9; % Alpha Value (for type 3 and 6)

% Alpha Limits (Values cut off at these points)
alpha_max = 1.2; % Maximum value of alpha allowed to appear on alpha map
alpha_min = 0;

% Diffusion Limits (Both D1 and D2 values cut off at these points)
diffusionMin = 0; % micro m^2 / s
diffusionMax = Inf;

% Deconvolution on SOFI Image (1 = yes, 0 = no)
doDecon = 1;
doDecon3D = 0;

% Bin Size for fcs Binning. Allow for faster D detection
binSize = 1;

% 3D Reconstruction? (1 = yes)
Reconstruction = 0;

% %% Result Settings %% %

% Plot figures? (1 = yes)
plotfigures = 1;

% Save data files? (1 = yes)
savethedata = 1; 

% END USER INPUT
fprintf('Running...\n');

%% Paths
addpath(strcat(pwd, '\gpufit\Debug\matlab'))
addpath(genpath(strcat(pwd, '\fcsSOFI_external_functions')))

% Select File Paths
fileNames = cell(1, numberFiles*numberFCSSOFIRuns); 
paths = cell(1, numberFiles*numberFCSSOFIRuns);

if strcmp(multiSelect,'true')==0
    for nFile = 1:(numberFiles*numberFCSSOFIRuns)
    [fileName, path] = uigetfile(startloc, '*');
    fileNames(1, nFile) = {fileName};
    paths(1, nFile) = {path};
    addpath(path);
    end
else
        [fileName, path] = uigetfile(startloc, ...
   'Select One or More Files', ...
   'MultiSelect', 'on');

    for nFile = 1:(numberFiles*numberFCSSOFIRuns)

    fileNames(1, nFile) = {fileName(nFile)};
    paths(1, nFile) = {path(nFile)};
    addpath(path);
    end
end

for run = 1:numberFCSSOFIRuns
%% Convert Tif to Mat / Load Data

% Index of first file in this run
startFileIndex = 1 + (run-1)*numberFiles;

% Alows for ultiple files to be added together
if useTiffFile==1
    % Start global timer after user selects files
    timeStart = tic;
    
    % Read all the files
    Data = tifRdFunc(fileNames{1, startFileIndex}, ...
                     paths{1, startFileIndex}, framesLength, tiffReadVol);
    fprintf('Tiff file 1 loaded \n');
    for i = 2:numberFiles
        Data = cat(3, Data, tifRdFunc(fileNames{1, startFileIndex + i}, ...
                                      paths{1, startFileIndex + i}, ...
                                      framesLength, tiffReadVol));
        fprintf('Tiff file %g loaded \n', i);
    end
    
    filenm = extractBefore(fileNames{1, 1}, ".tif");
    
    % Save to mat converted file
    fname = char(strcat(filenm, '_Combined.mat'));
    save(fname, 'Data', '-v7.3')

else % If data was already converted
    % Start global timer after user selects files
    timeStart = tic;

    fname = fileNames{1, startFileIndex};
    load(char(fname));
        
end

fprintf(strcat(char(fname), ' loaded\n'));

% Display time for file load
time = toc(timeStart);
timeOut = ['Loading File Complete, Execution Time: ', num2str(floor(time / 60)), ' Minutes, ', num2str(mod(time, 60)), ' Seconds'];
disp(timeOut);

%% Bin The Data

% Set ROI
Data = Data(ymin:ymax, xmin:xmax, tmin:tmax);

% Make sure bin size fits
remain = mod(size(Data), binSize);
Data = Data(1:end-remain(1), 1:end-remain(2), :);

% Bin The Data into bin sizes to allow for faster D detection
% A little confusing to follow, but sums values in each bin
fcsData = reshape(Data, binSize, [], size(Data, 3));
fcsData = sum(fcsData, 1);
fcsData = reshape(fcsData, size(Data,1) / binSize, [], size(Data, 3));
fcsData = pagetranspose(fcsData);
fcsData = reshape(fcsData, binSize, [], size(Data, 3));
fcsData = sum(fcsData, 1);
fcsData = reshape(fcsData, size(Data, 2) / binSize, [], size(Data, 3));
fcsData = pagetranspose(fcsData);

%% Background Subtraction

% Start background subtration timer
timeBack = tic;

thrDataFcs = zeros(size(fcsData));
thrData = zeros(size(Data));

for i = 1:size(fcsData, 3) %i is the frame number
        Bkg = LRG_SuperRes_LocalThrMap(fcsData(:, :, i), true); %local background calcualted with LRG code
        thrDataFcs(:, :, i) = double(fcsData(:, :, i)) - double(Bkg); %background subtraction step

        Bkg = LRG_SuperRes_LocalThrMap(Data(:, :, i), true); %local background calcualted with LRG code
        thrData(:, :, i) = double(Data(:, :, i)) - double(Bkg); %background subtraction step
end
DataCombined = thrData;
fcsData = thrDataFcs;
%{
if Reconstruction == 1 
    for i = 1:size(fcsData, 3) %i is the frame number
        Bkg = LRG_SuperRes_LocalThrMap(fcsData(:, :, i), true); %local background calcualted with LRG code
        thrDataFcs(:, :, i) = double(fcsData(:, :, i)) - double(Bkg); %background subtraction step

        Bkg = LRG_SuperRes_LocalThrMap(Data(:, :, i), true); %local background calcualted with LRG code
        thrData(:, :, i) = double(Data(:, :, i)) - double(Bkg); %background subtraction step
    end
    DataCombined = thrData;
    fcsData = thrDataFcs; 
else
    for i = 1:size(fcsData, 3) %i is the frame number
        Bkg = LRG_SuperRes_LocalThrMap(fcsData(:, :, i), true); %local background calcualted with LRG code
        thrDataFcs(:, :, i) = double(fcsData(:, :, i)) - double(Bkg); %background subtraction step
    end
    DataCombined = Data;
    fcsData = thrDataFcs;
end
%}
%% %%%%%%%%%%%%%%%%%%%% STEP 1: blink_AConly (SOFI) %%%%%%%%%%%%%%%%%%%%%%%%%

% Start timer before SOFI step
timeSofi = tic;

%% Calculate the correlation (2-3th orders, AC and XC)
addpath(genpath(strcat(pwd,'\fcsSOFI_external_functions\sofiFunctions')))
[crossSofiMap, ~, ~, sigma] = crossSofi(DataCombined);
[sofiMap, ~, ~, ~, ~, ~, ~] = autoSofi(DataCombined);

% Produce average image
avgim = mean(DataCombined, 3);

%% Deconvolution
%{
if doDecon == 1
        [deconAC, deconXC, deconAvg] = decon(avgim, {sofiMap}, {crossSofiMap}, sigma);
        sofiMapDecon = deconAC{1};
        crossSofiMapDecon = deconXC{1};
else
        sofiMapDecon = sofiMap;
        crossSofiMapDecon = crossSofiMap;
end
%}
if doDecon == 1
    if Reconstruction == 0
        [deconAC, deconXC, deconAvg] = decon(avgim, {sofiMap}, {crossSofiMap}, sigma);
        sofiMapDecon = deconAC{1};
        crossSofiMapDecon = deconXC{1};
    else
        sofiMapDecon = sofiMap;
        crossSofiMapDecon = crossSofiMap;
    end
else
    sofiMapDecon = sofiMap;
    crossSofiMapDecon = crossSofiMap;
end

% display execution time of SOFI step
time = toc(timeSofi);
timeOut = ['Sofi Complete, Execution Time: ', num2str(floor(time / 60)), ' Minutes, ', num2str(mod(time, 60)), ' Seconds'];
disp(timeOut);

%% %%%%%%%%%%%%%%%%%%%% STEP 2: BinFitData (fcs) %%%%%%%%%%%%%%%%%%%%%%%%%

% start timer for fcs step
timeFcs = tic;

[~, ~, ~, sigmaBin] = crossSofi(fcsData);

pixelsize = pixelsize*binSize;
%sigmaBin = sigma/binSize;

% Calculate Auto Correlation Curves
[autoCorrelations] = fcsCorrelate(fcsData);

% Bin Correlation Curves 
maxLag = size(autoCorrelations, 1);
lags = 1:maxLag;
ddwell = 1;
[logBinLags, logBinAC] = logbindata(lags, autoCorrelations, ddwell, maxLag);


% %% Set Up Curve Fit %% %
% Data To Fit
fitData = single(logBinAC(2:end, :) ./ max(logBinAC(2:end, :), [], 1));

% Weights on Data Points
weights = ones(size(fitData));
weights(fitData < 0) = 0;
weights = single(weights);

% Model to fit with
model_id = [ModelID.BROWNIAN_1COMP; ModelID.BROWNIAN_2COMP; ModelID.ANOMALOUS;...
    ModelID.BROWNIAN_1COMP_NORM; ModelID.BROWNIAN_1COMP_TAUA; ...
    ModelID.ANOMALOUS_2PARAM_TAUA]; model_id = model_id(type);

% Initial Perameters
initialA = ones(1, size(fitData, 2)) * A_stp;
initialBack = mean(fitData(round(3*end/4:end), :), 1);
initialTd = ones(1, size(fitData, 2)) * ((pixelsize ^ 2) / (D_stp * 4));
initialTd2 = ones(1, size(fitData, 2)) * ((pixelsize ^ 2) / (D2_stp * 4));
initialMax = max(fitData, [], 1);
initialAlpha = ones(1, size(fitData, 2)) * alpha_stp;
switch type
    case 1 % Brownian
        initial_parameters = cat(1, initialA, initialBack, initialTd);
        tuaDIndex = 3;
    case 2 % Brownian 2 component
        initial_parameters = cat(1, initialMax, initialMax, initialBack, initialTd, initialTd2);
        tuaDIndex = 4;
    case 3 % Anomalous
        initial_parameters = cat(1, initialMax.*2, initialBack, initialTd, initialAlpha);
        tuaDIndex = 3;
        alphaIndex = 4;
    case 4 % Brownian Norm
        initial_parameters = initialTd;
        tuaDIndex = 1;
    case 5 % Bronian With Tua and A
        initial_parameters = cat(1, initialA, initialTd);
        tuaDIndex = 2;
    case 6 % Anomalous Tua and A
        initial_parameters = cat(1, initialTd, initialAlpha);
        tuaDIndex = 1;
        alphaIndex = 2;
end
initial_parameters = single(initial_parameters);

% Tolerance
tolerance = 1e-3;

% Maximum number of iterations per pixel
max_n_iterations = 100000;

% Estimator id
estimator_id = EstimatorID.LSE;

% Lag Times (x)
fitTimePoints = single(logBinLags(2:end) .* dT);


% %% Fit Data %% %

% Compute Fit
[parameters, states, chi_squares, n_iterations, gputime] = gpufit(fitData, weights, ...
    model_id, initial_parameters, tolerance, max_n_iterations, [], estimator_id, fitTimePoints);

fit_time = gputime;


% %% Extract Fit Info %% %

% Find Converged Fits
converged = states == 0; 

% Calculate fit values at the data time points
fitTimePoints = repmat(fitTimePoints, 1, size(fitData, 2));
switch type
    case 1 % Brownian
        model_fit = parameters(1, :) .* (1./(1+(fitTimePoints./parameters(3, :)))) + parameters(2, :);
    case 2 % Brownian 2 component
        model_fit = parameters(1, :) .* (1./(1+(fitTimePoints./parameters(4, :)))) + parameters(2, :).*(1./(1+(fitTimePoints./parameters(5, :)))) + parameters(3, :);
    case 3 % Anomalous
        model_fit = parameters(1, :) .* (1./(1+(fitTimePoints./parameters(3, :)).^parameters(4, :))) + parameters(2, :); 
    case 4 % Brownian Norm
        model_fit = 1./(1+(fitTimePoints./parameters(1, :)));
    case 5 % Bronian With Tua and A
        model_fit = parameters(1, :).* (1./(1+(fitTimePoints./parameters(2, :)))); 
    case 6 % Anomalous Tua and A
        model_fit = (1./(1+(fitTimePoints./parameters(1, :)).^parameters(2, :)));
end

% Find pixel dimentions of fcs image
fcsSz = size(fcsData);
fcsSz = fcsSz(1:2);

% R-square
residuals = fitData - model_fit;
a = (fitData - model_fit) .^ 2 ./ model_fit;
a(isinf(a)) = 0;
R2 = 1 - sum(residuals .^ 2, 1) ./ sum(fitData .^ 2, 1);
R2map = reshape(R2, fcsSz);

% Chi-Squared Map
chiMap = reshape(chi_squares, fcsSz);

% Characteristic time
tauD = parameters(tuaDIndex, :); % in in seconds
tauDmap = reshape(tauD, fcsSz);
    
% Diffusion Coefficient
w = pixelsize * sigmaBin * 2.355; % Use PSFsample instead if you know the full width half mast;
D = (w .^ 2) ./ (4 * tauD); % In micro meters^2/s
D_corrected = abs(D);
D_corrected(D_corrected > diffusionMax) = diffusionMax;
D_corrected(D_corrected < diffusionMin) = diffusionMin;
D_corrected(~converged) = NaN;
Dmap_corrected = reshape(D_corrected, fcsSz);

% Second diffusion coefficient if using 2-component model
if type == 2
    tauD2 = parameters(5, :); % in in seconds
    tauD2map = reshape(tauD2, fcsSz);
    D2 = (w .^ 2) ./ (4 * tauD2);
    D2_corrected = abs(D2);
    D2_corrected(D2_corrected > diffusionMax) = diffusionMax;
    D2_corrected(D2_corrected < diffusionMin) = diffusionMin;
    D2_corrected(~converged) = NaN;
    D2map_corrected = reshape(D2_corrected, fcsSz);
end

% Alpha map if using anomalous model
if (type == 3) || (type == 6)
    alpha = parameters(alphaIndex, :);
    alpha_corrected = abs(alpha);
    alpha_corrected(~converged) = NaN;
    alpha_corrected(alpha_corrected < alpha_min ) = 0;
    alpha_corrected(alpha_corrected > alpha_max ) = alpha_max;
    alphaMap = reshape(alpha_corrected, fcsSz);
end

% Display execution time of fcs step
time = toc(timeFcs);
timeOut = ['FCS Complete, Execution Time: ', num2str(floor(time / 60)), ' Minutes, ', num2str(mod(time, 60)), ' Seconds'];
disp(timeOut);

%% %%%%%%%%%%%%%%%%%%%% STEP 3: DataVisualization (fcsSOFI) %%%%%%%%%%%%%%%%%%%%%%%%%

% start timer for fcsSOFI combination
timeCombine = tic;

%% Diffusion CDF Creation

% Only values with good R2 values used
DhighSOFIvaluesR2 = Dmap_corrected(R2 > R2cutoff);

% Reshape Dmap data into a vector
DhighSOFIvaluesR2 = unique(reshape(DhighSOFIvaluesR2, 1, []));
TimeArray = unique(DhighSOFIvaluesR2); % All posible values of diffusion data

% Produce Cumulative Distribution
[DWellFinal, IndexFinal] = cumuldist(DhighSOFIvaluesR2, TimeArray);

% Need to size cross sofi down to match fcs size
% Each fcs weight will be the total sum of sofi values in the fcs values location
ecdfSOFI = padarray(crossSofiMap, [2, 2], "both");
ecdfSOFI = padarray(ecdfSOFI, [1, 1], "post");
ecdfSOFI = binData(ecdfSOFI, 2*binSize);

[dVals, ecdfProb] = weightedECDF(Dmap_corrected, ecdfSOFI);

%% Filter Out Really Bad Fits in FCS Plot

% Filter out poor fits if you want to here with an R2cutoff
%R2cutoff = 0.5; %set R^2 cutoff (0-1); == 0, no filtering
DmapAlpha = ones(size(Dmap_corrected)); % Used For Plotting

Dmap_corrected(R2map < R2cutoff) = NaN;
DmapAlpha(isnan(Dmap_corrected)) = 0;

%% Set the limits of the SOFI image

% Normalize between [0 1]
sofiMap = rescale(sofiMap); % sofi no decon
sofiMapDecon = rescale(sofiMapDecon); % sofi decon
crossSofiMap = rescale(crossSofiMap); % Cross sofi no decon
crossSofiMapDecon = rescale(crossSofiMapDecon); % Cross sofi decon

sofiMapSat = sofiMap;
sofiMapDeconSat = sofiMapDecon;
crossSofiMapSat = crossSofiMap;
crossSofiMapDeconSat = crossSofiMapDecon;

% Adjust with sat max/min
sofiMapSat(sofiMap < satMin) = satMin;
sofiMapDeconSat(sofiMapDecon < satMin) = satMin;
crossSofiMapSat(crossSofiMap < satMin) = satMin;
crossSofiMapDeconSat(crossSofiMapDecon < satMin) = satMin;

sofiMapSat(sofiMap > satMax) = satMax;
sofiMapDeconSat(sofiMapDecon > satMax) = satMax;
crossSofiMapSat(crossSofiMap > crossSatMax) = crossSatMax;
crossSofiMapDeconSat(crossSofiMapDecon > crossSatMax) = crossSatMax;

% Re Normalize between [0 1]
sofiMapSat = rescale(sofiMapSat); % sofi no decon
sofiMapDeconSat = rescale(sofiMapDeconSat); % sofi decon
crossSofiMapSat = rescale(crossSofiMapSat); % Cross sofi no decon
crossSofiMapDeconSat = rescale(crossSofiMapDeconSat); % Cross sofi decon

%% Creating Larger Dmap Images

% Need to fit D data ontop of SOFI data with extra pixels
% Resize the binned data back up
sizedDmap = imresize(Dmap_corrected, size(Dmap_corrected).*binSize, 'nearest');

% Resize the binned data to the cross sofi dimentions 
crossDmap = imresize(Dmap_corrected, size(Dmap_corrected).*2.*binSize, 'nearest');
crossDmap = crossDmap(3:end-3, 3:end-3);

% display execution time of fcsSOFI combination
time = toc(timeCombine);
timeOut = ['Image Fusion Complete, Execution Time: ', num2str(floor(time / 60)), ' Minutes, ', num2str(mod(time, 60)), ' Seconds'];
disp(timeOut);

%% Figures

figureNumber = 1; % Used to keep track of figures to save

% The color map used for all the figures
customColorMap = magma(size(crossSofiMap, 2));%jet;

if plotfigures == 1 
    
    % Structure Results (Plus CDF)
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;

    % Average Image
    subplot(3, 3, 3); imagesc(avgim); colormap(gray); axis image;
    title('Average image'); set(gca, 'xtick', [], 'ytick', [])
%{    
    % AC SOFI Image
    subplot(3, 3, 2); imagesc(sofiMapSat); axis image
    title('AC G^2(0+\tau)'); set(gca, 'xtick', [], 'ytick', [])

    % AC Deconvolved SOFI Image
    subplot(3, 3, 5); imagesc(sofiMapDeconSat); axis image; 
    title('Deconvolved'); set(gca, 'xtick', [], 'ytick', [])
%}    
    % XC SOFI Image
    subplot(3, 3, 1); imagesc(crossSofiMapSat); axis image; 
    title('Cross SOFI super-resolution'); set(gca, 'xtick', [], 'ytick', []); colormap(gray)
    
    % XC Deconvolved SOFI Image 
    subplot(3, 3, 2); imagesc(crossSofiMapDeconSat); axis image; 
    title('Cross SOFI super-resolution with Decon'); set(gca, 'xtick', [], 'ytick', []); colormap(gray)
    
    % CDF Plot
    subplot(3, 3, 9);
    plot(DWellFinal, IndexFinal, 'k.')
    ylim([0 1])
    title("CDF")
    xlabel('Diffusion Coefficient (\mum^2s^{-1})')
    ylabel('Probability')
%{
    plot(dVals, ecdfProb, 'k.')
    ylim([0 1])
    title("Weighted eCDF")
    xlabel('Diffusion Coefficient (\mum^2s^{-1})')
    ylabel('Probability')
 %}
        
    % Dynamics Results
    %figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    
    % FCS Diffusion Map
    subplot(3, 3, 6);
    DFigure = imagesc(sizedDmap); axis image; title('FCS')
    DFigure.AlphaData = ~isnan(sizedDmap); colormap(customColorMap);
    c = colorbar; c.Label.String = 'D \mum^2s^{-1}';
    set(gca, 'Color', [0, 0, 0]) % Set background color to black
    set(gca, 'FontSize', 14); set(gca, 'xtick', [], 'ytick', []) % Removes axis tick marks
    
    % R-square Map
    subplot(3, 3, 8);
    imagesc(R2map); colorbar; axis image; title('R^2 map');
    
    % Tau Diffusion Map
    subplot(3, 3, 7);
    imagesc(tauDmap); colorbar; axis image; title('\tau_D map')
    
    % Second set of BinFitData subplots if using 2-comp model
    if type == 2
        figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
        
        % Tau Diffusion Map 2 Image
        imagesc(tauD2map); axis image; title('D2: \tau_D map')
    end
      
    % Alpha Map if using anomalous model
    if type == 3 || type == 6
        sofiBinarized = imbinarize(sofiMapDeconSat, 0.05);
        alphamap(sofiBinarized == 0) = 0;
       
        figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
        imagesc(alphamap); axis image; title('\alpha map')
        c = colorbar; c.Label.String = '\alpha'; c.Label.FontSize = 20;
        set(gca, 'xtick', [], 'ytick', [])
    end
%{
    % AC fcsSOFI figure creation 
    subplot(2, 3, 2);
    fcsSofiPlot = imagesc(sizedDmap); axis image; title('Combined fcsSOFI image')
    fcsSofiPlot.AlphaData = sofiMapSat .* ~isnan(sizedDmap); % Uses the SOFI data as a transparency map
    colormap(customColorMap); c = colorbar; c.Label.String = 'D \mum^2s^{-1}';
    set(gca, 'Color', [0, 0, 0]) % Set background color to black
    set(gca, 'FontSize', 14); set(gca,'xtick',[],'ytick',[])
    
    % AC Decon fcsSOFI figure creation
    subplot(2, 3, 5);
    fcsSofiPlot = imagesc(sizedDmap); axis image; title('Combined fcsSOFI image with Decon')
    fcsSofiPlot.AlphaData = sofiMapDeconSat .* ~isnan(sizedDmap); % Uses the SOFI data as a transparency map
    colormap(customColorMap); c = colorbar; c.Label.String = 'D \mum^2s^{-1}';
    set(gca, 'Color', [0, 0, 0]) % Set background color to black
    set(gca, 'FontSize', 14); set(gca,'xtick',[],'ytick',[])
%}       
    % XC fcsSOFI figure creation 
    subplot(3, 3, 4);
    fcsSofiPlot = imagesc(crossDmap); axis image; title('Combined Cross fcsSOFI image')
    fcsSofiPlot.AlphaData = crossSofiMapSat .* ~isnan(crossDmap); % Uses the SOFI data as a transparency map
    colormap(customColorMap); c = colorbar; c.Label.String = 'D \mum^2s^{-1}';
    set(gca, 'Color', [0, 0, 0]) % Set background color to black
    set(gca, 'FontSize', 14); set(gca,'xtick',[],'ytick',[])
    
    % XC Decon fcsSOFI figure creation 
    subplot(3, 3, 5);
    fcsSofiPlot = imagesc(crossDmap); axis image; title('Combined Cross fcsSOFI image with Decon')
    fcsSofiPlot.AlphaData = crossSofiMapDeconSat .* ~isnan(crossDmap); % Uses the SOFI data as a transparency map
    colormap(customColorMap); c = colorbar; c.Label.String = 'D \mum^2s^{-1}';
    set(gca, 'Color', [0, 0, 0]) % Set background color to black
    set(gca, 'FontSize', 14); set(gca,'xtick',[],'ytick',[])
end

%% save data (Changing the naming convention 10/03/23)
if savethedata == 1 
    
    % The Variables to save
    date = datestr(now,'yyyymmdd');
    folderName = strcat(FinalFileName, '_AnalyzedOn_', date, '_', num2str(run));

    if type ~= 2
        D2map_corrected = NaN;
    end
    
    if type ~= 3
        alphamap = NaN;
    end
    
    % Creates file names
    figureFileName = strcat(folderName, '.fig');
    dataFileName = strcat(folderName, '.mat');
    mkdir(folderName);
        
    % Saves the figures and files
    savefig(figureArray, figureFileName);
    save(dataFileName, 'fitData', 'fitTimePoints', 'model_fit', 'parameters', ...
        'sofiMap', 'sofiMapDecon', 'crossSofiMap', 'crossSofiMapDecon', ...
        'sofiMapSat', 'sofiMapDeconSat', 'crossSofiMapSat', 'crossSofiMapDeconSat',...
        'Dmap_corrected', 'sizedDmap', 'crossDmap', 'chiMap', ...
        'dVals', 'ecdfProb', 'sigma', 'sigmaBin', 'binSize', ...
        'satMax', 'crossSatMax', 'satMin', 'dT', 'customColorMap', ...
        'D2map_corrected', 'alphamap', '-v7.3');

    % Moves the files into the folder created
    movefile(figureFileName, folderName);
    movefile(dataFileName, folderName);

    if useTiffFile
        movefile(fname, folderName);
    end

    fileList(:,run) = {dataFileName};

end   

%% total computation time
time = toc(timeStart);
timeOut = ['Total Execution Time: ', num2str(floor(time / 60)), ' Minutes, ', num2str(mod(time, 60)), ' Seconds'];
disp(timeOut);

fit_timeOut = ['Total Time in GPU Fit: ', num2str(floor(fit_time / 60)), ' Minutes, ', num2str(mod(fit_time, 60)), ' Seconds'];
disp(fit_timeOut);

clear
end

%%  Vold3d plotting
if Reconstruction == 1 
    ReconFileName = '3DReconstruction_';
           
    for stack = 1:numberFCSSOFIRuns
        dataFilePath = strcat('.\', extractBefore(fileList{stack}, '.mat'), '\', fileList{stack});

        loadStack = load(dataFilePath);
                        
        %Creating XC Non-Decon Stack
        acrossDmap = loadStack.crossDmap.*(loadStack.crossSofiMapSat .* ~isnan(loadStack.crossDmap));
        acrossDmapAll(:,:,stack) = acrossDmap;
        acrossDmapAll(isnan(acrossDmapAll))=0;
        
        %Creating XC Decon Stack
        acrossDmapDecon = loadStack.crossDmap.*(loadStack.crossSofiMapDeconSat .* ~isnan(loadStack.crossDmap));
        acrossDmapAllDecon(:,:,stack) = acrossDmapDecon;
        acrossDmapAllDecon(isnan(acrossDmapAllDecon))=0;
       
    end

    %Correct for the LS microscope offset
    offsetValY = 5.4*zstep; %~12 px for a 5 um step
    offsetValX = 1.1*zstep; %~5 px for a 50 um shift
        
    %XC Non-Decon data
    thrDataCor = double(acrossDmapAll);
    transVect = [offsetValY offsetValX 0];
    transData = imtranslate(thrDataCor(:,:,1),transVect(1:2));
    
    for n = 2:size(thrDataCor,3)
        transVect(1) = transVect(1) + offsetValY;
        transVect(2) = transVect(2) + offsetValX;
        transData(:,:,n) = imtranslate(thrDataCor(:,:,n),transVect(1:2));
    end
    if transVect(3) ~= 0
        transDatab = imtranslate(squeeze(transData(1,:,:)),[0 0 transVect(3)]);
        transDatab = permute(transDatab,[3 1 2]);
        for n = 2:size(transData,1)
            transDatab(n,:,:) = imtranslate(squeeze(transData(n,:,:)),[0 0 transVect(3)]);
        end
        for n = 2:size(transData,2)
            transDatab(:,n,:) = imtranslate(squeeze(transData(:,n,:)),[0 0 transVect(3)]);
        end
        transData = transDatab;
    end
    
    %XC Decon data
    thrDataCorDecon = double(acrossDmapAllDecon);
    transVectDecon = [offsetValY offsetValX 0];
    transDataDecon = imtranslate(thrDataCorDecon(:,:,1),transVectDecon(1:2));
    
    for n = 2:size(thrDataCorDecon,3)
        transVectDecon(1) = transVectDecon(1) + offsetValY;
        transVectDecon(2) = transVectDecon(2) + offsetValX;
        transDataDecon(:,:,n) = imtranslate(thrDataCorDecon(:,:,n),transVectDecon(1:2));
    end
    if transVectDecon(3) ~= 0
        transDatabDecon = imtranslate(squeeze(transDataDecon(1,:,:)),[0 0 transVectDecon(3)]);
        transDatabDecon = permute(transDatabDecon,[3 1 2]);
        for n = 2:size(transDataDecon,1)
            transDatabDecon(n,:,:) = imtranslate(squeeze(transDataDecon(n,:,:)),[0 0 transVectDecon(3)]);
        end
        for n = 2:size(transDataDecon,2)
            transDatabDecon(:,n,:) = imtranslate(squeeze(transDataDecon(:,n,:)),[0 0 transVectDecon(3)]);
        end
        transDataDecon = transDatabDecon;
    end
    
    %Do 3D Deconvolution
    if doDecon3D == 1
        iterations = 10;
        beta = 0.1;
    
        % Generate 3D Gaussian PSF
        psf = fspecial3('gaussian', [size(transData,2), size(transData,2), size(transData,3)], sigmaBin);
        
        % Normalize PSF (optional)
        psft = psf / sum(psf(:));
        
        % iterative deconvolution
        % first guess of the object  = spatially filtered measured image
        CTF = fftshift(fftn(psft));
        measured = transData;
        objectk1 = measured;
            
        for k = 1:iterations
            fprintf('Iteration: %d\n', k)
    
            % assigning object_k from the previous iteration
            objectk = objectk1;
                        
            % Object_k = Fourier-transform of object_k
            Objectk = fftshift(fftn(objectk));
            
            % measured_k = object_k convolved with PSF
            Measuredk = Objectk .* CTF;
            measuredk = ifftshift(ifftn(ifftshift(Measuredk)));
            
                % normalization of measuredk
                a = max(max(max(abs(measured))));
                b = min(min(min(abs(measured))));
                a0 = max(max(max(abs(measuredk))));
                b0 = min(min(min(abs(measuredk))));
                measuredk = (measuredk - b0) * ((a - b)/(a0-b0)) + b;
                  
            % object_k+1 = object_k multiplied with measured divided with measured_k (multiplicative)
            objectk1 = objectk .* measured .* conj(measuredk) ./ (measuredk .* conj(measuredk) + beta);
            
        end
        object = objectk1;% The resulting object is the intermediate result object_k+1 from the last iteration
    else
        object = transDataDecon;
    end            
    customColorMap = magma(size(crossSofiMap, 2));

 %close all;
    %Plot the XC Non-Decon data
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    fcsSofi3D = vol3d('Cdata',squeeze(transData()),...
        'ZData',[0 zstep.*size(acrossDmapAll, 3)],...%sets the range (um) for the z-axis based on the number and size of steps
        'XData',[0 pixelsize.*size(Data, 2)],...%sets the range (um) for the x-axis
        'YData',[0 pixelsize.*size(Data, 1)]);%sets the range (um) for the y-axis
    axis image; title('3D fcsSOFI Reconstruction')
    colormap(customColorMap); c = colorbar; c.Label.String = 'D \mum^2s^{-1}';
    set(gca,'Color', [0, 0, 0])%Sets background of each slice to black
    set(gca,'GridColor','k'); view(45,45)
    ylabel('Y (um)','FontSize',20, 'Rotation',40); xlabel('X (um)','FontSize',20, 'Rotation',-40); zlabel('Z (um)','FontSize',20);
    set(gca,'LineWidth',2); set(gca, 'FontSize', 14);

    hold on
    grid on

    %Plot the XC Decon data
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    fcsSofi3D = vol3d('Cdata',squeeze(object()),...
        'ZData',[0 zstep.*size(acrossDmapAllDecon, 3)],...%sets the range (um) for the z-axis based on the number and size of steps
        'XData',[0 pixelsize.*size(Data, 2)],...%sets the range (um) for the x-axis
        'YData',[0 pixelsize.*size(Data, 1)]);%sets the range (um) for the y-axis
    axis image; title('3D fcsSOFI Decon Reconstruction')
    colormap(customColorMap); c = colorbar; c.Label.String = 'D \mum^2s^{-1}';
    set(gca,'Color', [0, 0, 0])%Sets background of each slice to black
    set(gca,'GridColor','k'); view(45,45)
    ylabel('Y (um)','FontSize',20, 'Rotation',40); xlabel('X (um)','FontSize',20, 'Rotation',-40); zlabel('Z (um)','FontSize',20);
    set(gca,'LineWidth',2); set(gca, 'FontSize', 14);

    hold on
    grid on

    hold all

    %% save data
    if savethedata == 1 
    
        % The Variables to save
        ReconFileName = strcat(extractBefore(folderName, '_AnalyzedOn_'), '_', ReconFileName, date);
  
        % Creates file names
        figureFileName = strcat(ReconFileName, '.fig');
        mkdir(ReconFileName);
    
        % Saves the figures and files
        savefig(figureArray, figureFileName);
   
        % Moves the files into the folder created
        movefile(figureFileName, ReconFileName);
   
    end   
end