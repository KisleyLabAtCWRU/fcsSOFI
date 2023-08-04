clear; close all hidden; clc;
%% User Input
% Start Location of file prompter
startloc = '\\kisleylab1\test\BenjaminWellnitz\fcsSOFI Master';


% %% Input Data Settings %% %

% Microscope Set Up
pixelsize = 0.109; % In micro meters (IX83); needed to accurately calculate D
dT = 0.001; % Time between frames in s; needed to accurately calculate D

% Whether you are using a .tiff file (other option is a .mat file) (1 = yes, 0 = no)
useTiffFile = 1;

% If using tiff file and using matlab 2021b or newer, tiffReadVolume is faster (1 = yes)
tiffReadVol = 1;

% Number of files to load used together and length of each file (Will combine files if more than one)
numberFiles = 1;
framesLength = 40000;

% Region of interest in pixels (of all files added together)
ymin = 1;
ymax = 100;
xmin = 1;
xmax = 100;
tmin = 1; % Start frame
tmax = 40000; % End frame


% %% Sample and Anylisis Settings %% %

% Choose type of diffusion (1 = Brownian, 2 = 2-Comp Brownian, 3 = Anomalous, ...
        ... 4 = Brownian 1 Comp with tau, 5 = 1-comp Brownian with tau and A, ...
        ... 6 = Anomalous with Tau and alpha)
type = 4;

% Initial Condition For Fitting
D_stp = 50;    % Diffusion Coeficient (in micro m^2 / s)
D2_stp = 1e6;   % 2d Diffusion Coeficient (For two compnent)
A_stp = 1;      % A Value (for type 1 and 4)
alpha_stp = .9; % Alpha Value (for type 3 and 6)

% Alpha Limits (Values cut off at these points)
alpha_max = 1.2; % Maximum value of alpha allowed to appear on alpha map
alpha_min = 0;

% Diffusion Limits (Both D1 and D2 values cut off at these points)
diffusionMin = 0; % micro m^2 / s
diffusionMax = Inf;

% R Squared Cut off for cumulative distribution of D, beads=0.95, 76kDa=0.8, 2000kDa=0.9, BSA=0.88
R2cutoff = 0.90;

% Deconvolution on SOFI Image (1 = yes, 0 = no)
doDecon = 1;

% SOFI scaling, Normalized (0-1) data will be cut ff and re-nomarlized at these values
satMin = 0;
satMax = 1;
crossSatMax = satMax + 0;

% Bin Size for fcs Binning. Allow for faster D detection
binSize = 1;


% %% Result Settings %% %

% Plot figures? (1 = yes)
plotfigures = 1;

% Save data files? (1 = yes)
savethedata = 1; 

% Optional example single pixel curve fit plot (1 = yes)
examplecf = 1;
column_index = 10; % X coordinate
row_index = 10; % Y coordinate (also used for sofi cross sections)

% Use defualt color scheme (1 = yes)
defualtColors = 1;

% If not using defualt colors. Top and bottom colors for the color bar / color scale
red = [1 0 0]; green = [0 0.5 0]; blue = [0 0 1]; lime = [0 1 0]; cyan = [0 1 1]; yellow = [1 1 0];
magenta = [1 0 1]; maroon = [0.5 0 0]; olive = [0.5 0.5 0]; purple = [0.5 0 0.5]; teal = [0 0.5 0.5]; navy = [0 0 0.5];
topC = red;
botC = blue;

% Old User Input layout
%{
% Diffusion coefficient parameters
pixelsize = 0.109; % In micro meters (IX83); needed to accurately calculate D
PSFsample = 5; % In pixel; based off of PSF from moving samples
vPSFsample = PSFsample*2-1;
dT = 0.002; % Time between frames in s; needed to accurately calculate D

% Set PSF for deconvolution of sofi image
sigma = (PSFsample / 2.355) / (2 ^ 0.5); % Standart deviation of PSF in pixels
vSigma = (vPSFsample / 2.355) / (2 ^ 0.5);

% Number of files used together and length of each file
numberFiles = 1;
framesLength = 60000;

% Region of interest in pixels (of all files added together)
ymin = 1;
ymax = 25;
xmin = 1;
xmax = 25;
tmin = 1; % Start frame
tmax = 1000; % End frame

% Choose type of diffusion (1 = Brownian, 2 = 2-Comp Brownian, 3 = Anomalous, ...
        ... 4 = Brownian 1 Comp with tau, 5 = 1-comp Brownian with tau and A, ...
        ... 6 = Anomalous with Tau and alpha)
type = 4;

% Bin Size
binSize = 1;

% R Squared Cut off for cumulative distribution of D, beads=0.95, 76kDa=0.8, 2000kDa=0.9, BSA=0.88
R2cutoff = 0.95;

% Initial Condition For Fitting
D_stp = 1e5;
D2_stp = 1e6;
A_stp = 1;
alpha_stp = .9;

% Alpha threshold
alpha_max = 1.2; % Maximum value of alpha allowed to appear on alpha map
alpha_min = 0;

% Set the limits which the diffsion data will be trimmed to
diffusionMin = -Inf; 
diffusionMax = Inf;

% Number of fit iterations per pixel
number_fits = 1000;

% Plot figures? (1 = yes)
plotfigures = 1;

% Save data files? (1 = yes)
savethedata = 1; 

% Optional example single pixel curve fit plot (1 = yes)
examplecf = 1;
column_index = 10; % X coordinate
row_index = 10; % Y coordinate

% Deconvolution on SOFI Image (1 = yes, 0 = no)
doDecon = 1;

% SOFI scaling
satMin = 0;
satMax = 1;
crossSatMax = satMax + 0;

% Whether you are using a .tiff file (other option is a .mat file) (1 = yes, 0 = no)
useTiffFile = 0;

% If using tiff file and using matlab 2021b or newer, tiffReadVolume is faster (1 = yes)
tiffReadVol = 0;

% Use already background subtracted data. Must be using a mat file if yes (1 = yes)
useBCData = 0;

% Use defualt color scheme (1 = yes)
defualtColors = 1;

% If not using defualt colors. Top and bottom colors for the color bar / color scale
red = [1 0 0]; green = [0 0.5 0]; blue = [0 0 1]; lime = [0 1 0]; cyan = [0 1 1]; yellow = [1 1 0];
magenta = [1 0 1]; maroon = [0.5 0 0]; olive = [0.5 0.5 0]; purple = [0.5 0 0.5]; teal = [0 0.5 0.5]; navy = [0 0 0.5];
topC = red;
botC = blue;
%}

% END USER INPUT
fprintf('Running...\n');

%% Paths
addpath(strcat(pwd, '\gpufit\Debug\matlab'))
addpath(genpath(strcat(pwd, '\fcsSOFI_external_functions')))

%% Convert Tif to Mat / Load Data
% Alows for ultiple files to be added together
if useTiffFile
    
    fileNames = cell(1, numberFiles); paths = cell(1, numberFiles);
    % Select all the files
    for i = 1:numberFiles
        [fileName, path] = uigetfile(startloc, '*.tiff');
        fileNames(1, i) = {fileName};
        paths(1, i) = {path};
        addpath(path);
    end
    
    % Start global timer after user selects files
    timeStart = tic;

    % Read all the files
    Data = tifRdFunc(fileNames{1, 1}, paths{1, 1}, framesLength, tiffReadVol);
    fprintf('Tiff file 1 loaded \n');
    for i = 2:numberFiles
        Data = cat(3, Data, tifRdFunc(fileNames{1, 1}, paths{1, 1}, framesLength, tiffReadVol));
        fprintf('Tiff file %g loaded \n', i);
    end
    
    filenm = extractBefore(fileNames{1, 1}, ".tif");
    
    % Save to mat converted file
    fname = strcat(filenm, '_Combined.mat');
    save(fname, 'Data', '-v7.3')

else % If data was already converted
    [fileName, path] = uigetfile(startloc, '*');

    % Start global timer after user selects files
    timeStart = tic;

    fname = fileName;
    addpath(path);
    load(fileName);
    
    % No Background Option
    %{
    if useBCData
        Data = thrData;
    end
    %}
end

fprintf(strcat(fname, ' loaded\n'));

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

% Display time for background subtration
time = toc(timeBack);
timeOut = ['Background Subtration Finished, Execution Time: ', num2str(floor(time / 60)), ' Minutes, ', num2str(mod(time, 60)), ' Seconds'];
disp(timeOut);

%% %%%%%%%%%%%%%%%%%%%% STEP 1: blink_AConly (SOFI) %%%%%%%%%%%%%%%%%%%%%%%%%

% Start timer before SOFI step
timeSofi = tic;

%% start code and setup data 

%{
% Set ROI
DataCombined = DataCombined(ymin:ymax, xmin:xmax, tmin:tmax);

% Make sure bin size fits
remain = mod(size(DataCombined), binSize);
DataCombined = DataCombined(1:end-remain(1), 1:end-remain(2), :);
%}

% Produce average image
avgim = mean(DataCombined, 3);

% Old fcs implimentation
%{
% Reshape 2D image to 1D for analysis
% shapes columnwise into a 1x(ymax-ymin+1)(xmax-xmin+1)
DataVector = zeros(tmax,(xmax-xmin+1)*(ymax-ymin+1));
for i = 1:size(DataCombined, 3)
    DataVector(i, :) = reshape(DataCombined(:, :, i), 1, ...
        size(DataCombined(:, :, i), 1) * size(DataCombined(:, :, i), 2));
end

% Only upper left points are run through analysis to avoid repetition
[innerpts] = getUpperLeft(ymax - ymin + 1, xmax - xmin + 1, 1);

%% Calculate the correlation (2-4th orders, AC and XC)
[ACXC_all] = CalcCorr(innerpts, DataVector); % Calculate
%}

%% Calculate the correlation (2-3th orders, AC and XC)
[crossSofiMap, ~, ~, sigma] = crossSofi(DataCombined);
[sofiMap, ~, ~, ~, ~, ~, ~] = autoSofi(DataCombined);

% Old fcs implimentation
%{
%% Calculate intensity for images by different methods     
AC_G2 = zeros(1, numel(ACXC_all));
for i = 1:numel(ACXC_all)
    % AC only - first point of different orders
    AC_G2(i) = ACXC_all(1, i).Order2(1, 2);
end

%% Reshape matrices for images
% reshape easy ones first
AC_G2_im = [0, AC_G2, 0];% pad matrices with first points, so back to original size
AC_G2_im = reshape(AC_G2_im, ymax - ymin, xmax - xmin);% reshape

M = zeros(size(AC_G2_im));
M(1:end - 1, :) = AC_G2_im(2:end, :);
M(end, :) = circshift(AC_G2_im(1, :), numel(AC_G2_im(1, :)) - 1);
AC_G2_im = M;
sofiMap = AC_G2_im;
%}

%% Deconvolution

if doDecon
    [deconAC, deconXC, deconAvg] = decon(avgim, {sofiMap}, {crossSofiMap}, sigma);
    sofiMapDecon = deconAC{1};
    crossSofiMapDecon = deconXC{1};
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
D = (w .^ 2) ./ (4 * tauD); %in micro meters^2/s
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

% Old fcs Implimentation (Slower and harder to read/modify)
%{
% ROI pixels of image to analyze
xmn = 1; xmx = xmax - xmin;
ymn = 1; ymx = ymax - ymin;

%% load data 

%reshape ACXC_all to get ROI
ACXC_all2 = [ACXC_all, ACXC_all(1, 1), ACXC_all(1, 1)];
ACXC_all_reshape = reshape(ACXC_all2, size(AC_G2_im));
ACXC_all_ROI = ACXC_all_reshape(ymn:ymx, xmn:xmx);
AC_XC_all_save = reshape(ACXC_all_ROI, 1, size(ACXC_all_ROI, 1) * size(ACXC_all_ROI, 2)); % Same as ACXC_all2 since RIO already done
AC_all(1, 1).curves = AC_XC_all_save;

for i = 1:numel(AC_all(1, 1).curves)
        % add raw AC curves
    ACadd(1, :) = AC_all(1, 1).curves(1, i).Order2; 
    AC_avg(1, i).curves = mean(ACadd, 1); % Doesn't Average since ACdd in second dimension, AC_avg same as AC_all
end

%% log bin the averaged data
for i = 1:numel(AC_all(1, 1).curves)

    AC_aver = (AC_avg(1, i).curves);
    max_lag = numel(AC_avg(1, i).curves);
    lags = 1:max_lag;
    ddwell = 1;

    [new_lags, new_AC] = logbindata(lags, AC_aver', ddwell, max_lag);

    AC_logbin(i, :) = new_AC;
    AC_loglag(i, :) = new_lags;

end


rowdim = size(ACXC_all_ROI, 1);
coldim = size(ACXC_all_ROI, 2);


%% Curve fitting

% initialize variable to keep track of GPU computation time
fit_time = 0;

% number of parameters
number_parameters = [3; 5; 4; 1; 2; 2]; number_parameters = number_parameters(type);

% estimator id
estimator_id = EstimatorID.LSE;

% model ID
model_id = [ModelID.BROWNIAN_1COMP; ModelID.BROWNIAN_2COMP; ModelID.ANOMALOUS;...
    ModelID.BROWNIAN_1COMP_NORM; ModelID.BROWNIAN_1COMP_TAUA; ...
    ModelID.ANOMALOUS_2PARAM_TAUA]; model_id = model_id(type);

% tolerance
tolerance = 1e-3;

% maximum number of iterations per pixel
max_n_iterations = 10000;

% preallocate variables 
tauD = zeros(1, xmx * ymx); tauD2 = tauD; D = tauD; D2 = tauD; alpha = tauD; chi = tauD;

%% Perform curve fitting
for i = 1:size(AC_logbin, 1)

    % display progress at 25%, 50%, and 75% complete
    if i == ceil(size(AC_logbin, 1) / 4)
        fprintf('Curve-fitting 25%% complete \n');
    end
    if i == ceil(size(AC_logbin, 1) / 2)
        fprintf('Curve-fitting 50%% complete \n');
    end
    if i == ceil(3 / 4 * size(AC_logbin, 1))
        fprintf('Curve-fitting 75%% complete \n');
    end

    % extract auto correlation curve 
    ACcurve = AC_logbin(i, :);
    timelag = AC_loglag(i, :);     

    % convert to x and y variables
    x = (timelag .* dT); %convert x values to seconds   SHAWN 
    y = ACcurve;
    % remove first timelag point tau=lag
    ind = numel(x);
    x = x(2:ind);
    y = y(2:ind);
    y = y ./ max(y);
    
    % choose startpoint tau_D
    td_stp = (pixelsize ^ 2) / (D_stp * 4);
    td2_stp = (pixelsize ^ 2) / (D2_stp * 4);
    
    % declare start points based on diffusion type
    sp_struct = struct; % start point structure
    sp_struct.brownian = [A_stp, mean(y(round((3*numel(y)/4)):numel(y))), td_stp];
    sp_struct.brownian2comp = [max(y), max(y), mean(y(round((3*numel(y)/4)):numel(y))), 1/2*td_stp, 1/2*td_stp];
    sp_struct.anomalous = [max(y)*2, mean(y(round((3*numel(y)/4)):numel(y))), td_stp, alpha_stp];
    sp_struct.browniannorm = td_stp;
    sp_struct.browniantaua = [A_stp, td_stp];
    sp_struct.anomalous2paramtaua = [td_stp, alpha_stp];
    sp_cell = struct2cell(sp_struct);
    start_points = sp_cell{type};

    % initial parameters
    initial_parameters = repmat(single(single(start_points)'), [1, number_fits]);

    % convert raw data to single precision and format for GPU fitting
    data = single(y); data = repmat(data(:), [1, number_fits]);

    % user info (independent variables)
    user_info = single(x);

    % weights
    weights = data ./ data;
    weights(data < 0) = 0;
    
    % Run Gpufit
    [parameters, states, chi_squares, n_iterations, gputime] = gpufit(data, weights, ...
        model_id, initial_parameters, tolerance, max_n_iterations, [], estimator_id, user_info);

    % converged parameters
    converged = states == 0; 
    converged_parameters = parameters(:, converged);

    % if parameters do not converge, use last iteration
    if isempty(converged_parameters) == 1
        model_coefs = parameters(1:number_parameters);
    else
        model_coefs = converged_parameters(1:number_parameters);
    end

    % construct fit result curve
    n_struct = struct('brownian', 3, 'brownian2comp', [4; 5], 'anomalous', 3, 'brownian_norm', 1, ...
                        'brownian_taua', 2, 'anomalous_2param_taua', 2);
    n_cell = struct2cell(n_struct);
    n = n_cell{type};
    len_x = numel(x);
    if type == 1
        model_fit(1:len_x) = model_coefs(1) .* (1./(1+(x(1:len_x)./model_coefs(3)))) + model_coefs(2);
    elseif type == 2
        model_fit(1:len_x) = model_coefs(1) .* (1./(1+(x(1:len_x)./model_coefs(4)))) + model_coefs(2).*(1./(1+(x(1:len_x)./model_coefs(5)))) + model_coefs(3);
    elseif type == 3
        model_fit(1:len_x) = model_coefs(1) .* (1./(1+(x(1:len_x)./model_coefs(3)).^model_coefs(4))) + model_coefs(2); 
    elseif type == 4
        model_fit(1:len_x) = 1./(1+(x(1:len_x)./model_coefs(1)));
    elseif type == 5
        model_fit(1:len_x) = model_coefs(1).* (1./(1+(x(1:len_x)./model_coefs(2)))); 
    elseif type == 6
        model_fit(1:len_x) = (1./(1+(x(1:len_x)./model_coefs(1)).^model_coefs(2)));
    end

    % R-square
    residuals = y - model_fit;
    a = (y - model_fit) .^ 2 ./ model_fit;
    a(isinf(a)) = 0;
    rsquare = 1 - sum(residuals .^ 2) / sum(y .^ 2);

    % Chi-Squared
    chi(i) = chi_squares(1);

    % fit result structure
    fitresult(1, i) = struct('rawdata', [x', y'], 'rsquare', rsquare, 'model_fit', model_fit);

    % characteristic time
    tauD(i) = model_coefs(n(1)); % in in seconds
    
    % diffusion coefficient
    w = pixelsize * sigma * 2.355; % Use PSFsample instead if you know the full width half mast;
    D(i) = (w .^ 2) / (4 * tauD(i)); %in micro meters^2/s
    
    % second diffusion coefficient if using 2-component model
    if type == 2
        tauD2(i) = model_coefs(n(2)) * dT; % in in seconds
        D2(i) = (pixelsize .^ 2) / (4 * tauD(i));
    end

    % alpha map if using anomalous model
    if type == 3
        alpha(i) = model_coefs(4);
    end
    
    % alpha map if using anomalous model with 2 parameters
    if type == 6
        alpha(i) = model_coefs(2);
    end

    % compute total Gpufit time
    fit_time = fit_time + gputime;
end

%% Post fit data manipulation
% reshape fit result 
fitresult2 = reshape(fitresult, rowdim, coldim);

%Diffusion coefficient map
Dmap = reshape(D, rowdim, coldim);
chiMap = reshape(chi, rowdim, coldim);

% create tauD map
tauDmap = reshape(tauD, rowdim, coldim);

% remove poor fits
D_corrected = zeros(1, numel(D));
for i = 1:numel(D)
    if fitresult(1, i).rsquare < 0.
        D_corrected(i) = 0;
    else
        D_corrected(i) = abs(D(i));
    end
end
Dmap_corrected = reshape(D_corrected, rowdim, coldim);

% second diffusion coefficeint map if 2-component brownian model
if type == 2
    D2map = reshape(D2, rowdim, coldim);
    tauD2map = reshape(tauD2, rowdim, coldim);
    D2_corrected = zeros(1, numel(D2));
    for i = 1:numel(D2)
        if fitresult(1, i).rsquare < 0.
            D2_corrected(i) = 0;
        else
            D2_corrected(i) = abs(D2(i));
        end
    end
    D2map_corrected = reshape(D2_corrected, rowdim, coldim);
end

% alpha map (anomalous diffusion)
if type == 3   ||  type == 6
    alpha_corrected = zeros(1, numel(alpha));
    % remove bad alphas
    for i = 1:numel(alpha)
        if fitresult(1, i).rsquare < 0.
            alpha_corrected(i) = 0;
        elseif alpha(i) < 0
            alpha_corrected(i) = 0;
        elseif alpha(i) < alpha_min
            alpha_corrected(i) = 0;
        elseif alpha(i) > alpha_max
            alpha_corrected(i) = 0;
        else
            alpha_corrected(i) = abs(alpha(i));
        end
    end
    alphamap = reshape(alpha_corrected, rowdim, coldim);
end

% make map of R^2 values
R2 = zeros(1, numel(fitresult));
for i = 1:numel(fitresult)
    R2(i) = fitresult(1, i).rsquare;
end
R2map = reshape(R2, rowdim, coldim);
%}

% Display execution time of fcs step
time = toc(timeFcs);
timeOut = ['FCS Complete, Execution Time: ', num2str(floor(time / 60)), ' Minutes, ', num2str(mod(time, 60)), ' Seconds'];
disp(timeOut);


%% %%%%%%%%%%%%%%%%%%%% STEP 3: CombineTempSpat (fcsSOFI) %%%%%%%%%%%%%%%%%%%%%%%%%

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

% Used to keep track of figures to save
figureNumber = 1;
%{
% Plot Diffusion Data
figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
plot(1:numel(DhighSOFIvaluesR2), DhighSOFIvaluesR2);
title('Diffusion Vector') 

% Plot Probability Distribution
figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
histogram(DhighSOFIvaluesR2);
title('Probability distribution')
%}

% Plot Cumulative Distribution
figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
plot(flipud(DWellFinal), IndexFinal, 'k.');
title('Cumulative distribution')
xlabel('Diffusion Coefficient (\mum^2s^{-1})')
ylabel('Probability')


%% Filter Out Really Bad Fits in FCS Plot

% Filter out poor fits if you want to here with an R2cutoff
R2cutoff = 0.5; %set R^2 cutoff (0-1); == 0, no filtering
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
sizedDmapAlpha = imresize(DmapAlpha, size(Dmap_corrected).*binSize, 'nearest');

% Resize the binned data to the cross sofi dimentions 
crossDmap = imresize(Dmap_corrected, size(Dmap_corrected).*2.*binSize, 'nearest');
crossDmapAlpha = imresize(DmapAlpha, size(Dmap_corrected).*2.*binSize, 'nearest');
crossDmap = crossDmap(3:end-3, 3:end-3);
crossDmapAlpha = crossDmapAlpha(3:end-3, 3:end-3);

%% Finish the timer for image combination

% display execution time of fcsSOFI combination
time = toc(timeCombine);
timeOut = ['Image Fusion Complete, Execution Time: ', num2str(floor(time / 60)), ' Minutes, ', num2str(mod(time, 60)), ' Seconds'];
disp(timeOut);

%% Figures

% The color map used for all the figures
if defualtColors == 1
    customColorMap = parula;
else
    customColorMap = [linspace(botC(1), topC(1)); linspace(botC(2), topC(2)); linspace(botC(3), topC(3))].';
end

if plotfigures == 1 
    
    % blinkAConly Subplots (SOFI)
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;

    % Average Image
    subplot(2, 2, 1); imagesc(avgim); colormap(gray); axis image;
    title('Average image'); set(gca, 'xtick', [], 'ytick', [])

    % Autocorrelation
    subplot(2, 2, 2); imagesc(sofiMapSat); axis image
    title('AC G^2(0+\tau)'); set(gca, 'xtick', [], 'ytick', [])

    % Deconvolved Image
    subplot(2, 2, 3); imagesc(sofiMapDeconSat); axis image; 
    title('Deconvolved');set(gca, 'xtick', [], 'ytick', [])

    % Line Sections of single point row
    subplot(2, 2, 4); hold on
    plot(avgim(row_index, :) ./ max(avgim(row_index, :)), '-b')
    plot(sofiMapSat(row_index, :) ./ max(sofiMapSat(row_index, :)), '-r')
    plot(sofiMapDeconSat(row_index, :) ./ max(sofiMapDeconSat(row_index, :)), '-k')
    axis square; title('Line sections'); ylim([0 1]); xlim([0 size(sofiMapSat(row_index, :), 2)])


    % BinFitData subplots (fcs) 
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    
    % Tua Diffusion Map
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


    % R-square Map
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    imagesc(R2map); axis image; title('R^2 map');
    colorbar;

    % fcs figure creation
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    DFigure = imagesc(sizedDmap); axis image; title('FCS: log(D)')
    DFigure.AlphaData = sizedDmapAlpha; colormap(customColorMap);
    c = colorbar; c.Label.String = 'log(D/(\mum^2s^{-1}))';
    set(gca, 'Color', [0, 0, 0]) % Set background color to black
    set(gca, 'FontSize', 14); set(gca, 'xtick', [], 'ytick', []) % Removes axis tick marks
    
    % SOFI super resolution image
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    imagesc(sofiMapSat); axis image; title('SOFI super-resolution')
    set(gca, 'xtick', [], 'ytick', []); colormap(gray)

    % Decon SOFI super resolution image
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    imagesc(sofiMapDeconSat); axis image; title('SOFI super-resolution with Decon')
    set(gca, 'xtick', [], 'ytick', []); colormap(gray)
    
    % Cross SOFI super resolution image
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    imagesc(crossSofiMapSat); axis image; title('Cross SOFI super-resolution')
    set(gca, 'xtick', [], 'ytick', []); colormap(gray)
    
    % Cross Decon SOFI super resolution image 
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    imagesc(crossSofiMapDeconSat); axis image; title('Cross SOFI super-resolution with Decon')
    set(gca, 'xtick', [], 'ytick', []); colormap(gray)
    
    % fcsSOFI figure creation 
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    fcsSofiPlot = imagesc(sizedDmap); axis image; title('Combined fcsSOFI image')
    fcsSofiPlot.AlphaData = sofiMapSat .* sizedDmapAlpha; % Uses the SOFI data as a transparency map
    colormap(customColorMap); c = colorbar; c.Label.String = 'D \mum^2s^{-1}';
    set(gca, 'Color', [0, 0, 0]) % Set background color to black
    set(gca, 'FontSize', 14); set(gca,'xtick',[],'ytick',[])

    % Decon fcsSOFI figure creation
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    fcsSofiPlot = imagesc(sizedDmap); axis image; title('Combined fcsSOFI image with Decon')
    fcsSofiPlot.AlphaData = sofiMapDeconSat .* sizedDmapAlpha; % Uses the SOFI data as a transparency map
    colormap(customColorMap); c = colorbar; c.Label.String = 'D \mum^2s^{-1}';
    set(gca, 'Color', [0, 0, 0]) % Set background color to black
    set(gca, 'FontSize', 14); set(gca,'xtick',[],'ytick',[])
    
    % Cross fcsSOFI figure creation 
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    fcsSofiPlot = imagesc(crossDmap); axis image; title('Combined Cross fcsSOFI image')
    fcsSofiPlot.AlphaData = crossSofiMapSat .* crossDmapAlpha; % Uses the SOFI data as a transparency map
    colormap(customColorMap); c = colorbar; c.Label.String = 'D \mum^2s^{-1}';
    set(gca, 'Color', [0, 0, 0]) % Set background color to black
    set(gca, 'FontSize', 14); set(gca,'xtick',[],'ytick',[])
    
    % Cross Decon fcsSOFI figure creation 
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    fcsSofiPlot = imagesc(crossDmap); axis image; title('Combined Cross fcsSOFI image with Decon')
    fcsSofiPlot.AlphaData = crossSofiMapDeconSat .* crossDmapAlpha; % Uses the SOFI data as a transparency map
    colormap(customColorMap); c = colorbar; c.Label.String = 'D \mum^2s^{-1}';
    set(gca, 'Color', [0, 0, 0]) % Set background color to black
    set(gca, 'FontSize', 14); set(gca,'xtick',[],'ytick',[])
end

%% Single Pixel Results
    name = ["Brownian","2-Component Brownian","Anomalous","Brownian Norm",...
        "Brownian 1-Component Normalized", "Brownian 1-Component with Amplitude",...
        "Anomalous 2 Parameters Tau and Alpha"];name = name(type);

    % optional single pixel curve fit result figure
    if examplecf == 1
        printRow = row_index;
        printCol = column_index;
        row_index = ceil(row_index / binSize); %row index
        column_index = ceil(column_index/ binSize); %column index
        linear_index = sub2ind(fcsSz, row_index, column_index);
        singleParams = parameters(:, linear_index);

        figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
        x2 = fitTimePoints(:, linear_index); 
        y2 = fitData(:, linear_index);
        plot(x2, y2, 'or', 'LineWidth', 2)
        hold on
        plot(x2, model_fit(:, linear_index), '--k', 'LineWidth', 2)
        hold on
        set(gca,'xscale','log')
        xlabel('\tau')
        ylabel('G(\tau)')
        legend('Raw Data', 'Fit Result');
        title(strcat(name, ' Diffusion Curve Fit with Gpufit')); 
        
        % error bars
        number_parameters = [3; 5; 4; 1; 2; 2];
        [rsq, chisq, J, MSE, ci] = gofStats(type,...%type
            singleParams(1:number_parameters(type)),... %parameter values
            model_fit(:, linear_index),...    %fit curve
            x2,... %x data
            y2);   %y data
        gof_gpu = [rsq chisq];
        ci = ci';

        ebars = zeros(1, size(ci, 2));
        ebars(1:size(ci, 2)) = abs(ci(2, 1:size(ci, 2)) - ci(1, 1:size(ci, 2))) / 2;
        
        modeleqn = ["G(tau) = a * 1/(1 + tau/tauD) + b",...
            "G(tau) = a1 * 1/(1 + tau/tauD1) + a2 * 1/(1 + tau/tauD2) + b",...
            "G(tau) = a * 1/(1 + (tau/tauD)^alpha) + b", ...
            "G(tau) = 1/(1 + tau/tauD)", ...
            "G(tau) = a * 1/(1 + tau/tauD)",...
            "G(tau) = 1/(1 + (tau/tauD)^alpha)"]; modeleqn = modeleqn(type);

        fprintf('\n'); fprintf(fname); fprintf('\nPixel (%i,%i)\n', printRow, printCol);
        fprintf(strcat(name, ' Fit Model:\n', modeleqn, '\n\n'));
        fprintf('Fit Result Parameters\n');
        
        % print error bars
        if type == 1
          fprintf('a =    %6.2e ± %6.2e\n', singleParams(1), ebars(1));
          fprintf('b =    %6.2e ± %6.2e\n', singleParams(2), ebars(2));
          fprintf('tauD =     %6.2e ± %6.2e\n', singleParams(3), ebars(3));
          fprintf('\n')
          fprintf('D =         %6.3e\n', Dmap_corrected(row_index, column_index))
          fprintf('R-square:  %6.4f\n', R2map(row_index, column_index));
        
        elseif type == 2
          fprintf('a1 =    %6.2e ± %6.2e\n', singleParams(1), ebars(1));
          fprintf('a2 =    %6.2e ± %6.2e\n', singleParams(2), ebars(2));
          fprintf('b =     %6.2e ± %6.2e\n', singleParams(3), ebars(3));
          fprintf('tauD1 = %6.4f ± %6.4e\n', singleParams(4), ebars(4));
          fprintf('tauD2 = %6.4f ± %6.4e\n', singleParams(5), ebars(5));  
          fprintf('\n')
          fprintf('D1:         %6.3e\n', Dmap_corrected(row_index, column_index))
          fprintf('D2:         %6.3e\n', D2map_corrected(row_index, column_index))
          fprintf('R-square:  %6.4f\n', R2map(row_index, column_index));   

        elseif type == 3
          fprintf('a =    %6.2e ± %6.2e\n', singleParams(1), ebars(1));
          fprintf('b =    %6.2e ± %6.2e\n', singleParams(2), ebars(2));
          fprintf('tauD =     %6.2e ± %6.2e\n', singleParams(3), ebars(3));
          fprintf('alpha = %6.4f ± %6.4f\n', singleParams(4), ebars(4));
          fprintf('\n')
          fprintf('D =         %6.3e\n', Dmap_corrected(row_index, column_index))
          fprintf('alpha:     %6.4f\n\n', alphamap(row_index, column_index))
          fprintf('R-square:  %6.4f\n', R2map(row_index, column_index));
          
        elseif type == 4
          fprintf('tauD =     %6.2e\n ± %6.2e\n', tauDmap(row_index, column_index), ebars(1));
          fprintf('\n')
          fprintf('D =         %6.3e\n', Dmap_corrected(row_index, column_index))
          fprintf('R-square:  %6.4f\n', R2map(row_index, column_index));    
        
        elseif type == 5
          fprintf('a =    %6.2e ± %6.2e\n', singleParams(1), ebars(1));
          fprintf('tauD =     %6.2e ± %6.2e\n', singleParams(3), ebars(3));
          fprintf('\n')
          fprintf('D =         %6.3e\n', Dmap_corrected(row_index, column_index))
          fprintf('R-square:  %6.4f\n', R2map(row_index, column_index));
          
       elseif type == 6
          fprintf('tauD =     %6.2e ± %6.2e\n', singleParams(1), ebars(1));
          fprintf('alpha = %6.4f ± %6.4f\n', singleParams(2), ebars(2));
          fprintf('\n')
          fprintf('D =         %6.3e\n', Dmap_corrected(row_index, column_index))
          fprintf('alpha:     %6.4f\n\n', alphamap(row_index, column_index))
          fprintf('R-square:  %6.4f\n', R2map(row_index, column_index));
        end
    end

%% save data
if savethedata == 1 
    
    % The Variables to save
    date = datestr(now,'mm-dd-yyyy_HH-MM');

    folderNameStart = erase(fname, '.mat');
    folderName = strcat(folderNameStart, '_analyzed_', date);

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
        'Dmap_corrected', 'DmapAlpha', 'sizedDmap', 'sizedDmapAlpha', ...
        'crossDmap', 'crossDmapAlpha', 'R2map', 'chiMap', ...
        'DhighSOFIvaluesR2', 'TimeArray', 'sigma', 'sigmaBin', ...
        'satMax', 'crossSatMax', 'satMin', 'dT', 'customColorMap', ...
        'D2map_corrected', 'alphamap', '-v7.3');

    % Moves the files into the folder created
    movefile(figureFileName, folderName);
    movefile(dataFileName, folderName);

    if useTiffFile
        movefile(fname, folderName);
    end

    % No Longer Saving BC
    %{
    if ~useBCData
        backFileName = strcat(folderName, '_BC.mat');
        save(backFileName, 'thrData', '-v7.3');
        movefile(backFileName, folderName);
    end
    %}

end   

%% total computation time
time = toc(timeStart);
timeOut = ['Total Execution Time: ', num2str(floor(time / 60)), ' Minutes, ', num2str(mod(time, 60)), ' Seconds'];
disp(timeOut);

fit_timeOut = ['Total Time in GPU Fit: ', num2str(floor(fit_time / 60)), ' Minutes, ', num2str(mod(fit_time, 60)), ' Seconds'];
disp(fit_timeOut);