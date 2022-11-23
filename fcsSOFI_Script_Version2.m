clear; close all hidden; clc;
%% User Input
startloc = 'Z:\BenjaminWellnitz\MATLAB Practice\20220915';

% Diffusion coefficient parameters
pixelsize = 109; % In nm (IX83); needed to accurately calculate D
PSFsample = 5; % In pixel; based off of PSF from moving samples
dT = 0.005; % In s; needed to accurately calculate D

% Set PSF for deconvolution of sofi image
sigma = (PSFsample / 2.355) / (2 ^ 0.5); % Standart deviation of PSF in pixels

% Number of files used together and length of each file
numberFiles = 1;
framesLength = 10000;

% Region of interest in pixels (of all files added together)
ymin = 1; 
ymax = 100;
xmin = 1;
xmax = 100;
tmin = 1; % Start frame
tmax = 10000; % End frame

% Choose type of diffusion (1 = Brownian, 2 = 2-Comp Brownian, 3 = Anomalous, ...
        ... 4 = Brownian 1 Comp with tau, 5 = 1-comp Brownian with tau and A, ...
        ... 6 = Anomalous with Tau and alpha)
type = 4;

% Choose alpha start point (Anomalous diffusion model only)
A_stp = 1;
alpha_stp = .9;
D_stp = 1e5;
D2_stp = 1e6;

% Alpha threshold
alpha_max = 1.2; % Maximum value of alpha allowed to appear on alpha map
alpha_min = 0;

% Set the limits which the diffsion data will be trimmed to
diffusionMin = 0; 
diffusionMax = Inf;

% Number of fit iterations per pixel
number_fits = 1000;

% Plot figures? (1 = yes)
plotfigures = 1;

% Store execution times in external text file? (1 = yes);
store_execution_times = 0;

% Save data files? (1 = yes)
savethedata = 1; 

% Optional example single pixel curve fit plot (1 = yes)
examplecf = 1;
column_index = 10; % X coordinate
row_index = 10; % Y coordinate

% Deconvolution on SOFI Image (1 = yes, 0 = no)
doDecon = 1;

% SOFI scaling
satmin = 0;
satmax = 1;

% Whether you are using a tiff or mat file (1 = yes, 0 = no)
doFileConversionAndBackSubtraction = 1;

% Top and bottom colors for the color bar / color scale
red = [1 0 0]; green = [0 0.5 0]; blue = [0 0 1]; lime = [0 1 0]; cyan = [0 1 1]; yellow = [1 1 0];
magenta = [1 0 1]; maroon = [0.5 0 0]; olive = [0.5 0.5 0]; purple = [0.5 0 0.5]; teal = [0 0.5 0.5]; navy = [0 0 0.5];
topC = blue;
botC = red;

% Use defualt color scheme (1 = yes)
defualtColors = 1;

% END USER INPUT
fprintf('Running...\n');

%% Convert Tif to Mat
% Alows for ultiple files to be added together
if doFileConversionAndBackSubtraction
    
    fileNames = cell(1, numberFiles); paths = cell(1, numberFiles);
    % Select all the files
    for i = 1:numberFiles
        [fileName, path] = uigetfile(startloc, '*.tiff');
        fileNames(1, i) = {fileName};
        paths(1, i) = {path};
    end
    % Read all the files
    Data = TiffReadRM(fileNames{1, 1}, paths{1, 1}, 1, framesLength);
    fprintf('Tiff file 1 loaded \n');
    for i = 2:numberFiles
        Data = cat(3, Data, TiffReadRM(fileNames{1, i}, paths{1, i}, 1, framesLength));
        fprintf('Tiff file %g loaded \n', i);
    end
    
    filenm = extractBefore(fileNames{1, 1}, ".tif");

    %% Background Subtraction
    thrData = Data; %will be the background subtracted dataset
    for i = 1:size(Data, 3) %i is the frame number
        Bkg = LRG_SuperRes_LocalThrMap(Data(:, :, i), true);%local background calcualted with LRG code
        thrData(:, :, i) = double(Data(:, :, i)) - double(Bkg); %background subtraction step
    end

    %Background corrected file name
    fname = strcat(filenm, '_Combined_BC.mat');
    save(fname, 'thrData', '-v7.3')
    DataCombined = thrData;

else % If data was already converted
    [fileName, path] = uigetfile(startloc, '*');
    fname = fileName;
    addpath(path);
    load(fileName);
    DataCombined = thrData;
end

%% Paths
addpath(strcat(pwd, '\gpufit\Debug\matlab'))
addpath(strcat(pwd, '\fcsSOFI_external_functions'))

%% %%%%%%%%%%%%%%%%%%%% STEP 1: blink_AConly (SOFI) %%%%%%%%%%%%%%%%%%%%%%%%%
% start global timer
before = clock;

% start timer for SOFI step
blink_before = clock;

%% start code/load and setup data 

% Load data
fprintf(strcat(fname, ' loaded\n'));

% Set ROI
DataCombined = DataCombined(ymin:ymax, xmin:xmax, tmin:tmax);

% Produce average image
avgimage = sum(DataCombined(:, :, :), 3) ./ size(DataCombined, 3);

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
[ACXC_all] = CalcCorrCut(innerpts, DataVector); %calculate 

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

%% Deconvolution
avgim = avgimage;
im = AC_G2_im;

% Define the PSF 
gauss1 = customgauss([100 100], sigma, sigma, 0, 0, 1, [5 5]); % Create a 2D PSF
PSF = gauss1(45:65, 45:65); % Only use the center where the PSF is located at

if doDecon
    filtim = deconvlucy(im, PSF); % Based on Geissbuehler bSOFI paper
else
    filtim = im;
end

% display execution time of SOFI step
blink_after = clock;
fprintf('SOFI complete, execution time: %6.2f seconds\n', etime(blink_after, blink_before));

%% %%%%%%%%%%%%%%%%%%%% STEP 2: BinFitData (fcs) %%%%%%%%%%%%%%%%%%%%%%%%%

% start timer for fcs step
Bin_before = clock;

% ROI pixels of image to analyze
xmn = 1; xmx = xmax - xmin;
ymn = 1; ymx = ymax - ymin;

%% load data 

%reshape ACXC_all to get ROI
ACXC_all2 = [ACXC_all, ACXC_all(1, 1), ACXC_all(1, 1)];
ACXC_all_reshape = reshape(ACXC_all2, size(AC_G2_im));
ACXC_all_ROI = ACXC_all_reshape(ymn:ymx, xmn:xmx);
AC_XC_all_save = reshape(ACXC_all_ROI, 1, size(ACXC_all_ROI, 1) * size(ACXC_all_ROI, 2));
AC_all(1, 1).curves = AC_XC_all_save;

for i = 1:numel(AC_all(1, 1).curves)
        % add raw AC curves
    ACadd(1, :) = AC_all(1, 1).curves(1, i).Order2;
    AC_avg(1, i).curves = mean(ACadd, 1);   
end

%% log bin the averaged data
for i = 1:numel(AC_all(1, 1).curves)

    AC_aver = (AC_avg(1, i).curves);
    max_lag = numel(AC_avg(1, i).curves);
    lags = 1:max_lag;
    ddwell = 1;

    [new_lags, new_AC] = logbindata(lags, AC_aver, ddwell, max_lag);

    AC_logbin(i, :) = new_AC;
    AC_loglag(i, :) = new_lags;

end

rowdim = size(ACXC_all_ROI, 1);
coldim = size(ACXC_all_ROI, 2);


%% Set up curve fitting

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
tauD = zeros(1, xmx * ymx); tauD2 = tauD; D = tauD; D2 = tauD; alpha = tauD;

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

    % fit result structure
    fitresult(1, i) = struct('rawdata', [x', y'], 'rsquare', rsquare, 'model_fit', model_fit);

    % characteristic time
    tauD(i) = model_coefs(n(1)); % in in seconds
    
    % diffusion coefficient
    w = pixelsize * PSFsample;
    D(i) = (w .^ 2) / (4 * tauD(i)); %in nm^2/s
    
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

% display execution time of fcs step
Bin_after = clock;
fprintf('FCS complete, execution time: %6.2f seconds\n', etime(Bin_after, Bin_before));

%% %%%%%%%%%%%%%%%%%%%% STEP 3: CombineTempSpat (fcsSOFI) %%%%%%%%%%%%%%%%%%%%%%%%%

% start timer for fcsSOFI combination
Combine_before = clock;
    
%% Diffusion map

% Create log base Dmap
Dmap2log = log10(Dmap_corrected); 

for i = 1:size(Dmap2log, 1)
    for j = 1:size(Dmap2log, 2)
        if Dmap2log(i, j) == -Inf
            Dmap2log(i,j ) = 0;
        end
    end
end

% Filter out poor fits if you want to here with an R2cutoff
R2cutoff = 0.5; %set R^2 cutoff (0-1); == 0, no filtering
Dmap2logAlpha = ones(size(Dmap2log, 1), size(Dmap2log, 2));
for i = 1:size(Dmap2log, 1)
    for j = 1:size(Dmap2log, 2)
        if R2map(i, j) < R2cutoff
            Dmap2log(i, j) = NaN;
            Dmap2logAlpha(i, j) = 0;
        end
    end
end

%R2 cutoff from D histogram; beads=0.95, 76kDa=0.8, 2000kDa=0.9, BSA=0.88
R2cutoff = 0.95;
DhighSOFIvaluesR2 = [];
for i = 1:size(Dmap2log, 1)
    for j = 1:size(Dmap2log, 2)
        if R2map(i, j) > R2cutoff
            DhighSOFIvaluesR2 = [DhighSOFIvaluesR2, Dmap_corrected(i, j)];
        end
    end
end

% Used to keep track of figures to save
figureNumber = 1; figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
hist(DhighSOFIvaluesR2)

% Trims the RGB Diffusion map data to realistic values
trimDmap2log = Dmap2log;

for i = 1:size(Dmap2log, 1)
    for j = 1:size(Dmap2log, 2)
        if Dmap2log(i,j ) < diffusionMin
            trimDmap2log(i, j) = diffusionMin;
        elseif Dmap2log(i, j) > diffusionMax
            trimDmap2log(i, j) = diffusionMax;
        end
    end
end


%% Set the limits of the SOFI image

sofiMap = filtim;
sofiMap = sofiMap ./ (max(max(sofiMap))); %normalize
im = im ./ (max(max(im)));
for i = 1:size(sofiMap, 1)
    for j = 1:size(sofiMap, 2)
        % Deconvolved sofi
        if sofiMap(i, j) < satmin
            sofiMap(i, j) = satmin;
        elseif sofiMap(i, j) > satmax
            sofiMap(i, j) = satmax;
        end
        % Non-Deconvolved sofi
        if im(i, j) < satmin
            im(i, j) = satmin;
        elseif im(i, j) > satmax
            im(i, j) = satmax;
        end 
    end
end
sofiMap = sofiMap ./ (max(max(sofiMap))); %normalize again
im = im ./ (max(max(im)));

%% Finish the timer for image combination

% display execution time of fcsSOFI combination
Combine_after = clock;
fprintf('Image fusion complete, execution time: %6.2f seconds\n', etime(Combine_after, Combine_before));

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
    subplot(2, 2, 2); imagesc(im); axis image
    title('AC G^2(0+\tau)'); set(gca, 'xtick', [], 'ytick', [])

    % Deconvolved Image
    subplot(2, 2, 3); imagesc(sofiMap); axis image; 
    title('Deconvolved');set(gca, 'xtick', [], 'ytick', [])

    % Line Sections of row 25
    subplot(2, 2, 4); hold on
    plot(avgim(25, :) ./ max(avgim(25, :)), '-b')
    plot(im(25, :) ./ max(im(25, :)), '-r')
    plot(sofiMap(25, :) ./ max(sofiMap(25, :)), '-k')
    axis square; title('Line sections'); ylim([0 1]); xlim([0 size(im(25, :), 2)])


    % BinFitData subplots (fcs) 
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    
    % Tua Diffusion Map
    subplot(2, 2, 1); imagesc(tauDmap); colorbar; axis image; title('\tau_D map')
    
    % Diffusion Map with nothing removed
    subplot(2, 2, 2); imagesc(Dmap); colormap(customColorMap); 
    colorbar; axis image; title('D map, nothing removed')

    % Diffusion Map with poor fits removed
    subplot(2, 2, 3); imagesc(Dmap_corrected); colormap(customColorMap);
    colorbar; axis image; title('Fits of R^2 < 0.5 removed')

    % Diffusion Map on a log base 10 scale
    subplot(2, 2, 4); imagesc(Dmap2log); colormap(customColorMap);
    colorbar; axis image; title('log scale c axis')

    % Second set of BinFitData subplots if using 2-comp model
    if type == 2
        figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
        
        % Tau Diffusion Map 2 Image
        subplot(2, 2, 1); imagesc(tauD2map); axis image; title('D2: \tau_D map')
        
        % Diffusion Map 2 With Nothing Removed
        subplot(2, 2, 2); imagesc(D2map); colormap(customColorMap);
        axis image; title('D2: D map, nothing removed')
        
        % Diffution Map 2 with poor fits removed
        subplot(2, 2, 3); imagesc(D2map_corrected); colormap(customColorMap);
        axis image; title('D2: Fits of R^2 < 0.5 removed')
        
        % Diffusion Map 2 on a log base 10 scale
        D2map2log = log10(D2map_corrected);
        subplot(2, 2, 4); imagesc(D2map2log)
        colormap(customColorMap); axis image; title('D2: log scale c axis')
    end
    

    % Alpha Map if using anomalous model
    if type == 3 || type == 6
        sofiBinarized = imbinarize(sofiMap, 0.05);
        alphamap(sofiBinarized == 0) = 0;
       
        figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
        imagesc(alphamap); axis image; title('\alpha map')
        c = colorbar; c.Label.String = '\alpha'; c.Label.FontSize = 20;
        set(gca, 'xtick', [], 'ytick', [])
    end


    % R-square Map
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    imagesc(R2map); axis image; title('R^2 map'); colorbar
    

    % fcs figure creation
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    DFigure = imagesc(trimDmap2log); axis image; title('FCS: log(D)')
    DFigure.AlphaData = Dmap2logAlpha; colormap(customColorMap);
    c = colorbar; c.Label.String = 'log(D) (nm^2/s)';
    patch([0 xmax-xmin+1 xmax-xmin+1 0], [0 0 ymax-ymin+1 ymax-ymin+1], 'k'); % Patches a black background in front
    set(gca, 'children', flipud(get(gca, 'children'))); % Moves Black Background to back
    set(gca, 'FontSize', 14); set(gca, 'xtick', [], 'ytick', []) % Removes axis tick marks
    
    
    % SOFI super resolution image
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    imagesc(sofiMap); axis image; title('SOFI super-resolution')
    set(gca, 'xtick', [], 'ytick', []); colormap(gray)

    
    % fcsSOFI figure creation 
    figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
    fcsSofiPlot = imagesc(trimDmap2log); axis image; title('Combined fcsSOFI image')
    fcsSofiPlot.AlphaData = sofiMap .* Dmap2logAlpha; % Uses the SOFI data as a transparency map
    colormap(customColorMap); c = colorbar; c.Label.String = 'log(D) (nm^2/s)';
    patch([0 xmax-xmin+1 xmax-xmin+1 0], [0 0 ymax-ymin+1 ymax-ymin+1], 'k'); % Patches a black background in front
    set(gca, 'children', flipud(get(gca, 'children'))); % Moves Black Background to back
    set(gca, 'FontSize', 14); set(gca,'xtick',[],'ytick',[])
end

%% Single Pixel Results
    name = ["Brownian","2-Component Brownian","Anomalous","Brownian Norm",...
        "Brownian 1-Component Normalized", "Brownian 1-Component with Amplitude",...
        "Anomalous 2 Parameters Tau and Alpha"];name = name(type);

    % optional single pixel curve fit result figure
    if examplecf == 1
        i = row_index; %row index
        j = column_index; %column index
        figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
        x2 = fitresult2(i, j).rawdata(:, 1); 
        y2 = fitresult2(i, j).rawdata(:, 2);
        plot(x2, y2, 'or', 'LineWidth', 2)
        hold on
        plot(x2, fitresult2(i, j).model_fit, '--k', 'LineWidth', 2)
        hold on
        set(gca,'xscale','log')
        xlabel('\tau')
        ylabel('G(\tau)')
        legend('Raw Data', 'Fit Result');
        title(strcat(name, ' Diffusion Curve Fit with Gpufit')); 
        
        % error bars
        number_parameters = [3; 5; 4; 1; 2; 2];
        [rsq, chisq, J, MSE, ci] = gofStats(type,...%type
            converged_parameters(1:number_parameters(type)),... %parameter values
            fitresult2(row_index, column_index).model_fit,...    %fit curve
            fitresult2(row_index, column_index).rawdata(:, 1)',... %x data
            fitresult2(row_index, column_index).rawdata(:, 2)');   %y data
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

        fprintf('\n'); fprintf(fname); fprintf('\nPixel (%i,%i)\n', row_index, column_index);
        fprintf(strcat(name, ' Fit Model:\n', modeleqn, '\n\n'));
        fprintf('Fit Result Parameters\n');
        
        % print error bars
        if type == 1
          fprintf('a =    %6.2e ± %6.2e\n', converged_parameters(1), ebars(1));
          fprintf('b =    %6.2e ± %6.2e\n', converged_parameters(2), ebars(2));
          fprintf('tauD =     %6.2e ± %6.2e\n', converged_parameters(3), ebars(3));
          fprintf('\n')
          fprintf('D =         %6.3e\n', Dmap_corrected(row_index, column_index))
          fprintf('log10(D):  %6.3f\n\n', Dmap2log(row_index, column_index))
          fprintf('R-square:  %6.4f\n', R2map(row_index, column_index));
        
        elseif type == 2
          fprintf('a1 =    %6.2e ± %6.2e\n', converged_parameters(1), ebars(1));
          fprintf('a2 =    %6.2e ± %6.2e\n', converged_parameters(2), ebars(2));
          fprintf('b =     %6.2e ± %6.2e\n', converged_parameters(3), ebars(3));
          fprintf('tauD1 = %6.4f ± %6.4e\n', converged_parameters(4), ebars(4));
          fprintf('tauD2 = %6.4f ± %6.4e\n', converged_parameters(5), ebars(5));  
          fprintf('\n')
          fprintf('D1:         %6.3e\n', Dmap_corrected(row_index, column_index))
          fprintf('D2:         %6.3e\n', D2map_corrected(row_index, column_index))
          fprintf('log10(D1):  %6.3f\n', Dmap2log(row_index, column_index))
          fprintf('log10(D2):  %6.3f\n\n', D2map2log(row_index, column_index))
          fprintf('R-square:  %6.4f\n', R2map(row_index, column_index));   

        elseif type == 3
          fprintf('a =    %6.2e ± %6.2e\n', converged_parameters(1), ebars(1));
          fprintf('b =    %6.2e ± %6.2e\n', converged_parameters(2), ebars(2));
          fprintf('tauD =     %6.2e ± %6.2e\n', converged_parameters(3), ebars(3));
          fprintf('alpha = %6.4f ± %6.4f\n', converged_parameters(4), ebars(4));
          fprintf('\n')
          fprintf('D =         %6.3e\n', Dmap_corrected(row_index, column_index))
          fprintf('log10(D):  %6.3f\n\n', Dmap2log(row_index, column_index))
          fprintf('alpha:     %6.4f\n\n', alphamap(row_index, column_index))
          fprintf('R-square:  %6.4f\n', R2map(row_index, column_index));
          
        elseif type == 4
          fprintf('tauD =     %6.2e\n ± %6.2e\n', tauDmap(row_index, column_index), ebars(1));
          fprintf('\n')
          fprintf('D =         %6.3e\n', Dmap_corrected(row_index, column_index))
          fprintf('log10(D):  %6.3f\n\n', Dmap2log(row_index, column_index))
          fprintf('R-square:  %6.4f\n', R2map(row_index, column_index));    
        
        elseif type == 5
          fprintf('a =    %6.2e ± %6.2e\n', converged_parameters(1), ebars(1));
          fprintf('tauD =     %6.2e ± %6.2e\n', converged_parameters(3), ebars(3));
          fprintf('\n')
          fprintf('D =         %6.3e\n', Dmap_corrected(row_index, column_index))
          fprintf('log10(D):  %6.3f\n\n', Dmap2log(row_index, column_index))
          fprintf('R-square:  %6.4f\n', R2map(row_index, column_index));
          
       elseif type == 6
          fprintf('tauD =     %6.2e ± %6.2e\n', converged_parameters(1), ebars(1));
          fprintf('alpha = %6.4f ± %6.4f\n', converged_parameters(2), ebars(2));
          fprintf('\n')
          fprintf('D =         %6.3e\n', Dmap_corrected(row_index, column_index))
          fprintf('log10(D):  %6.3f\n\n', Dmap2log(row_index, column_index))
          fprintf('alpha:     %6.4f\n\n', alphamap(row_index, column_index))
          fprintf('R-square:  %6.4f\n', R2map(row_index, column_index));
        end
    end

%% save data
if savethedata == 1 
    
    % The Variables to save
    date = datestr(now,'mm-dd-yyyy_HH-MM');
    fit_curves = fitresult2;
    fit_parameters = double(converged_parameters);
    sofiMapMatrix = sofiMap;
    DmapMatrix = Dmap;
    Dmap_correctedMatrix = Dmap_corrected;
    RsquareMapMatrix = R2map;
    trimDmap2logMatrix = trimDmap2log;
    D2_mapMatrix = NaN;
    D2_mapCorrectedMatrix = NaN;
    alphaMapMatrix = NaN;
    
    folderNameStart = erase(fname, '.mat');
    folderName = strcat(folderNameStart, '_analyzed_', date);
    
    switch type
        case 2
            D2_mapMatrix = D2map;
            D2_map_correctedMatrix = D2map_corrected;
        case 3
            alphaMapMatrix = alphamap;
    end
    
    % Creates file names
    figureFileName = strcat(folderName, '.fig');
    dataFileName = strcat(folderName, '.mat');
    mkdir(folderName);
    
    % Saves the figures and files
    savefig(figureArray, figureFileName);
    save(dataFileName, 'fit_curves', 'fit_parameters', 'sofiMapMatrix', ...
        'DmapMatrix', 'Dmap_correctedMatrix', 'RsquareMapMatrix', 'trimDmap2logMatrix', ...
        'D2_mapMatrix', 'D2_mapCorrectedMatrix', 'alphaMapMatrix', '-v7.3');
    
    % Moves the files into the folder created
    movefile(figureFileName, folderName);
    movefile(dataFileName, folderName);
    
    % If using background subtraction, then tiff-to-mat converted file and
    %   background subtraction file saved aswell
    if doFileConversionAndBackSubtraction
        noBackFileName = strcat(filenm, '_converted.mat');
        save(noBackFileName, 'Data', '-v7.3')
        movefile(noBackFileName, folderName);
        movefile(fname, folderName);
    end
    
end   

%% total computation time
totaltime = etime(clock,before);
fprintf(['\nTotal fcsSOFI execution time:' ' ' num2str(totaltime/60) ' ' 'minutes (' num2str(totaltime) ' ' 'seconds)\n\n']);

% write computation time to text file for remote access
if store_execution_times == 1
    DateString = datestr(datetime);
    xdim = num2str((xmax-xmin+1)); ydim = num2str((ymax-ymin+1));
    fid = fopen('fcsSOFI_execution_times.txt','at');
    fprintf(fid, [DateString '\n']);
    fprintf(fid, ['Data File:' ' ' fname '.m' ' (' xdim 'x' ydim ' ' 'image)\n']); 
    fprintf(fid, ['Total execution time:' ' ' num2str(totaltime/60) ' ' 'minutes (' num2str(totaltime) ' ' 'seconds)\n']);
    fprintf(fid, ['Total GPU Only time:' ' ' num2str(fit_time) ' ' 'seconds\n\n']);
    fclose(fid);
end
