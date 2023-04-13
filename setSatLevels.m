%% Info %%
%{
This is used to take already analyzed data from the fcsSOFI script
and create new images based on the given sat values.
Is much faster than re running the data all the way through.
Must take the data file saved from the fcsSOFI script.

THE FCSSOFI SCRIPT MUST NEEDS TO BE FROM 4/4/2023 OR LATER.
If you ran your data on an older version of fcsSOFI it will not have to 
varibalessaved need for this script.
%}

%% User Input %%

% Load Saved Data
start = 'Your Start Location';
[fileName, path] = uigetfile(start, "-mat");
addpath(path);
load(fileName);

%% 

% User Inpute
satMin = 0;
satMax = 0.5;
crossSatMax = satMax + 0.1;

saveFigures = 0;

%% Re adjust the sat max %%
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


%% Re Plot %%

xmax = size(sofiMap, 1);
ymax = size(sofiMap, 2);

% fcs figure creation
figureNumber = 1; 
figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
DFigure = imagesc(trimDmap2log); axis image; title('FCS: log(D)')
DFigure.AlphaData = Dmap2logAlpha; colormap(customColorMap);
c = colorbar; c.Label.String = 'log(D/(\mum^2s^{-1}))';
patch([0 xmax xmax 0], [0 0 ymax ymax], 'k'); % Patches a black background in front
set(gca, 'children', flipud(get(gca, 'children'))); % Moves Black Background to back
set(gca, 'FontSize', 14); set(gca, 'xtick', [], 'ytick', []) % Removes axis tick marks

% Large fcs figure creation
figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
DFigure = imagesc(largTrimDmap2log); axis image; title('Cross FCS: log(D)')
DFigure.AlphaData = largDmap2logAlpha; colormap(customColorMap);
c = colorbar; c.Label.String = 'log(D/(\mum^2s^{-1}))';
patch([1 (xmax)*2 (xmax)*2 1], [1 1 (ymax)*2 (ymax)*2], 'k'); % Patches a black background in front
set(gca, 'children', flipud(get(gca, 'children'))); % Moves Black Background to back
set(gca, 'FontSize', 14); set(gca, 'xtick', [], 'ytick', []) % Removes axis tick marks

% SOFI super resolution image
figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
imagesc(sofiMapSat); axis image; title('SOFI super-resolution')
set(gca, 'xtick', [], 'ytick', []); colormap(gray)

% Decon SOFI super resolution image
figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
imagesc(sofiMapDeconSat); axis image; title('SOFI super-resolution with Decon')
set(gca, 'xtick', [], 'ytick', []); colormap(gray)

% Large SOFI super resolution image
figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
imagesc(crossSofiMapSat); axis image; title('Cross SOFI super-resolution')
set(gca, 'xtick', [], 'ytick', []); colormap(gray)

% Large Decon SOFI super resolution image 
figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
imagesc(crossSofiMapDeconSat); axis image; title('Cross SOFI super-resolutionwith Decon')
set(gca, 'xtick', [], 'ytick', []); colormap(gray)

% fcsSOFI figure creation 
figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
fcsSofiPlot = imagesc(trimDmap2log); axis image; title('Combined fcsSOFI image')
fcsSofiPlot.AlphaData = sofiMapSat .* Dmap2logAlpha; % Uses the SOFI data as a transparency map
colormap(customColorMap); c = colorbar; c.Label.String = 'log(D/(\mum^2s^{-1}))';
patch([1 xmax xmax 1], [1 1 ymax ymax], 'k'); % Patches a black background in front
set(gca, 'children', flipud(get(gca, 'children'))); % Moves Black Background to back
set(gca, 'FontSize', 14); set(gca,'xtick',[],'ytick',[])

% Decon fcsSOFI figure creation
figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
fcsSofiPlot = imagesc(trimDmap2log); axis image; title('Combined fcsSOFI image with Decon')
fcsSofiPlot.AlphaData = sofiMapDeconSat .* Dmap2logAlpha; % Uses the SOFI data as a transparency map
colormap(customColorMap); c = colorbar; c.Label.String = 'log(D/(\mum^2s^{-1}))';
patch([1 xmax xmax 1], [1 1 ymax ymax], 'k'); % Patches a black background in front
set(gca, 'children', flipud(get(gca, 'children'))); % Moves Black Background to back
set(gca, 'FontSize', 14); set(gca,'xtick',[],'ytick',[])

% Large fcsSOFI figure creation 
figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
fcsSofiPlot = imagesc(largTrimDmap2log); axis image; title('Combined Cross fcsSOFI image')
fcsSofiPlot.AlphaData = crossSofiMapSat .* largDmap2logAlpha; % Uses the SOFI data as a transparency map
colormap(customColorMap); c = colorbar; c.Label.String = 'log(D/(\mum^2s^{-1}))';
patch([1 (xmax)*2 (xmax)*2 1], [1 1 (ymax)*2 (ymax)*2], 'k'); % Patches a black background in front
set(gca, 'children', flipud(get(gca, 'children'))); % Moves Black Background to back
set(gca, 'FontSize', 14); set(gca,'xtick',[],'ytick',[])

% Large Decon fcsSOFI figure creation 
figureArray(figureNumber) = figure; figureNumber = figureNumber + 1;
fcsSofiPlot = imagesc(largTrimDmap2log); axis image; title('Combined Cross fcsSOFI image with Decon')
fcsSofiPlot.AlphaData = crossSofiMapDeconSat .* largDmap2logAlpha; % Uses the SOFI data as a transparency map
colormap(customColorMap); c = colorbar; c.Label.String = 'log(D/(\mum^2s^{-1}))';
patch([1 (xmax)*2 (xmax)*2 1], [1 1 (ymax)*2 (ymax)*2], 'k'); % Patches a black background in front
set(gca, 'children', flipud(get(gca, 'children'))); % Moves Black Background to back
set(gca, 'FontSize', 14); set(gca,'xtick',[],'ytick',[])

%% Save Figures %%

if saveFigures
    %fileName = erase(fileName, '.mat');
    fileName = strcat(fileName, '_satMax', num2str(satMax), '-', num2str(crossSatMax), '.fig');
    savefig(figureArray, fileName);
end

