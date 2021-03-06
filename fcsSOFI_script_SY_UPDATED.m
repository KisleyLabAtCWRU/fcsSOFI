%GPU-parallelized fcsSOFI script
%Will Schmid
%Kisley Lab

%Perform complete fcsSOFI analysis from start to finish.

clear; close all; clc
%% User Input

%data file name
fname = 'dataset77'; % don't include ".mat" in file name

%diffusion coefficient parameters
pixelsize=50; %47.6in nm; needed to accurately calculate D
dT=0.04; %in s; needed to accurately calculate D

% set PSF for deconvolution     
FWHM=2.7; %FWHM of PSF in pixels

%caxis scale for simulated data
minScale = 0;
maxScale = 15000;

%region of interest in pixels
ymin=1;%98; 
ymax=30;%173;
xmin=1;%98;
xmax=30;%258;
tmin = 1;
tmax= 5000;
    
%choose type of diffusion (1 = Brownian, 2 = 2-Comp Brownian, 3 = Anomalous, ...
        ... 4 = Brownian 1 Comp with tau, 5 = 1-comp Brownian with tau and A, ...
        ... 6 = Anomalous with Tau and alpha)
type = 4;

%choose alpha start point (Anomalous diffusion model only)
A_stp = 1;
alpha_stp = .9;
D_stp = 1e5;
D2_stp = 1e6;

A_stp_test = 1;
alpha_stp_test = 1;
D_stp_test = 1e4;

% alpha threshold
alpha_max = 1.2; % maximum value of alpha allowed to appear on alpha map
alpha_min = 0;

%set SOFI saturation limits
cminAC=0; % line 853 in GUI code
cmaxAC=3e8;

% set caxis limits for fcsSOFI diffusion map (color scaling)
cmin=2; 
cmax=6;

%number of fit iterations per pixel
number_fits = 1000;

%plot figures? (1 = yes)
plotfigures = 1;

% store execution times in external text file? (1 = yes);
store_execution_times = 0;

%save data files? (1 = yes)
savethedata = 0; 

%optional example single pixel curve fit plot
examplecf = 1; %plot example curve fit plot for single pixel?
row_index = 15; %pixel row index
column_index = 20; %pixel column index
row_index2 = 15;
column_index2 = 10;

%SOFI scaling
satmax = .5;
satmin = 0;


% END USER INPUT

%% Paths
addpath(strcat(pwd,'\gpufit\Debug\matlab'))
addpath(strcat(pwd,'\fcsSOFI_external_functions'))

%% %%%%%%%%%%%%%%%%%%%% STEP 1: blink_AConly (SOFI) %%%%%%%%%%%%%%%%%%%%%%%%%
% start global timer
before = clock;

% start timer for SOFI step
blink_before = clock;

%% start code/load and setup data 
fprintf('Running...');

% Load data
load(fname); clc;fprintf(strcat(fname,'.mat loaded\n'));
DataCombined = Data;

% Set ROI
DataCombined=DataCombined(ymin:ymax,xmin:xmax,tmin:tmax);

% produce average image
avgimage=sum(DataCombined(:,:,:),3)./size(DataCombined,3);

% Reshape 2D image to 1D for analysis
% shapes columnwise into a 1x(ymax-ymin+1)(xmax-xmin+1)
DataVector = zeros(tmax,(xmax-xmin+1)*(ymax-ymin+1));
for i=1:size(DataCombined,3)
    DataVector(i,:)=reshape(DataCombined(:,:,i),1,size(DataCombined(:,:,i),1)*...
        size(DataCombined(:,:,i),2));
end

% part of distance factor-- c = sigma    
  c = 3.5; %experimentally based

% only upper left points are run through analysis to avoid repetition
[innerpts] = getUpperLeft(ymax-ymin+1, xmax-xmin+1, 1);
L = ymax-ymin;
W = xmax-xmin;

%% Calculate the correlation (2-4th orders, AC and XC)
[ACXC_all]=CalcCorr(innerpts,DataVector,c,L); %calculate 

%% Calculate intensity for images by different methods     
AC_G2 = zeros(1,numel(ACXC_all));
for i=1:numel(ACXC_all)
    % AC only - first point of different orders
    AC_G2(i) = ACXC_all(1,i).Order2(1,2);
end

%% Reshape matrices for images
% reshape easy ones first
AC_G2_im=[0,AC_G2,0];% pad matrices with first points, so back to original size
AC_G2_im=reshape(AC_G2_im,L,W);% reshape

M = zeros(size(AC_G2_im));
M(1:end-1,:) = AC_G2_im(2:end,:);
M(end,:) = circshift(AC_G2_im(1,:),numel(AC_G2_im(1,:))-1);
AC_G2_im = M;

%% Deconvolution
avgim=avgimage;
im=AC_G2_im;

% define the PSF 
intensity = 1;
gauss1=customgauss([100 100],FWHM,FWHM,0,0,intensity,[5 5]); %create a 2D PSF
PSF=gauss1(45:65,45:65); %only use the center where the PSF is located at

filtim=deconvlucy(im,PSF); % Based on Geissbuehler bSOFI paper
%filtim=im;
% display execution time of SOFI step
blink_after = clock;
clc;fprintf('SOFI complete, execution time: %6.2f seconds\n',etime(blink_after,blink_before));

%% %%%%%%%%%%%%%%%%%%%% STEP 2: BinFitData (fcs) %%%%%%%%%%%%%%%%%%%%%%%%%

% start timer for fcs step
Bin_before = clock;

% ROI pixels of image to analyze
xmn=1; xmx=xmax-xmin;
ymn=1; ymx=ymax-ymin;

%% load data 

%reshape ACXC_all to get ROI
ACXC_all2=[ACXC_all,ACXC_all(1,1),ACXC_all(1,1)];
ACXC_all_reshape=reshape(ACXC_all2,size(AC_G2_im));
ACXC_all_ROI=ACXC_all_reshape(ymn:ymx,xmn:xmx);
AC_XC_all_save=reshape(ACXC_all_ROI,1,size(ACXC_all_ROI,1)*size(ACXC_all_ROI,2));
AC_all(1,1).curves=AC_XC_all_save;

for i=1:numel(AC_all(1,1).curves)
        % add raw AC curves
    ACadd(1,:)=AC_all(1,1).curves(1,i).Order2;
    AC_avg(1,i).curves=mean(ACadd,1);   
end

%% log bin the averaged data
for i=1:numel(AC_all(1,1).curves)

    AC_aver=(AC_avg(1,i).curves);
    max_lag=numel(AC_avg(1,i).curves);
    lags=1:max_lag;
    ddwell=1;

    [new_lags, new_AC] = logbindata(lags,AC_aver,ddwell,max_lag);

    AC_logbin(i,:)=new_AC;
    AC_loglag(i,:)=new_lags;

end

% take first lag point to create super-resolution image
AC_im = zeros(1,size(AC_logbin,1));
for j=1:size(AC_logbin,1)
    AC_im(j)=AC_logbin(j,2);
end

rowdim=size(ACXC_all_ROI,1);
coldim=size(ACXC_all_ROI,2);

AC_im2=reshape([AC_im],rowdim,coldim);

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
tauD = zeros(1,xmx*ymx); tauD2 = tauD; D = tauD; D2 = tauD; alpha = tauD;

%% Perform curve fitting
for i=1:size(AC_logbin,1)

    % display progress at 25%, 50%, and 75% complete
    if i == ceil(size(AC_logbin,1)/4)
        clc; fprintf('Curve-fitting 25%% complete');
    end
    if i == ceil(size(AC_logbin,1)/2)
        clc; fprintf('Curve-fitting 50%% complete');
    end
    if i == ceil(3/4*size(AC_logbin,1))
        clc; fprintf('Curve-fitting 75%% complete');
    end

    % extract auto correlation curve 
    ACcurve=AC_logbin(i,:);
    timelag=AC_loglag(i,:);     

    % convert to x and y variables
    x=(timelag.*dT); %convert x values to seconds   SHAWN 
    y=ACcurve;
    % remove first timelag point tau=lag
    ind=numel(x);
    x=x(2:ind);
    y=y(2:ind);
    y=y./max(y);
    
    % choose startpoint tau_D
    td_stp = (pixelsize^2)/(D_stp*4); %Shawn
    td2_stp = (pixelsize^2)/(D2_stp*4);
    %td_stp_test = (pixelsize^2)/(4*D_stp_test);
    %td_stp = 0.3;            
    %D = (pixelsize.^2)/(4*0.36787)
    
    % declare start points based on diffusion type
    sp_struct = struct; % start point structure
    sp_struct.brownian = [A_stp,mean(y(round((3*numel(y)/4)):numel(y))),td_stp];
    sp_struct.brownian2comp = [max(y),max(y),mean(y(round((3*numel(y)/4)):numel(y))),1/2*td_stp,1/2*td_stp];
    sp_struct.anomalous = [max(y)*2,mean(y(round((3*numel(y)/4)):numel(y))), td_stp, alpha_stp];
    sp_struct.browniannorm = td_stp;
    sp_struct.browniantaua = [A_stp,td_stp];
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
    %weights = data;%1./abs(sqrt(data));
    %weights = data./data;
    weights = data;
    weights(data < 0) = 0;
    % Run Gpufit
    [parameters, states, chi_squares, n_iterations, gputime] = gpufit(data, weights, ...
        model_id, initial_parameters, tolerance, max_n_iterations, [], estimator_id, user_info);
    %{
    % run Gpufit
    [parameters, states, chi_squares, n_iterations, gputime] = gpufit(data, [], ...
     model_id, initial_parameters, tolerance, max_n_iterations, [], estimator_id, user_info);
    %}
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
    n_struct = struct('brownian',3,'brownian2comp',[4; 5],'anomalous',3,'brownian_norm',1,...
                        'brownian_taua',2, 'anomalous_2param_taua',2);
    n_cell = struct2cell(n_struct);
    n = n_cell{type};
    len_x = numel(x);
    if type == 1
        model_fit(1:len_x) = model_coefs(1).* (1./(1+(x(1:len_x)./model_coefs(3)))) + model_coefs(2);
    elseif type == 2
        model_fit(1:len_x)= model_coefs(1).* (1./(1+(x(1:len_x)./model_coefs(4)))) + model_coefs(2).* (1./(1+(x(1:len_x)./model_coefs(5)))) + model_coefs(3);
    elseif type == 3
        model_fit(1:len_x) = model_coefs(1).* (1./(1+(x(1:len_x)./model_coefs(3)).^model_coefs(4))) + model_coefs(2); 
    elseif type == 4
        model_fit(1:len_x) = 1./(1+(x(1:len_x)./model_coefs(1)));
    elseif type == 5
        model_fit(1:len_x) = model_coefs(1).* (1./(1+(x(1:len_x)./model_coefs(2)))); 
    elseif type == 6
        %disp(model_coefs(1))
        model_fit(1:len_x) = (1./(1+(x(1:len_x)./model_coefs(1)).^model_coefs(2)));
        %model_fit(1:len_x) = (1./(1+(dT.*x(1:len_x)./((pixelsize^2)/(D_stp*4))).^model_coefs(2)));
    end

    % R-square
    residuals = y - model_fit;
    a = (y - model_fit).^2./model_fit;
    a(isinf(a)) = 0;
    rsquare = 1-sum(residuals.^2)/sum(y.^2);

    
    % fit result structure
    fitresult(1,i)=struct('rawdata',[x',y'],'rsquare',rsquare,'model_fit',model_fit);

    % characteristic time
    %tauD(i)=model_coefs(n(1)); % in in seconds THIS NEEDS TO BE CHANGED
    %SHAWN check integer or decimal value WRONG
    tauD(i)=model_coefs(1)*dT;
    %disp(tauD(i))
    %tauD(i) = ((pixelsize^2)/(D_stp*4));
    
    % diffusion coefficient
    D(i)=(pixelsize.^2)/(4*tauD(i)); %in nm^2/s
    %disp(D(i))
    % second diffusion coefficient if using 2-component model
    if type == 2
        tauD2(i)=model_coefs(n(2))*dT; % in in seconds
        D2(i)=(pixelsize.^2)/(4*tauD(i));
    end

    % alpha map if using anomalous model
    if type == 3
        alpha(i)= model_coefs(4);
    end
    
    % alpha map if using anomalous model with 2 parameters
    if type == 6
        alpha(i)= model_coefs(2);
    end

    % compute total Gpufit time
    fit_time = fit_time + gputime;
end


%% Post fit data manipulation
% reshape fit result 
fitresult2=reshape(fitresult,rowdim,coldim);

%Diffusion coefficient map
Dmap=reshape(D,rowdim,coldim);

% create tauD map
tauDmap=reshape(tauD,rowdim,coldim);

% remove poor fits
D_corrected = zeros(1,numel(D));
for i=1:numel(D)
    if fitresult(1,i).rsquare<0.
        D_corrected(i)=0;
    else
        D_corrected(i)=abs(D(i));
    end
end
Dmap_corrected=reshape([D_corrected],rowdim,coldim);

% second diffusion coefficeint map if 2-component brownian model
if type == 2
    D2map=reshape(D2,rowdim,coldim);
    tauD2map=reshape(tauD2,rowdim,coldim);
    D2_corrected = zeros(1,numel(D2));
    for i=1:numel(D2)
        if fitresult(1,i).rsquare<0.
            D2_corrected(i)=0;
        else
            D2_corrected(i)=abs(D2(i));
        end
    end
D2map_corrected=reshape(D2_corrected,rowdim,coldim);
end

% alpha map (anomalous diffusion)
if type == 3   ||  type == 6
    alpha_corrected = zeros(1,numel(alpha));
    % remove bad alphas
    for i=1:numel(alpha)
        if fitresult(1,i).rsquare<0.
            alpha_corrected(i)=0;
        elseif alpha(i) < 0
            alpha_corrected(i)=0;
        elseif alpha(i)< alpha_min
            alpha_corrected(i)=0;
        elseif alpha(i) > alpha_max
            alpha_corrected(i)=0;
        else
            alpha_corrected(i)=abs(alpha(i));
        end
    end
    alphamap=reshape([alpha_corrected],rowdim,coldim);
    %alphamap(alphamap > alpha_max) = alpha_max;
end

% make map of R^2 values
R2 = zeros(1,numel(fitresult));
for i=1:numel(fitresult)
    R2(i)=fitresult(1,i).rsquare;
end
R2map=reshape(R2,rowdim,coldim);

name = ["Brownian","2-Component Brownian","Anomalous","Brownian 1 Component Normalized",...
            "Brownian 1 Component with Amplitude", "Anomalous 2 Parameters Tau and Alpha"];name = name(type);

% display execution time of fcs step
Bin_after = clock;
fprintf('FCS complete, execution time: %6.2f seconds\n',etime(Bin_after,Bin_before));

%% %%%%%%%%%%%%%%%%%%%% STEP 3: CombineTempSpat (fcsSOFI) %%%%%%%%%%%%%%%%%%%%%%%%%

% start timer for fcsSOFI combination
Combine_before = clock;
    
%% Diffusion map
szmap1=size(Dmap,1);
szmap2=size(Dmap,2);

Dmap2log=log10(Dmap); 

for i=1:size(Dmap2log,1)
    for j=1:size(Dmap2log,2)
        if Dmap2log(i,j)==-Inf
            Dmap2log(i,j)=0;
        end
    end
end

%filter out poor fits if you want to here with an R2cutoff
R2cutoff=0; %set R^2 cutoff (0-1); == 0, no filtering
for i=1:size(Dmap2log,1)
    for j=1:size(Dmap2log,2)
        if R2map(i,j)<R2cutoff
            Dmap2log(i,j)=0;
        end
    end
end

normcoef=10; %instead set to standard here
normDmap2log=Dmap2log./normcoef;

for i=1:size(normDmap2log,1)
    for j=1:size(normDmap2log,2)
        if normDmap2log(i,j)==0
            normDmap2log(i,j)=0;
        elseif normDmap2log(i,j)<cmin/normcoef
            normDmap2log(i,j)=cmin/normcoef;
        elseif normDmap2log(i,j)>cmax/normcoef
            normDmap2log(i,j)=cmax/normcoef;
        end
    end
end
     
%% Create scaling/stretching/shifting factor to create colormap
%changing shift, scale will change the colormap colors, range
maxvalue=cmax/normcoef; %this is the max value on the colormap
shift=0.05;%shift left right
scale=0.7;%factor to multiply everything by 

for i=1:size(normDmap2log,1)
    for j=1:size(normDmap2log,2)
        if normDmap2log(i,j)==0
            normDmap2log(i,j)=0;
        else
            normDmap2log(i,j)=(normDmap2log(i,j)-shift).*scale;
        end
    end
end

% %create colormap from the cmin/cmax
ncmin=((cmin/normcoef)-shift).*scale;
ncmax=((cmax/normcoef)-shift).*scale;
steps=25; %number of steps between min/max
cgrad=[ncmin:((ncmax-ncmin)/steps):ncmax]';%create gradient

%set cmap the same way as diffusion map 
cmap(1:steps+1,1,1)=cgrad; %set hue = color
cmap(1:steps+1,1,2)=ones(steps+1,1); %set saturation ("gray")
cmap(1:steps+1,1,3)=ones(steps+1,1); %set brightness
rgb_cmap=1-hsv2rgb(cmap); %convert to rgb

dmap_hsv(1:szmap1,1:szmap2,1)=normDmap2log; %set hue = color
dmap_hsv(1:szmap1,1:szmap2,2)=ones(szmap1,szmap2); %set saturation ("gray")
dmap_hsv(1:szmap1,1:szmap2,3)=ones(szmap1,szmap2); %set brightness
dmap_hsv=1-hsv2rgb(real(dmap_hsv)); %convert to rgb
for i=1:size(dmap_hsv,1) %filter out background rgb(D)=0, then make pixel black
    for j=1:size(dmap_hsv,2)
        if dmap_hsv(i,j,1)==0
            dmap_hsv(i,j,:)=0;
        end
    end
end

%% make super resolution only map
% set caxis limits for SR map
cscaleACim=filtim;
%{
for i=1:size(cscaleACim,1) %filter out values above/below, change to limits
    for j=1:size(cscaleACim,2)
        if cscaleACim(i,j)<cminAC
            cscaleACim(i,j)=cminAC;
        elseif cscaleACim(i,j)>cmaxAC
            cscaleACim(i,j)=cmaxAC;
        end
    end
end
%}
norm_ca_AC=cscaleACim./(max(max(cscaleACim))); %normalize



% HSV values
srmap_hsv(1:szmap1,1:szmap2,1)=ones(szmap1,szmap2); %set hue = color
% srmap_hsv(1:szmap1,1:szmap2,2)=AC_G2_im./(max(max(AC_G2_im))); %set saturation ("gray")
% srmap_hsv(1:szmap1,1:szmap2,2)=AC_im2./(max(max(AC_im2))); %set saturation ("gray")
srmap_hsv(1:szmap1,1:szmap2,2)=norm_ca_AC;
srmap_hsv(1:szmap1,1:szmap2,3)=ones(szmap1,szmap2); %set brightness
srmap_hsv=1-hsv2rgb(srmap_hsv); %convert to rgb
srmap_hsv=rgb2gray(srmap_hsv);% convert to gray scale

%% set the limits of the SOFI image
%Shawn special start

for i=1:size(srmap_hsv,1)
    for j=1:size(srmap_hsv,2)
        if srmap_hsv(i,j)<satmin
            srmap_hsv(i,j)=satmin;
        end
        if srmap_hsv(i,j)>satmax
            srmap_hsv(i,j)=satmax;
        end
    end
end
srmap_hsv=srmap_hsv./max(max(srmap_hsv)); %renormalize so on scale 0-1

%Shawn special end


%% combine D and Super res. for HSV
hsvmap(1:szmap1,1:szmap2,1)=normDmap2log; %set hue = color
hsvmap(1:szmap1,1:szmap2,2)=srmap_hsv;%norm_ca_AC;% Shawn change
hsvmap(1:szmap1,1:szmap2,3)=ones(szmap1,szmap2); %set brightness
hsv2rgbmap=1-hsv2rgb(real(hsvmap)); %convert to rgb


% display execution time of fcsSOFI combination
Combine_after = clock;
clc;fprintf('Image fusion complete, execution time: %6.2f seconds\n',etime(Combine_after,Combine_before));

%% Figures

if plotfigures == 1 
    fprintf('\nPlotting...');
    
    % blinkAConly Subplots (SOFI)
    figure% average image
    subplot(2,2,1)
    imagesc(avgim);
    axis image
    title('Average image')
    set(gca,'xtick',[],'ytick',[])
    subplot(2,2,2)
    imagesc(im) %autocorrelation
    axis image
    title('AC G^2(0+\tau)')
    set(gca,'xtick',[],'ytick',[])
    subplot(2,2,3)
    imagesc(filtim) %deconvolved
    axis image
    title('Deconvolved')
    set(gca,'xtick',[],'ytick',[])
    subplot(2,2,4) %line sections
    hold on
    plot(avgim(25,:)./max(avgim(25,:)),'-b')
    plot(im(25,:)./max(im(25,:)),'-r')
    plot(filtim(25,:)./max(filtim(25,:)),'-k')
    axis square
    title('Line sections')
    ylim([0 1])
    xlim([0 size(im(25,:),2)])

    % BinFitData subplots (fcs) 
    figure %tauDmap
    subplot(2,2,1)
    imagesc(tauDmap);
    axis image
    title('\tau_D map')
    subplot(2,2,2) % figure of D with nothing removed 
    imagesc(Dmap);
    caxis([0 max(max((Dmap)))])
    axis image
    title('D map, nothing removed')
    subplot(2,2,3) % figure of D with poor fits removed
    imagesc(Dmap_corrected)
    caxis([0 max(max(Dmap_corrected))])
    axis image
    title('Fits of R^2 < 0.5 removed')
    subplot(2,2,4) % figure of D on log scale
    Dmap2log=log10(Dmap_corrected);
    imagesc(Dmap2log)
    caxis([0 max(max(Dmap2log))])
    axis image
    title('log scale c axis')

    % second set of BinFitData subplots if using 2-comp model
    if type == 2
        figure % tauD map
        subplot(2,2,1)
        imagesc(tauD2map);
        axis image
        title('D2: \tau_D map')
        subplot(2,2,2)
        imagesc(D2map); % figure of D with nothing removed
        caxis([0 max(max((D2map)))])
        axis image
        title('D2: D map, nothing removed')
        subplot(2,2,3) % figure of D2 with poor fits removed
        imagesc(D2map_corrected)
        caxis([0 max(max(D2map_corrected))])
        axis image
        title('D2: Fits of R^2 < 0.5 removed')
        subplot(2,2,4) % figure of D2 on log scale
        D2map2log=log10(D2map_corrected);
        imagesc(D2map2log)
        caxis([0 max(max(D2map2log))])
        axis image
        title('D2: log scale c axis')
    end
    
    % alpha map if using anomalous model
    if type == 3 || type == 6
        
        sofiBinarized = imbinarize(srmap_hsv, 0.05);
        %sofiBinarized = imbinarize(srmap_hsv, 'adaptive');
        alphamap(sofiBinarized == 0) = 0;
        
        figure
        imagesc(alphamap);
        caxis([0 max(max((alphamap)))])
        axis image
        title('\alpha map')
        c=colorbar;
        c.Label.String = '\alpha';
        c.Label.FontSize = 20;
        set(gca,'xtick',[],'ytick',[])
    end

    % R-square Map
    h2 = figure(4);
    imagesc(R2map)
    caxis([0 1])
    axis image
    title('R^2 map')
    colorbar

    % CombineTempSpat subplots
    h3 = figure;
    l=subplot(1,2,1); % log(D) map
    subplot(1,2,1);
    imagesc(dmap_hsv);
    axis image
    title('FCS: log(D)')
    set(gca,'xtick',[],'ytick',[])
    c=subplot(1,2,2); %colormap
    imagesc(rgb_cmap)
    sz2=get(l,'position');
    % set(c,'size',[sz(4)-sz(3)])
    set(gca,'ylim',[1 26],'ytick',[1 steps+1],'yticklabel',[num2str(cmin); num2str(cmax)],...
        'xtick',[],'position',[0.77 0.413 0.02 0.22])
    ylabel('log(D) (nm^2/s)')
    axis xy
    set(findall(gcf,'-property','FontSize'),'FontSize',14)

    % SOFI super resolution image
    figure;
    imagesc(srmap_hsv);
    axis image
    title('SOFI super-resolution')
    set(gca,'xtick',[],'ytick',[])
    colormap(gray)

    % Combined fcsSOFI image
    figure;
    k=subplot(1,2,1);
    imagesc(hsv2rgbmap) % combined
    sz=get(l,'position');
    axis image
    title('Combined fcsSOFI image')
    set(gca,'xtick',[],'ytick',[])
    c=subplot(1,2,2); %colormap
    imagesc(rgb_cmap)
    sz2=get(l,'position');
    % set(c,'size',[sz(4)-sz(3)])
    set(gca,'ylim',[1 26],'ytick',[1 steps+1],'yticklabel',[num2str(cmin); num2str(cmax)],...
        'xtick',[],'position',[0.77 0.413 0.02 0.22])
    ylabel('log(D) (nm^2/s)')
    axis xy
    set(findall(gcf,'-property','FontSize'),'FontSize',14)

end

%% Single Pixel Results
    name = ["Brownian","2-Component Brownian","Anomalous","Brownian Norm",...
        "Brownian 1-Component Normalized", "Brownian 1-Component with Amplitude",...
        "Anomalous 2 Parameters Tau and Alpha"];name = name(type);

    % optional single pixel curve fit result figure
    if examplecf == 1
        i=row_index; %row index
        j=column_index; %column index
        figure;
        x2=fitresult2(i,j).rawdata(:,1); 
        y2=fitresult2(i,j).rawdata(:,2);
        plot(x2,y2,'or','LineWidth',2)
        hold on
        plot(x2,fitresult2(i,j).model_fit,'--k','LineWidth',2)
        hold on
        %plot(x2,1./(1+(x2./model_coefs(1))),'--b','LineWidth',2)
        %plot(x2,1./(1+(x2./converged_parameters(1))),'--b','LineWidth',2)
        %hold on
        %plot(x2,1./(1+(dT.*x2./((47.6^2)/(4*10^6))^.7787)),'--b','LineWidth',2)
        %hold on
        %plot(x2,1./(1+(x2./td2_stp)),'--r','LineWidth',2)
        set(gca,'xscale','log')
        xlabel('\tau')
        ylabel('G(\tau)')
        legend('Raw Data','Fit Result');
        title(strcat(name,' Diffusion Curve Fit with Gpufit')); 
        
        %______________________________________
        %{
        i=row_index2; %row index
        j=column_index2; %column index
        figure;
        x2=fitresult2(i,j).rawdata(:,1);
        y2=fitresult2(i,j).rawdata(:,2);
        plot(x2,y2,'or','LineWidth',2)
        hold on
        plot(x2,fitresult2(i,j).model_fit,'--k','LineWidth',2)
        hold on
        
        %Making a sample plot to test D accuracy
        td_stp_test = (pixelsize^2)/(4*1*10^5)/dT;
        alpha_stp_test = 1;
        plot(x2,1.5./(1+(x2./td_stp_test).^alpha_stp_test),'--b','LineWidth',2)
        %}
        set(gca,'xscale','log')
        xlabel('\tau')
        ylabel('G(\tau)')
        legend('Raw Data','Fit Result');
        title(strcat(name,' Diffusion Curve Fit with Gpufit')); 
        
        %______________________________________
        
        % error bars
        number_parameters = [3; 5; 4; 1; 2; 2];
        [rsq,chisq,J,MSE,ci] = gofStats(type,...%type
            converged_parameters(1:number_parameters(type)),... %parameter values
            fitresult2(row_index,column_index).model_fit,...    %fit curve
            fitresult2(row_index,column_index).rawdata(:,1)',... %x data
            fitresult2(row_index,column_index).rawdata(:,2)');   %y data
        gof_gpu = [rsq chisq];
        ci = ci';

        ebars = zeros(1,size(ci,2));
        ebars(1:size(ci,2)) = abs(ci(2,1:size(ci,2))-ci(1,1:size(ci,2)))/2;
        
        modeleqn = ["G(tau) = a * 1/(1 + tau/tauD) + b",...
            "G(tau) = a1 * 1/(1 + tau/tauD1) + a2 * 1/(1 + tau/tauD2) + b",...
            "G(tau) = a * 1/(1 + (tau/tauD)^alpha) + b", ...
            "G(tau) = 1/(1 + tau/tauD)", ...
            "G(tau) = a * 1/(1 + tau/tauD)",...
            "G(tau) = 1/(1 + (tau/tauD)^alpha)"]; modeleqn = modeleqn(type);

        clc; fprintf(fname);fprintf('\nPixel (%i,%i)\n',row_index,column_index );
        fprintf(strcat(name,' Fit Model:\n',modeleqn,'\n\n'));
        fprintf('Fit Result Parameters\n');
        
        % print error bars
        if type == 1
          fprintf('a =    %6.2e � %6.2e\n',converged_parameters(1),ebars(1));
          fprintf('b =    %6.2e � %6.2e\n',converged_parameters(2),ebars(2));
          fprintf('tauD =     %6.2e � %6.2e\n',converged_parameters(3),ebars(3));
          fprintf('\n')
          fprintf('D =         %6.3e\n',Dmap_corrected(row_index,column_index ))
          fprintf('log10(D):  %6.3f\n\n',Dmap2log(row_index,column_index ))
          fprintf('R-square:  %6.4f\n',R2map(row_index,column_index ));
        
        elseif type == 2
          fprintf('a1 =    %6.2e � %6.2e\n',converged_parameters(1),ebars(1));
          fprintf('a2 =    %6.2e � %6.2e\n',converged_parameters(2),ebars(2));
          fprintf('b =     %6.2e � %6.2e\n',converged_parameters(3),ebars(3));
          fprintf('tauD1 = %6.4f � %6.4e\n',converged_parameters(4),ebars(4));
          fprintf('tauD2 = %6.4f � %6.4e\n',converged_parameters(5),ebars(5));  
          fprintf('\n')
          fprintf('D1:         %6.3e\n',Dmap_corrected(row_index,column_index ))
          fprintf('D2:         %6.3e\n',D2map_corrected(row_index,column_index ))
          fprintf('log10(D1):  %6.3f\n',Dmap2log(row_index,column_index ))
          fprintf('log10(D2):  %6.3f\n\n',D2map2log(row_index,column_index ))
          fprintf('R-square:  %6.4f\n',R2map(row_index,column_index ));   

        elseif type == 3
          fprintf('a =    %6.2e � %6.2e\n',converged_parameters(1),ebars(1));
          fprintf('b =    %6.2e � %6.2e\n',converged_parameters(2),ebars(2));
          fprintf('tauD =     %6.2e � %6.2e\n',converged_parameters(3),ebars(3));
          fprintf('alpha = %6.4f � %6.4f\n',converged_parameters(4),ebars(4));
          fprintf('\n')
          fprintf('D =         %6.3e\n',Dmap_corrected(row_index,column_index ))
          fprintf('log10(D):  %6.3f\n\n',Dmap2log(row_index,column_index ))
          fprintf('alpha:     %6.4f\n\n',alphamap(row_index,column_index ))
          fprintf('R-square:  %6.4f\n',R2map(row_index,column_index ));
          
        elseif type == 4
          fprintf('tauD =     %6.2e � %6.2e\n',converged_parameters(1),ebars(1));
          fprintf('\n')
          fprintf('D =         %6.3e\n',Dmap_corrected(row_index,column_index ))
          fprintf('log10(D):  %6.3f\n\n',Dmap2log(row_index,column_index ))
          fprintf('R-square:  %6.4f\n',R2map(row_index,column_index ));    
        
        elseif type == 5
          fprintf('a =    %6.2e � %6.2e\n',converged_parameters(1),ebars(1));
          fprintf('tauD =     %6.2e � %6.2e\n',converged_parameters(3),ebars(3));
          fprintf('\n')
          fprintf('D =         %6.3e\n',Dmap_corrected(row_index,column_index ))
          fprintf('log10(D):  %6.3f\n\n',Dmap2log(row_index,column_index ))
          fprintf('R-square:  %6.4f\n',R2map(row_index,column_index ));
          
       elseif type == 6
          fprintf('tauD =     %6.2e � %6.2e\n',converged_parameters(1),ebars(1));
          fprintf('alpha = %6.4f � %6.4f\n',converged_parameters(2),ebars(2));
          fprintf('\n')
          fprintf('D =         %6.3e\n',Dmap_corrected(row_index,column_index ))
          fprintf('log10(D):  %6.3f\n\n',Dmap2log(row_index,column_index ))
          fprintf('alpha:     %6.4f\n\n',alphamap(row_index,column_index ))
          fprintf('R-square:  %6.4f\n',R2map(row_index,column_index ));
        end
        
    end

%% save data
if savethedata == 1 
    
    date = datestr(now,'mm-dd-yyyy_HH-MM');
    fit_curves = fitresult2;
    fit_parameters = double(converged_parameters);
    SOFI = srmap_hsv;
    D_map = Dmap;
    D_map_corrected = Dmap_corrected;
    Rsquare_map = R2map;
    fcsSOFI = hsv2rgbmap;
    fcsSOFI_cmap = rgb_cmap;
    if type == 1
         save(strcat(fname,'_analyzed_brownian_',date),'fit_curves','fit_parameters','SOFI','D_map','D_map_corrected','Rsquare_map','fcsSOFI','fcsSOFI_cmap',...
             'normDmap2log','normcoef','filtim','type','szmap1','szmap2');
    elseif type == 2
        D2_map = D2map;
        D2_map_corrected = D2map_corrected;
        save(strcat(fname,'_analyzed_2comp_',date),'fit_curves','fit_parameters','SOFI','D_map','D_map_corrected','D2_map','D2_map_corrected','Rsquare_map','fcsSOFI','fcsSOFI_cmap',...
            'normDmap2log','normcoef','filtim','type','szmap1','szmap2');
    elseif type == 3
        alpha_map = alphamap;
        save(strcat(fname,'_analyzed_anomalous_',date),'fit_curves','fit_parameters','SOFI','D_map','D_map_corrected','alpha_map','Rsquare_map','fcsSOFI','fcsSOFI_cmap',...
            'normDmap2log','normcoef','filtim','type','szmap1','szmap2');
    elseif type == 4
         save(strcat(fname,'_analyzed_brownian_norm_',date),'fit_curves','fit_parameters','SOFI','D_map','D_map_corrected','Rsquare_map','fcsSOFI','fcsSOFI_cmap',...
             'normDmap2log','normcoef','filtim','type','szmap1','szmap2');
    elseif type == 5
         save(strcat(fname,'_analyzed_brownian_2_parameters_',date),'fit_curves','fit_parameters','SOFI','D_map','D_map_corrected','Rsquare_map','fcsSOFI','fcsSOFI_cmap',...
             'normDmap2log','normcoef','filtim','type','szmap1','szmap2'); 
    elseif type == 6
         save(strcat(fname,'_analyzed_anomalous_2_parameters_',date),'fit_curves','fit_parameters','SOFI','D_map','D_map_corrected','Rsquare_map','fcsSOFI','fcsSOFI_cmap',...
             'normDmap2log','normcoef','filtim','type','szmap1','szmap2');
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