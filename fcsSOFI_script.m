%fcsSOFI_WS200131
%Will Schmid
%Kisley Lab

%Perform complete fcsSOFI analysis from start to finish.

clear; close all; clc
%% User Input

%data file name
fname = 'dataset82.mat';

%diffusion coefficient parameters
pixelsize=50; %in nm; needed to accurately calculate D
dT=0.04; %in s; needed to accurately calculate D

%caxis scale for simulated data
minScale = 0;
maxScale = 15000;

%region of interest in pixels
ymin=1; 
ymax=30;
xmin=1;
xmax=30;
tmin = 1;
tmax= 1000;
    
%choose type of diffusion (1 = Brownian, 2 = 2-Comp Brownian, 3 = Anomalous)
type = 3;

%choose alpha start point (Anomalous diffusion model only)
alpha_stp = 0.7;

%%%NEW alpha threshold
alpha_max = 2; % maximum value of alpha allowed to appear on alpha map
alpha_min = 0;

%set SOFI saturation limits
cminAC=0; % line 853 in GUI code
cmaxAC=3e8;

% set caxis limits for fcsSOFI diffusion map (color scaling)
cmin=2; %line
cmax=6;

% set PSF for deconvolution     
FWHM=2.7; %FWHM of PSF in pixels

%plot figures? (1 = yes)
plotfigures = 1;

%save data files? (1 = yes)
savethedata = 0; 
manipulatable_data = 0; % 1 = "I want to be able to scale color later" 

%number of fits per pixel
number_fits = 1000;

%optional example single pixel curve fit plot
examplecf = 1; %plot example curve fit plot for single pixel?
row_index = 8; %pixel row index
column_index = 18; %pixel column index


%%%% END USER INPUT %%%%












%% Paths
addpath(strcat(pwd,'\Gpufit_build-64_20190709\Debug\matlab'))
addpath(strcat(pwd,'\fcsSOFI_external_functions'))

%% %%%%%%%%%%%%%%%%%%%% STEP 1: blink_AConly (SOFI) %%%%%%%%%%%%%%%%%%%%%%%%%
before = clock;
blink_before = clock;

trialnos = 1;
% parameter = 'dataset'; %generic data set file name extension
% trialnos=[77]; %data set number (can analyze multiple data sets by entering multiple numbers)
for j = 1:numel(trialnos)
    %% %%%%%%%%%%%%%%%% start code/load and setup data %%%%%%%%%%%%%%%%%%%%
    fprintf('Running...');
%     fname = strcat(parameter,num2str(trialnos(j)));
    simpath=[fname];

    %determine length of trajectory
    intlength = tmax-tmin+1;

    % Load data
    load(fname); clc;fprintf(strcat(fname,'.mat loaded\n'));
    DataCombined=Data;

    % Set ROI
    DataCombined=DataCombined(ymin:ymax,xmin:xmax,tmin:tmax);
    size(DataCombined);

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

    %% %%%%%%%%%% Calculate the correlation (2-4th orders, AC and XC) %%%%%% %%
    rcSOFItime=tic;
    [ACXC_all]=CalcCorr(innerpts,DataVector,c,L); %calculate 


    %% %%%%%%%%%% Calculate intensity for images by different methods %%%%%%%%      
    AC_G2 = zeros(1,numel(ACXC_all));
    for i=1:numel(ACXC_all)
        % AC only - first point of different orders
        AC_G2(i) = ACXC_all(1,i).Order2(1,2);
    end

    %% %%%%%%%%%%%%%%%%%%% Reshape matrices for images %%%%%%%%%%%%%%%%%%%%% %%
    % reshape easy ones first
    AC_G2_im=[0,AC_G2,0];% pad matrices with first points, so back to original size
    AC_G2_im=reshape(AC_G2_im,L,W);% reshape

    %% Deconvolution
    avgim=avgimage;
    im=AC_G2_im;
    % define the PSF 

    intensity = 1;
    gauss1=customgauss([100 100],FWHM,FWHM,0,0,intensity,[5 5]); %create a 2D PSF
%%%%%%%%%%%%% OPTION TO INPUT PSF
PSF=gauss1(45:65,45:65); %only use the center where the PSF is located at

    filtim=deconvlucy(im,PSF); % Based on Geissbuehler bSOFI paper
    
end
blink_after = clock;
clc;fprintf('SOFI complete, execution time: %6.2f seconds\n',etime(blink_after,blink_before));

%% %%%%%%%%%%%%%%%%%%%% STEP 2: BinFitData (SOFI) %%%%%%%%%%%%%%%%%%%%%%%%%

% can we parallelize this further? MATLAB packages?
for index = 1:numel(trialnos)
% load(strcat(parameter,num2str(trialnos(index)),'analyzed.mat'));
Bin_before = clock;

% ROI pixels of image to analyze
xmn=1; xmx=xmax-xmin;
ymn=1; ymx=ymax-ymin;



%% load data 

uplind=sub2ind(size(AC_G2_im),ymn,xmn);
uprind=sub2ind(size(AC_G2_im),ymn,xmx);
dwlind=sub2ind(size(AC_G2_im),ymx,xmn);
dwrind=sub2ind(size(AC_G2_im),ymx,xmx);
    
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
number_parameters = [3; 5; 4]; number_parameters = number_parameters(type);

% estimator id
estimator_id = EstimatorID.LSE;
    
% model ID
model_id = [ModelID.BROWNIAN_1COMP; ModelID.BROWNIAN_2COMP; ModelID.ANOMALOUS]; model_id = model_id(type);

% tolerance
tolerance = 1e-3;

% maximum number of iterations per pixel
max_n_iterations = 1000;

% preallocate variables 
tauD = zeros(1,xmx*ymx); tauD2 = tauD; D = tauD; D2 = tauD; alpha = tauD;

%% Perform curve fitting
for i=1:size(AC_logbin,1)
    
    if i == ceil(size(AC_logbin,1)/4)
        clc; fprintf('Curve-fitting 25%% complete');
    end
    if i == ceil(size(AC_logbin,1)/2)
        clc; fprintf('Curve-fitting 50%% complete');
    end
    if i == ceil(3/4*size(AC_logbin,1))
        clc; fprintf('Curve-fitting 75%% complete');
    end
    
    ACcurve=AC_logbin(i,:);
    timelag=AC_loglag(i,:);     
    
    x=timelag.*dT; %convert x values to seconds   
    y=ACcurve;
    
    %remove first timelag point tau=lag
    ind=numel(x);
    x=x(2:ind);
    y=y(2:ind);

%     td_stp = max(x)/2;
    td_stp = 0.3678795;


   %declare start points based on diffusion type
    sp_struct = struct; % start point structure
    sp_struct.brownian = [max(y)*2,mean(y(round((3*numel(y)/4)):numel(y))),td_stp];
    sp_struct.brownian2comp = [max(y),max(y),mean(y(round((3*numel(y)/4)):numel(y))),1/2*td_stp,1/2*td_stp];
    sp_struct.anomalous = [max(y)*2,mean(y(round((3*numel(y)/4)):numel(y))), td_stp, alpha_stp];
    sp_cell = struct2cell(sp_struct);
    start_points = sp_cell{type};
    
    % initial parameters
    initial_parameters = repmat(single(single(start_points)'), [1, number_fits]);

    % fit data
    data = single(y); data = repmat(data(:), [1, number_fits]);

    % user info (independent variables)
    user_info = single(x);

    % Run Gpufit
    [parameters, states, chi_squares, n_iterations, gputime] = gpufit(data, [], ...
     model_id, initial_parameters, tolerance, max_n_iterations, [], estimator_id, user_info);

    % converged parameters
    converged = states == 0; 
    converged_parameters = parameters(:, converged);

    if isempty(converged_parameters) == 1
        model_coefs = parameters(1:number_parameters);
    else
        model_coefs = converged_parameters(1:number_parameters);
    end
    
    len_x = numel(x);
    
    % construct fit result curve
    n_struct = struct('brownian',3,'brownian2comp',[4; 5],'anomalous',3);
    n_cell = struct2cell(n_struct);
    n = n_cell{type};
    
    if type == 1
        model_fit(1:len_x) = model_coefs(1).* (1./(1+(x(1:len_x)./model_coefs(3)))) + model_coefs(2);
    elseif type == 2
        model_fit(1:len_x)= model_coefs(1).* (1./(1+(x(1:len_x)./model_coefs(4)))) + model_coefs(2).* (1./(1+(x(1:len_x)./model_coefs(5)))) + model_coefs(3);
    elseif type == 3
        model_fit(1:len_x) = model_coefs(1).* (1./(1+(x(1:len_x)./model_coefs(3)).^model_coefs(4))) + model_coefs(2); 
    end
    
    % R-Square
    residuals = y - model_fit;
    a = (y - model_fit).^2./model_fit;
    a(isinf(a)) = 0;
    rsquare = 1-sum(residuals.^2)/sum(y.^2);
    
    % fit result structure
    fitresult(1,i)=struct('rawdata',[x',y'],'rsquare',rsquare,'model_fit',model_fit);
    
    % characteristic time
    tauD(i)=model_coefs(n(1))*dT; % in in seconds
  
    % diffusion coefficient
    D(i)=(pixelsize.^2)/(4*tauD(i)); %in nm^2/s
    
    % second diffusion coefficient if using 2-component model
    if type == 2
        tauD2(i)=model_coefs(n(2))*dT; % in in seconds
        D2(i)=(pixelsize.^2)/(4*tauD(i));
    end
    
    % alpha map if using anomalous model
    if type == 3
        alpha(i)= model_coefs(4);
    end
    
    % compute total Gpufit time
    fit_time = fit_time + gputime;
end


%% Post fit data manipulation
% reshape fit result 
fitresult2=reshape(fitresult,rowdim,coldim);

%Diffusion coefficient map
Dmap=reshape([D],rowdim,coldim);
% Dmap = abs(Dmap);

% alpha map (anomalous diffusion)
if type == 3    
    alpha_corrected = zeros(1,numel(alpha));
    % remove bad alphas
    for i=1:numel(alpha)
        if fitresult(1,i).rsquare<0.5
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
end

% create tauD map
tauDmap=reshape([tauD],rowdim,coldim);

% remove poor fits
D_corrected = zeros(1,numel(D));
for i=1:numel(D)
    if fitresult(1,i).rsquare<0.5
        D_corrected(i)=0;
    else
        D_corrected(i)=abs(D(i));
    end
end
Dmap_corrected=reshape([D_corrected],rowdim,coldim);

% second diffusion coefficeint map if 2-component brownian model
if type == 2
    D2map=reshape([D2],rowdim,coldim);
    tauD2map=reshape([tauD2],rowdim,coldim);
    
    D2_corrected = zeros(1,numel(D2));
for i=1:numel(D2)
    if fitresult(1,i).rsquare<0.5
        D2_corrected(i)=0;
    else
        D2_corrected(i)=abs(D2(i));
    end
end
D2map_corrected=reshape([D2_corrected],rowdim,coldim);
end

% make map of R^2 values
R2 = zeros(1,numel(fitresult));
for i=1:numel(fitresult)
    R2(i)=fitresult(1,i).rsquare;
end
R2map=reshape(R2,rowdim,coldim);

name = ["Brownian","2-Component Brownian","Anomalous"];name = name(type);

Bin_after = clock;
fprintf('FCS complete, execution time: %6.2f seconds\n',etime(Bin_after,Bin_before));
end


%% %%%%%%%%%%%%%%%%%%%% STEP 3: CombineTempSpat (fcs) %%%%%%%%%%%%%%%%%%%%%%%%%
%% load data
for index = 1:numel(trialnos)
% load(strcat(parameter,num2str(trialnos(index)),'Dmap.mat'))
% load(strcat(parameter,num2str(trialnos(index)),'analyzed.mat'))

Combine_before = clock;
%% Diffusion map
szmap1=size(Dmap,1);
szmap2=size(Dmap,2);

Dmap2log=log10(Dmap); %actually use Dmap, unfiltered R^2; filter in this code

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

%set caxis limits for diffusion map % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!
% cmin=2; % USER INPUT set limits log(D) colorscale
% cmax=6;

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
shift=0.0;%shift left right
scale=0.7;%factor to multiply everything by %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!

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
% cscaleACim=AC_im2;
cscaleACim=filtim;
% cminAC=0; %USER INPUT of SOFI image limits
% cmaxAC=1e9;
for i=1:size(cscaleACim,1) %filter out values above/below, change to limits
    for j=1:size(cscaleACim,2)
        if cscaleACim(i,j)<cminAC
            cscaleACim(i,j)=cminAC;
        elseif cscaleACim(i,j)>cmaxAC
            cscaleACim(i,j)=cmaxAC;
        end
    end
end
norm_ca_AC=cscaleACim./(max(max(cscaleACim))); %normalize

% HSV values
srmap_hsv(1:szmap1,1:szmap2,1)=ones(szmap1,szmap2); %set hue = color
% srmap_hsv(1:szmap1,1:szmap2,2)=AC_G2_im./(max(max(AC_G2_im))); %set saturation ("gray")
% srmap_hsv(1:szmap1,1:szmap2,2)=AC_im2./(max(max(AC_im2))); %set saturation ("gray")
srmap_hsv(1:szmap1,1:szmap2,2)=norm_ca_AC;
srmap_hsv(1:szmap1,1:szmap2,3)=ones(szmap1,szmap2); %set brightness
srmap_hsv=1-hsv2rgb(srmap_hsv); %convert to rgb
srmap_hsv=rgb2gray(srmap_hsv);% convert to gray scale

%% combine D and Super res. for HSV
hsvmap(1:szmap1,1:szmap2,1)=normDmap2log; %set hue = color
hsvmap(1:szmap1,1:szmap2,2)=norm_ca_AC;
hsvmap(1:szmap1,1:szmap2,3)=ones(szmap1,szmap2); %set brightness
hsv2rgbmap=1-hsv2rgb(real(hsvmap)); %convert to rgb

%% computation time
Combine_after = clock;
clc;fprintf('Image fusion complete, execution time: %6.2f seconds\n',etime(Combine_after,Combine_before));
fprintf('\nPlotting...');

end

%% Figures
if plotfigures == 1
%%%%%% blinkAConly %%%%%% 
    % blinkAConly Subplots
    figure % average image
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

%%%%%% BinFitData %%%%%%
    % super resolution image
    h = figure;
    imagesc(AC_im2);
    title('SOFI super res figure')
    
    % BinFitData subplots 
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
    %%
    % alpha map if using anomalous model
    if type == 3
        figure
        imagesc(alphamap);
        caxis([0 max(max((alphamap)))])
        axis image
%         title('\alpha map')
        c=colorbar;
        c.Label.String = '\alpha';
        c.Label.FontSize = 20;
        set(gca,'xtick',[],'ytick',[])
        
    end
%%

    % R-square Map
    h2 = figure;
    imagesc(R2map)
    caxis([0 1])
    axis image
    title('R^2 map')
    colorbar

%%%%%% BinFitData %%%%%%
%     % CombineTempSpat subplots
%     h3 = figure;
%     
    l=subplot(1,2,1); % log(D) map
    figure;
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

    figure;
    imagesc(srmap_hsv);
    axis image
    title('SOFI super-resolution')
    set(gca,'xtick',[],'ytick',[])
    colormap(gray)
    
    figure;
    k=subplot(1,2,1);
    imagesc(hsv2rgbmap) % combined
    sz=get(l,'position');
    axis image
    title('Combined')
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

    % optional single pixel curve fit result figure
    if examplecf == 1
        i=row_index; %row index
        j=column_index; %column index
        figure
        x2=fitresult2(i,j).rawdata(:,1);
        y2=fitresult2(i,j).rawdata(:,2);
        N = max(y2);
        plot(x2,1/N.*y2,'or','LineWidth',2)
        hold on
        plot(x2',1/N.*fitresult2(i,j).model_fit,'--k','LineWidth',2)
        set(gca,'xscale','log')
        xlabel('\tau')
        ylabel('G(\tau)')
        legend('Raw Data','Fit Result');
        title(strcat(name,' Diffusion Curve Fit with Gpufit')); 

        % error bars
        number_parameters = [3; 5; 4];
        [rsq,chisq,J,MSE,ci] = gof_stats(type,...%type
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
            "G(tau) = a * 1/(1 + (tau/tauD)^alpha) + b"]; modeleqn = modeleqn(type);

        clc; fprintf(fname);fprintf('\nPixel (%i,%i)\n',row_index,column_index );
        fprintf(strcat(name,' Fit Model:\n',modeleqn,'\n\n'));
        fprintf('Fit Result Parameters\n');
        
        % print error bars
        if type == 1
          fprintf('a =    %6.2e ± %6.2e\n',converged_parameters(1),ebars(1));
          fprintf('b =    %6.2e ± %6.2e\n',converged_parameters(2),ebars(2));
          fprintf('tauD =     %6.2e ± %6.2e\n',converged_parameters(3),ebars(3));

          fprintf('\n')

          fprintf('D =         %6.3e\n',Dmap_corrected(row_index,column_index ))
          fprintf('log10(D):  %6.3f\n\n',Dmap2log(row_index,column_index ))
          fprintf('R-square:  %6.4f\n',R2map(row_index,column_index ));
        elseif type == 2
          fprintf('a1 =    %6.2e ± %6.2e\n',converged_parameters(1),ebars(1));
          fprintf('a2 =    %6.2e ± %6.2e\n',converged_parameters(2),ebars(2));
          fprintf('b =     %6.2e ± %6.2e\n',converged_parameters(3),ebars(3));
          fprintf('tauD1 = %6.4f ± %6.4e\n',converged_parameters(4),ebars(4));
          fprintf('tauD2 = %6.4f ± %6.4e\n',converged_parameters(5),ebars(5));  

          fprintf('\n')

          fprintf('D1:         %6.3e\n',Dmap_corrected(row_index,column_index ))
          fprintf('D2:         %6.3e\n',D2map_corrected(row_index,column_index ))
          fprintf('log10(D1):  %6.3f\n',Dmap2log(row_index,column_index ))
          fprintf('log10(D2):  %6.3f\n\n',D2map2log(row_index,column_index ))
          fprintf('R-square:  %6.4f\n',R2map(row_index,column_index ));   

        elseif type == 3
          fprintf('a =    %6.2e ± %6.2e\n',converged_parameters(1),ebars(1));
          fprintf('b =    %6.2e ± %6.2e\n',converged_parameters(2),ebars(2));
          fprintf('tauD =     %6.2e ± %6.2e\n',converged_parameters(3),ebars(3));
          fprintf('alpha = %6.4f ± %6.4f\n',converged_parameters(4),ebars(4));

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
    

if manipulatable_data == 1
    if type == 1
         save(strcat(fname,'_analyzed_brownian_manipulatable_',date),'fit_curves','fit_parameters','SOFI','D_map','D_map_corrected','Rsquare_map','fcsSOFI','fcsSOFI_cmap',...
             'normDmap2log','normcoef','filtim','type','szmap1','szmap2');
    elseif type == 2
        D2_map = D2map;
        D2_map_corrected = D2map_corrected;
        save(strcat(fname,'_analyzed_2comp_brown_manipulatable_',date),'fit_curves','fit_parameters','SOFI','D_map','D_map_corrected','D2_map','D2_map_corrected','Rsquare_map','fcsSOFI','fcsSOFI_cmap',...
            'normDmap2log','normcoef','filtim','type','szmap1','szmap2');
    elseif type == 3
        alpha_map = alphamap;
        save(strcat(fname,'_analyzed_anomalous_manipulatable_',date),'fit_curves','fit_parameters','SOFI','D_map','D_map_corrected','alpha_map','Rsquare_map','fcsSOFI','fcsSOFI_cmap',...
            'normDmap2log','normcoef','filtim','type','szmap1','szmap2');
    end
    
else
    if type == 1
         save(strcat(fname,'_analyzed_brownian_',date),'fit_curves','fit_parameters','SOFI','D_map','D_map_corrected','Rsquare_map','fcsSOFI','fcsSOFI_cmap','cmin','cmax','steps','type');
    elseif type == 2
        D2_map = D2map;
        D2_map_corrected = D2map_corrected;
        save(strcat(fname,'_analyzed_2comp_brown_',date),'fit_curves','fit_parameters','SOFI','D_map','D_map_corrected','D2_map','D2_map_corrected','Rsquare_map','fcsSOFI','fcsSOFI_cmap','cmin','cmax','steps','type');
    elseif type == 3
        alpha_map = alphamap;
        save(strcat(fname,'_analyzed_anomalous_',date),'fit_curves','fit_parameters','SOFI','D_map','D_map_corrected','alpha_map','Rsquare_map','fcsSOFI','fcsSOFI_cmap','cmin','cmax','steps','type');
    end

end
end   

%% total computation time
totaltime = etime(clock,before);
fprintf(['\nTotal fcsSOFI execution time:' ' ' num2str(totaltime/60) ' ' 'minutes (' num2str(totaltime) ' ' 'seconds)\n\n']);

% write computation time to text file for remote access
DateString = datestr(datetime);
xdim = num2str((xmax-xmin+1)); ydim = num2str((ymax-ymin+1));
fid = fopen('Execution_Time_March.txt','at');
fprintf(fid, [DateString '\n']);
fprintf(fid, ['Data File:' ' ' fname '.m' ' (' xdim 'x' ydim ' ' 'image)\n']); 
fprintf(fid, ['Total execution time:' ' ' num2str(totaltime/60) ' ' 'minutes (' num2str(totaltime) ' ' 'seconds)\n']);
fprintf(fid, ['Total GPU Only time:' ' ' num2str(fit_time) ' ' 'seconds\n\n']);
fclose(fid);


%% Goodness of fit statistics
function [rsq,chisq,J,MSE,ci] = gof_stats(type,parameters,fitresult,x,y)

%% Residuals
residuals = y - fitresult;
a = (y - fitresult).^2./fitresult;
a(isinf(a)) = 0;

%% Chi-square
chisq = sum(a);

%% R-square
rsq = 1-sum(residuals.^2)/sum(y.^2);

%% Partial Derivatives and Jacobian Matrix
% Brownian
if type == 1
    % partial derivatives
    dGda = parameters(3)./(parameters(3)+x);
    dGdb = ones(size(x));
    dGdtau = (parameters(1).*x)./(parameters(3)+x).^2;
    
    % Jacobian
    J = [dGda dGdb dGdtau];J = reshape(J,[length(x),length(parameters)]);
    
% 2-Componenet Brownian
elseif type == 2
    % partial derivatives
    dGda1 = parameters(4)./(parameters(4)+x);
    dGda2 = parameters(5)./(parameters(5)+x);
    dGdb = ones(size(x));
    dGdtau1 = (parameters(1).*x)./(parameters(4)+x).^2;
    dGdtau2 = (parameters(2).*x)./(parameters(5)+x).^2;
    
    % Jacobian
    J = [dGda1 dGda2 dGdb dGdtau1 dGdtau2];J = reshape(J,[length(x),length(parameters)]);
    
% Anomalous
elseif type == 3
    %partial derivatives
    dGda = 1./((x./parameters(3)).^(parameters(4)+1));
    dGdb = ones(size(x));
    dGdtau = parameters(1)*parameters(4).*(x./parameters(3)).^(parameters(4))./(parameters(3).*((x/parameters(3)).^(parameters(4))).^2);
    dGdalpha = (parameters(1)*(x./parameters(3)).^(parameters(4)).*log(x./parameters(3)))./((x./parameters(3)).^(parameters(4)+1)).^2;

    % Jacobian
    J = [dGda dGdb dGdtau dGdalpha];J = reshape(J,[length(x),length(parameters)]);
    
end
%% Covariance matrix

%Mean squared error
MSE = sum((y - fitresult).^2)/length(y); 

% ci = 1;
%% Confidence intervals
disp(size(parameters))
disp(size(residuals))
disp(size(J))

ci = nlparci(parameters,residuals,'jacobian',J);

end