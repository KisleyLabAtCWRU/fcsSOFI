function [LocatStore, Bkg] = LRG_SuperRes_Particle_Identify_org(im, e)
%% identify particles using pre-processed image data
% input: im: a single frame
% input: e.SNR_enhance = true || false, true if want to use SNR_booster
% function to increase the SNR ratio before analysis
% input: e.local_thd = true || false, true if want to use local background
% and threshold
% input: e.wide2 = 1, 1.5, 2, 2.4, 2.9, 3, ... 5, cut off distance to define a local maximum,
% defalst is 3 for wide field.
% input: e.Gauss_width = 1 to 3, Gaussian width of the PSF. default is 2.
% input: e.fitting = 'rc' || 'gs' || 'el', fitting funtion to be used, 
% radia symmetry:'rc', Gauss fitting: 'gs', Euler fitting: 'el'. default is
% 'rc'
% input: e.test = true || false, true if using test mode to generate a
% figure after particle identification. default is false.
%% initial conditions & input parameters
im = double(im);
try % use try statement to avoid input errors. If any input is missing, using default value
    if e.SNR_enhance == true
        im = SNR_booster(im);
    end
catch ME
    im = SNR_booster(im);
end
try
    local_thd = e.local_thd;% using local threshold (true) or not (false)
    wide2 = e.wide2;% cut off distance to define a local maximum
    Gauss_width = e.Gauss_width;% fitting regions will be 4*Gauss_width+1
    fitting = e.fitting;% specify the fitting algorithm to use
catch ME
    local_thd = false;
    wide2 = 3;
    Gauss_width = 2;
    fitting = 'rc';% radia symmetry:'rc', Gauss fitting: 'gs', Euler fitting: 'el'
end
try
    test = e.test;
catch ME
    test = false;
end
wide = floor(wide2);
n = e.sigma; % how many std to add up as a threshold
% FWHM = 2.35*Gauss_width
%% calculate threshold map
w = 10;% the local region to calculate the local background and threshold
% usually 50X50 and shift by 25 is a good choice for a 512X512 image
[v h] = size(im);
count = zeros(v, h);% count store the times each pixel has contributed in local
% background calculation
bg = count; % record the local background
sd = count; % record the local standard deviation
if local_thd == true
for i = 1 : ceil(v / w * 2) - 1
    for j = 1 : ceil(h / w * 2) - 1
        % 1, select the local region
        % 2, sort the pixels based on their intensities
        % 3, find the 50% and 75% point as discussed in the paper
        % 4, calculate local background(bg), standard deviation(sd) and
        % count
        im_local = im(1+w/2*(i-1):min(v,w/2*(i+1)), 1+w/2*(j-1):min(h,w/2*(j+1)));
        im_local = sort(im_local(:));
        n_loc = numel(im_local);
        bg(1+w/2*(i-1):min(v,w/2*(i+1)), 1+w/2*(j-1):min(h,w/2*(j+1))) =...
            bg(1+w/2*(i-1):min(v,w/2*(i+1)), 1+w/2*(j-1):min(h,w/2*(j+1))) +...
            im_local(round(n_loc/2));
        sd(1+w/2*(i-1):min(v,w/2*(i+1)), 1+w/2*(j-1):min(h,w/2*(j+1))) =...
            sd(1+w/2*(i-1):min(v,w/2*(i+1)), 1+w/2*(j-1):min(h,w/2*(j+1))) +...
            im_local(round(n_loc*0.5)) - im_local(round(n_loc*0.18));%sd = 0.82 to 0.5 of cumulative distribution
        count(1+w/2*(i-1):min(v,w/2*(i+1)), 1+w/2*(j-1):min(h,w/2*(j+1))) =...
            count(1+w/2*(i-1):min(v,w/2*(i+1)), 1+w/2*(j-1):min(h,w/2*(j+1))) + 1;
    end % for j
end % for i
bg = bg ./ count;
sd = sd ./ count;

%*******************************************
%******* Edited bg to bg*eBT -RMN 2/1/22 *******
%*******************************************
eBT = e.BackgroundThreshold ;
thd_map = bg*eBT + n * sd;% determine the local threshol
Bkg = thd_map;
%{
figure;
image(thd_map);
title('Threshhold after subtraction');

figure;
image(Bkg);
title('Threshhold beofore subtraction');
1 == 1;
%}
else
    % calculate the background and standard deviation once for the whole
    % image
    im_local = sort(im(:));
    n_loc = numel(im_local);
    bg(1:v, 1:h) = im_local(round(n_loc/2));
    sd(1:v, 1:h) = im_local(round(n_loc/2)) - im_local(round(n_loc/4));
    thd_map = bg*eBT + n * sd;
end % if local_thd

%% calculate max_map
% the idea of this part also originates from Arnauld Serge's work.
center = im(1+wide:v-wide, 1+wide:h-wide);
max_map = zeros(v, h);
pos_check = max_map;
if numel(thd_map) > 1
    thd_map = thd_map(1+wide:v-wide, 1+wide:h-wide);
end
max_map(1+wide:v-wide, 1+wide:h-wide) = center > thd_map;% each center is compared
% with its local threshold.
% the two loops below are intended to select the local maximums that meet two
% conditions: 
% 1, the selected neighbors are not brighter than the center;
% 2, the selected neighbors are also brighter than the local threshold.
for i = -wide : wide
    for j = -wide : wide
        if i^2 + j^2 <= wide2^2
            pos_check(1+wide:v-wide, 1+wide:h-wide) = ...
                im(1+wide+i:v-wide+i, 1+wide+j:h-wide+j) <= center & ...
                im(1+wide+i:v-wide+i, 1+wide+j:h-wide+j) > thd_map;
            max_map = max_map .* pos_check;
            pos_check = zeros(v, h);
        end
    end
end

max_map = max_map .* (im - bg);
%
%% calculate the subpixel position of each particle
% We use Parthasarathy's radial symmetry method here because it is fast and
% accurate. Detail see nmeth.2071
match_r = 2 * Gauss_width; % the size of fitting region is match_r*2+1
if isnan(sum(max_map(:))) || isinf(sum(max_map(:)))
    LocatStore = struct([]);
    return
end
LocatStore.PSFDataID = zeros(sum(max_map(:)~=0),2);
LocatStore.PSFData = zeros(sum(max_map(:)~=0),6);
LocatStore.PSFfinal = zeros(sum(max_map(:)~=0),6);
k = 1;
sig_thd = 0.204*(2*match_r+1)*0.9;% threshold basedon the width of noise. 
% Width of a real particle must smaller than 90% of the width of noise.
% the magic number 0.204 is the average value per pixel averaged by 1000
% trails
while sum(max_map(:)~=0) > 0
    [I, q] = max(max_map(:));
    q = q(1);
    I = I(1);
    bk = bg(q);
    max_map(q) = 0;
    row = ceil(q / v);
    row1 = max(row - match_r, 1);
    row2 = min(row + match_r, h);
    clm = q - (row - 1) * v;
    clm1 = max(clm - match_r, 1);
    clm2 = min(clm + match_r, v);
    LocatStore.PSFDataID(k, 1:2) = [clm, row];
    [xc yc sigma] = radialcenter(im(clm1:clm2,row1:row2));
    if fitting == 'gs'
        [I, xc, yc, ~, bk] = gaussfit2Dnonlin(im(clm1:clm2,row1:row2),[],[bk,row-row1+1,clm-clm1+1,match_r,I]);
    elseif fitting == 'el'
        [xc, yc] = Euler_fit(im(clm1:clm2,row1:row2), row-row1+1, clm-clm1+1);
    end
    LocatStore.PSFData(k, 1:6) = [yc+clm1-1, xc+row1-1, sigma, sigma, I, bk];
    %% this is the filter to exclude false positive
    if sigma < sig_thd
        LocatStore.PSFfinal(k, 1:6) = [yc+clm1-1, xc+row1-1, sigma, sigma, I, bk];
    end
    k = k+1;
end
LocatStore.PSFfilteredraw=LocatStore.PSFfinal;
indx=find(LocatStore.PSFfinal(:,1)~=0);
LocatStore.PSFfinal=LocatStore.PSFfinal(indx,:);
if strcmp(test,'true')
    figure
    imagesc(im)
    hold on
    indx = find(LocatStore.PSFfinal(:,1)~=0);
    plot(LocatStore.PSFfinal(indx,2),LocatStore.PSFfinal(indx,1),'or','MarkerSize',5)
end
%}