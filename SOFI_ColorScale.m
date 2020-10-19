close all
clear all
%Create script to try to edit the image fusion of the D_map and SOFI
%images more easily
%20200410LK
%20200415LK

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%% USER INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
fname='dataset77_analyzed_brownian_08-11-2020_09-55'; %string ending in .mat of your file of data in the current directory
%%% DIFFUSION CONSTANTS  - Set range of D plotted in fcsSOFI diffusion map
% select these STRATEGICALLY based on your expect diffusion coefficients
% defaults are from  fcsSOFI of 100 nm beads in 1% agarose from original
% paper but these will change based on YOUR data
cmin=2; %min log(D) plotted; default 3 
cmax=6; %max log(D); default 6

% HUE - set values for color scale in the diffusion and fcsSOFI images
% these are changed by trial and error
% Mostly change variables shift and scale 
% the hue will also change when you change cmin and cmax
shift=0;%shift left right; default 0
scale=.7;%factor to multiply everything by; default 0.7 
normcoef=100; %default 10
steps=25; %number of steps between min/max; default 25

% SATURATION - set the range of the SOFI and fcsSOFI image
satmax=.2; %set to 1 if want entire range of data; decrease <1 to increase saturation; default 1
satmin=0; %set to 0 if want entire range of data; increase >0 to change the range of saturation; default 0

%% %%%%%%%%%%%%%%%%%%%%%%%%% END USER INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%% %%
load(fname); %load data
% switch variable names from "saved" data in fcsSOFI_script.m back to what
% is used in the original script; changed since script was copied/pasted
%switch the names back


Dmap_corrected = D_map_corrected;
srmap_hsv=SOFI;
hsv2rgbmap=fcsSOFI;


% D_map = Dmap;
% D_map_corrected = Dmap_corrected;
% Rsquare_map = R2map;
% fcsSOFI = hsv2rgbmap;
% fcsSOFI_cmap = rgb_cmap;

Dmap2log=log10(D_map_corrected);
normDmap2log=Dmap2log./normcoef;
%copy/paste from  fcsSOFI_script before saving variables.
%% Create scaling/stretching/shifting factor to create colormap
%changing shift, scale will change the colormap colors, range
maxvalue=cmax/normcoef; %this is the max value on the colormap

for i=1:size(normDmap2log,1)
    for j=1:size(normDmap2log,2)
        if normDmap2log(i,j)==0
            normDmap2log(i,j)=0;
        else
            normDmap2log(i,j)=(normDmap2log(i,j)-shift).*scale;
        end
    end
end
normDmap2log(normDmap2log > 1) = 0;

% %create colormap from the cmin/cmax
ncmin=((cmin/normcoef)-shift).*scale;
ncmax=((cmax/normcoef)-shift).*scale;

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
 %for i=1:size(dmap_hsv,1) %filter out background rgb(D)=0, then make pixel black
 %    for j=1:size(dmap_hsv,2)
 %        if dmap_hsv(i,j,1)==0
 %            dmap_hsv(i,j,:)=0;
 %        end
 %    end
 %end

%% set the limits of the SOFI image
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


%% combine SOFI and fcs
hsvmap(1:szmap1,1:szmap2,1)=normDmap2log; %set hue = color
hsvmap(1:szmap1,1:szmap2,2)=srmap_hsv; %set saturation
hsvmap(1:szmap1,1:szmap2,3)=ones(szmap1,szmap2); %set brightness
hsv2rgbmap=1-hsv2rgb(real(hsvmap)); %convert to rgb

%% plot figures 
f=figure;

f.Position=[364 469 1160 412];
subplot(1,4,1) % log(D) map
imagesc(dmap_hsv);
axis image
title('FCS: log(D)')
set(gca,'xtick',[],'ytick',[])


subplot(1,4,2)
imagesc(srmap_hsv); %SOFI map
axis image
title('SOFI super-resolution')
set(gca,'xtick',[],'ytick',[])
colormap(gray)

l=subplot(1,4,3);
imagesc(hsv2rgbmap) % combined
sz=get(l,'position');
axis image
title('Combined')
set(gca,'xtick',[],'ytick',[])

c=subplot(1,4,4); %colormap
imagesc(rgb_cmap)
% a bunch of code to get the colorbar the right size and labels in correct
% position
sz2=get(l,'position');
set(gca,'ylim',[1 steps+1],'ytick',[1 steps+1],'yticklabel',[num2str(cmin); num2str(cmax)],...
    'xtick',[],'Position',[0.7484    0.4    0.02   0.3])
ylabel('log(D) (nm^2/s)','Rotation',270)
c.YAxisLocation='right';
ylh = get(gca,'ylabel');
gyl = get(ylh);                                                         
ylp = get(ylh, 'Position');
ylp = get(ylh, 'Position');
ext=get(ylh,'Extent');
set(ylh, 'Rotation',270, 'Position',ylp+[ext(3)+1 0 0])
axis xy

%change font, size of labels
set(findall(gcf,'-property','FontSize'),'FontSize',14,'FontName','Malgun Gothic')
if 0
    f=figure;
    f.Position=[364 469 1160 412];
    subplot(1,4,1) % log(D) map
    
    Dmap2log=5;
    normDmap2log=Dmap2log./normcoef;
    for i=1:size(normDmap2log,1)
        for j=1:size(normDmap2log,2)
            if normDmap2log(i,j)==0
                normDmap2log(i,j)=0;
            else
                normDmap2log(i,j)=(normDmap2log(i,j)-shift).*scale;
            end
        end
    end
    dmap_hsv(1:szmap1,1:szmap2,1)=normDmap2log; %set hue = color
    dmap_hsv(1:szmap1,1:szmap2,2)=ones(szmap1,szmap2); %set saturation ("gray")
    dmap_hsv(1:szmap1,1:szmap2,3)=ones(szmap1,szmap2); %set brightness
    dmap_hsv=1-hsv2rgb(real(dmap_hsv)); %convert to rgb
    
    imagesc(dmap_hsv);
    axis image
    title('FCS: log(D)')
    set(gca,'xtick',[],'ytick',[])

    fname='PAM_SEM_CEEC';
    load(fname);
    SOFI = binarizedImage(1:172,1:257);
    subplot(1,4,2)
    imagesc(SOFI); %SOFI map
    axis image
    title('SOFI super-resolution')
    set(gca,'xtick',[],'ytick',[])
    colormap(gray)

    hsvmap(1:szmap1,1:szmap2,1)=normDmap2log; %set hue = color
    hsvmap(1:szmap1,1:szmap2,2)=SOFI; %set saturation
    hsvmap(1:szmap1,1:szmap2,3)=ones(szmap1,szmap2); %set brightness
    hsv2rgbmap=1-hsv2rgb(real(hsvmap)); %convert to rgb
    
    l=subplot(1,4,3);
    imagesc(hsv2rgbmap) % combined
    sz=get(l,'position');
    axis image
    title('Combined')
    set(gca,'xtick',[],'ytick',[])

    c=subplot(1,4,4); %colormap
    imagesc(rgb_cmap)
    % a bunch of code to get the colorbar the right size and labels in correct
    % position
    sz2=get(l,'position');
    set(gca,'ylim',[1 steps+1],'ytick',[1 steps+1],'yticklabel',[num2str(cmin); num2str(cmax)],...
        'xtick',[],'Position',[0.7484    0.4    0.02   0.3])
    ylabel('log(D) (nm^2/s)','Rotation',270)
    c.YAxisLocation='right';
    ylh = get(gca,'ylabel');
    gyl = get(ylh);                                                         
    ylp = get(ylh, 'Position');
    ylp = get(ylh, 'Position');
    ext=get(ylh,'Extent');
    set(ylh, 'Rotation',270, 'Position',ylp+[ext(3)+1 0 0])
    axis xy

    %change font, size of labels
    set(findall(gcf,'-property','FontSize'),'FontSize',14,'FontName','Malgun Gothic')
end

if 1
    figure
    imagesc(hsv2rgbmap);
end