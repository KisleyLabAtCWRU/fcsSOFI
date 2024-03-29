%% LK 120914
% Create a 2D diffusion simulation  within a binary pore map 
% right now only use
%   - Brownian diffusion, single rate
%% SY 20200513
% added Levy flight
%% SY 20200529
% added two-component Brownian
%% BW 20230808 
% Allow z-level diffusion
%% MA 20231001
% Apply direct convolution using fspecial function

close all
clear;

%% User defined parameters
savedata = 1; %yes = 1;no = 0
% fdir='C:\Users\User\Desktop\SY\Kisley-lab\Simulation Code\'; %directory to save data
snum='69'; % file number for saved file

createmovie =0; %yes = 1;no = 0
create_truth_movie  = 0;%yes = 1;no = 0
pore_type = "Rings";
showMovie = createmovie; % See movie frames, really long, Yes = 1
show_truth_Movie = create_truth_movie;
addpath(genpath(strcat(pwd,"\fcsSOFI_external_functions")))

% %%% Simulation details %%% %
% image information
% load binary 2D pore map image
%load('C:\Users\User\Desktop\SY\Kisley-lab\Simulation Code\PAM_SEM_close2.mat')

%scaler to simulation a structure beyond nanoscale
scaler = 100;

% "microscope" and sample parameters
dT = 0.002; % Time between frames in s
pixelSize = 102 / scaler ; % 47.6; % pixel size in nm
nFrames = 10000;      % Number of frames
np = 20;    % Number of Particles
detectRange = round([-200, 0] ./ pixelSize); % Detection range in nm of microscope [low, high]

% PSF information
stdGauss = 2; % sigma of simulated Gaussian PSF in pixels
int_part = 150/50*3; % intensity of the PSF; taken from shot noise, Poisson rand
bg = 140/50*3*1.25; % max bg of image; bg/2 is average value (read noise)
%bg = 1; %manually input value
sbr= int_part/(bg/2);
SNR = int_part / sqrt((int_part + ((2*bg)^2)/12)); % Signal to Noise (mean / std)

% %%% Diffusion parameters %%% %
% Brownian parameters
type = 0; %Brownian = 0; Levy Flight = 1; Two-Component = 2;
D = 10e6; % D in nm^2/s 
extraSteps = 500; % How many steps are calculated inbetween frames

% Two-Component Brownian parameters
D1 = 6e6;
D2 = 5e5; 
P1 = 0.9; %Percentage of particles w/ D1
P2 = 0.1; %Won't actually be used but here for clarity

% Anomalous parameters
alpha = .8; % For anomalous

% Blinking Settingss
blink = 0; % Turn on blinking; 1 = yes, 0 = no
offOnTime = [0.300, 0.030]; % [off time, on time] in s

% NOT IMPLIMENTED Fluorophore parameters
addbleaching = 0; %yes = 1; no = 0
bleachtime = .1; %in seconds
tol = 10; %particle motion tolerance


%%Save parameter for later use%%
currentDateTime = datestr(now, 'yyyymmdd_HHMMSS');
%select directory for saving
%folder convention: Date/time_poreType_nframes_np_D(\mum^2s^-1)
fdir = strcat('\\129.22.135.181\Test\MaiAntarasen\data\', currentDateTime, '_' + pore_type+ '_nframes' +string(nFrames) + '_np' + string(np) + '_D' + string(D/1e6) + '_t');
if exist(fdir, 'dir') 
    disp(['Folder ''' fdir ''' already exists.']);
else
    mkdir(fdir);
    disp(['Folder ''' fdir ''' created.']);
end

% name and create a pore map
if pore_type == "Channels"
    %setting = [25, 25, 0.25, 4, 1] .* scaler; %current channel
    %setting = [25, 25, 3, 3, 0] .* scaler;
    setting = [50, 50, 10,20, 0] .* scaler;%for SI
    %setting = [25, 25, 3, 3, 5] .* scaler; %JPCB experiment
    % setting = [100, 100, 10, 10, 0] .* scaler;
    %structure_label = 1;
    poreMap = makeBinaryMap(1, setting, fdir);
end

if pore_type == "Rings"
    setting = [15, 15, 1, 8, 0].*scaler;
    %setting = [50, 50, 10, 5, 0] .* scaler;
    poreMap = makeBinaryMap(2, setting, fdir);
end

if pore_type == "Pores"
    %setting = [25, 25, 1 / scaler, 3, 0.5] .* scaler;
    %setting = [100, 100, 50/scaler, 10, 0.5] .* scaler; % ususal parameter\
    setting = [52, 52, 6 / scaler, 10, 3] .* scaler; %small/big pore for SI
    %structure_label = 3;
    poreMap = makeBinaryMap(3, setting, fdir); % Pores

end

%set poreMap Structure
if pore_type == "BeadsinWater"
    %poreMap = ones(25*scaler);
    poreMap = ones(50*scaler); %for SI
end


% Cords are in (i, j, k), or (y, x, z)
% z means in and out focus
% -100000 mean infinitely diffusion in z
minCords = [1, 1, -100000];
maxCords = [size(poreMap,1), size(poreMap,2), 0];

% %%% End user input %%%% %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Iteration to store positions of particles

intarr = int_part*ones(np, nFrames);
info = zeros(nFrames, np, 3);

% (p, 1) = off time length
% (p, 2) = on time length
% (p, 3) = state, 1 = off, 2 = on
% (p, 4) = current time
blinkInfo = zeros(np, 4);
blinkInfo(:, 1) = exprnd(offOnTime(1), [np, 1]);
blinkInfo(:, 2) = exprnd(offOnTime(2), [np, 1]);
blinkInfo(:, 3) = randi(2, [np, 1]);
blinkInfo(:, 4) = rand([np, 1]) * max(offOnTime);

fprintf("Simulating Particle Movement... ")

seed = 42;
rng(seed);
% Iterate through each particle
for particle = 1:np

    % Random start position
    ystart = randi([minCords(1), maxCords(1)]);
    xstart = randi([minCords(2), maxCords(2)]);
    zstart = randi([-500, 0]);
    while poreMap(ystart, xstart) == 0
         ystart = randi([minCords(1), maxCords(1)]);
         xstart = randi([minCords(2), maxCords(2)]);
    end

    % Indexed (frame, particle, (y, x, or z))
    % For array indexing:       (i, j, or k)
    info(1, particle, 1) = ystart;
    info(1, particle, 2) = xstart;
    info(1, particle, 3) = zstart;
    
    % Iterate through each frame
    static_counter = 0;
    particle_off = 0;
    for frame = 2:nFrames

        % %% Distance of movement %% %
        % Brownian diffusion
        if type == 0
            stepSize = (sqrt(2*D*dT/extraSteps)*(randn(extraSteps, 3)))./pixelSize;
                
        % Levy flight
        elseif type == 1
            stepSize = abs(sqrt(2*D*dT)*((1/randn(extraSteps, 3)).^(-1/alpha)))./pixelSize;

        % Two-component brownian
        elseif type == 2
            if particle <= P1*np
                stepSize = abs(sqrt(2*D1*dT)*randn(extraSteps, 3))./pixelSize;
            else
                stepSize = abs(sqrt(2*D2*dT)*randn(extraSteps, 3))./pixelSize;
            end
        end
        
        stepSize = sum(stepSize, 1); % Totals all intermediate step sizes

        % Set next frames starting positions
        info(frame, particle, 1) = info(frame - 1, particle, 1);
        info(frame, particle, 2) = info(frame - 1, particle, 2);
        info(frame, particle, 3) = info(frame - 1, particle, 3);

        % Whether x or y movement moves first
        kOrder = [1, 2, 3];
        if randi([0, 1])
            kOrder = [2, 1, 3];
        end

        % Move the position
        for k = kOrder
            % If moves more than one pixel, must move in increments smaller
            % than stepSize to find final stopping point
            stepIncrement = round(abs(stepSize(k)));

            stepMove = (stepSize(k) / stepIncrement);
            
            % Checks that particle can move
            counter = 1;
            while (counter <= stepIncrement && ...
                    info(frame, particle, k) + stepMove > minCords(k) && ...
                    info(frame, particle, k) + stepMove < maxCords(k) && ...
                    poreMap(round(info(frame, particle, 1)), round(info(frame, particle, 2))))

                info(frame, particle, k) = info(frame, particle, k) + stepMove;
                counter = counter + 1;
            end

            % Pore map check allows for one too many interations
            % Must check and fixe if went one too many moves
            if ~poreMap(round(info(frame, particle, 1)), round(info(frame, particle, 2)))
                info(frame, particle, k) = info(frame, particle, k) - stepMove;
            end
        end

        % % %% Blinking %% %
        % if blink
        %     intarr(particle, frame) = intarr(particle, frame) * (blinkInfo(particle, 3)-1);
    
        %     blinkInfo(particle, 4) = blinkInfo(particle, 4) + dT;
        %     % If off/on time length > current time
        %     if blinkInfo(particle, blinkInfo(particle, 3)) >= blinkInfo(particle, 4)
        %         blinkInfo(particle, 4) = 0; % Set Clock to zero
        %         blinkInfo(particle, blinkInfo(particle, 3)) = exprnd(offOnTime(blinkInfo(particle, 3))); % Set new off/on time
        %         blinkInfo(particle, 3) = mod(blinkInfo(particle, 3), 2) + 1; % Swap off/on state
        %     end
        % end
        % Bleaching not currently implimented with 3D version
        %{
        if addbleaching == 1
            if ((x(frame,particle) >= x(frame-1,particle) - tol)&&(x(frame,particle) <= x(frame-1,particle) + tol))&&...
                    ((y(frame,particle) >= y(frame-1,particle) - tol) && (y(frame,particle) <= y(frame-1,particle) + tol))&&...
                    (intarr(particle,frame-1) == int_part)
                static_counter = static_counter + 1;
                if static_counter >= 100
                    particle_off = 1;
                end
            else
                static_counter = 0;
            end
                     
            if particle_off >= 1 && intarr(particle,frame-1) >= bleachtime/dT
                intarr(particle,frame) = intarr(particle,frame) - particle_off*100;
                particle_off = particle_off + 1;
                %disp(particle_off);
            end
            
            if intarr(particle,frame-1) == 0
                intarr(particle,frame) = 0;
            end
        end
        %}
    end
end
fprintf("Finished \nAdding PSF... ")

%{
%% Create Ground Truth %%
maxCords = round(maxCords ./ scaler); % Scale everything back down
info = info ./ scaler;

%% Truth image generating %%
truth = zeros(maxCords(1), maxCords(2));
truth_image = zeros(maxCords(1), maxCords(2), nFrames);

for frame = 1:nFrames
    for particle = 1:np
        truth(ceil(info(frame, particle, 1)), ceil(info(frame, particle, 2))) = truth(ceil(info(frame, particle, 1)), ceil(info(frame, particle, 2))) + 1;
        truth_image(ceil(info(frame, particle, 1)), ceil(info(frame, particle, 2)), frame) = truth_image(ceil(info(frame, particle, 1)), ceil(info(frame, particle, 2)), frame) + 1;
    end
end
%}
%% Create Ground Truth (this is orignally from Ben's code so there is no scaling before saving truth image)%% 
truth = zeros(maxCords(1), maxCords(2));
for frame = 1:nFrames
    for particle = 1:np
        truth(round(info(frame, particle, 1)), round(info(frame, particle, 2))) = truth(round(info(frame, particle, 1)), round(info(frame, particle, 2))) + 1;
    end
end

maxCords = round(maxCords ./ scaler); % Scale everything back down
info = info ./ scaler;

%%Create ground truth movie%%
truth_movie_name = strcat(fdir, '/ground_truth_movie' ,'.tif');
truth_image_binary = zeros(maxCords(1), maxCords(2), nFrames);
truth_image_int = zeros(maxCords(1), maxCords(2), nFrames);

for frame = 1:nFrames
    for particle = 1:np
        % if info(frame, particle, 1) >= 0 && info(frame, particle, 2) >= 0
        % %truth_image_binary(ceil(info(frame, particle, 1)), ceil(info(frame, particle, 2)), frame) = truth_image_binary(ceil(info(frame, particle, 1)), ceil(info(frame, particle, 2)), frame) + 1;
        %     truth_image_binary(round(info(frame, particle, 1)), round(info(frame, particle, 2)), frame) = truth_image_binary(round(info(frame, particle, 1)), round(info(frame, particle, 2)), frame) + 1;
        %     truth_image_int(round(info(frame, particle, 1)), round(info(frame, particle, 2)), frame) =  poissrnd(intarr(particle, frame), 1, 1)*truth_image_binary(round(info(frame, particle, 1)), round(info(frame, particle, 2)), frame);
        % end
        %struth_image_binary(ceil(info(frame, particle, 1)), ceil(info(frame, particle, 2)), frame) = truth_image_binary(ceil(info(frame, particle, 1)), ceil(info(frame, particle, 2)), frame) + 1;
      
        if info(frame, particle, 3) >= detectRange(1) && info(frame, particle, 3) <= detectRange(2)
            truth_image_binary(ceil(info(frame, particle, 1)), ceil(info(frame, particle, 2)), frame) = truth_image_binary(ceil(info(frame, particle, 1)), ceil(info(frame, particle, 2)), frame) + 1;
            truth_image_int(ceil(info(frame, particle, 1)), ceil(info(frame, particle, 2)), frame) = poissrnd(intarr(particle, frame), 1, 1)*truth_image_binary(ceil(info(frame, particle, 1)), ceil(info(frame, particle, 2)), frame);
        end
    end
end

movies = [];
if show_truth_Movie
    fprintf('Generating ground truth image array! \n');
    for frame = 1:nFrames
        imagesc(truth_image_binary(:,:,frame))
        colormap(gray)
        movies=[movies getframe];
    end
end

if create_truth_movie
    %     moviename=strcat(snum,'movie.avi');
        writerObj = VideoWriter(strcat(truth_movie_name,'.avi'));
        open(writerObj);
        writeVideo(writerObj,movies);
        close(writerObj);
end
fprintf("\n");
fprintf('Done Generating ground truth image array! \n');

movies = [];%clear this for movie in the convolved data
guass_kernel = int_part*fspecial('gaussian', ceil(stdGauss+15), stdGauss);
h = guass_kernel/max(guass_kernel,[],'all'); %gaussian psf not really need this
% figure;
% imagesc(h);
% colormap('gray');

figure;
images = truth_image_int;
% convolve_images_bg = zeros(size(images));
bg_matrix = (bg .* rand(maxCords(1), maxCords(2), nFrames)) - bg/2;
%bg_matrix = (bg .* randn(maxCords(1), maxCords(2), nFrames)) - bg/2;
dark_noise = 1.8;
%bg_matrix = bg.*normrnd(0,dark_noise,size(images));

for frame = 1:nFrames
        convolve_images_bg(:,:,frame) = conv2(images(:,:,frame), h, 'same');
end

convolve_images_bg = poissrnd(convolve_images_bg) + bg_matrix;
%convolve_images_bg = poissrnd(convolve_images_bg) + bg_matrix;

fprintf("Finished adding convolution and background noise \n")

movie_name = strcat(fdir, '/convolved_movie','.tif') ; 
if showMovie
    for frame = 1:nFrames
        imagesc(convolve_images_bg(:,:,frame))
        colormap(gray)
        movies=[movies getframe];
    end
end

% Save data in mat_dataset for later use of fcsSOFI analysis
dataset_name = strcat(fdir, '/mat_dataset');
if savedata
    Data=convolve_images_bg;
    % save(strcat(fdir,'dataset',snum), 'Data', 'coords');
    save(dataset_name, 'Data', '-v7.3');
end

if createmovie
    %     moviename=strcat(snum,'movie.avi');
        writerObj = VideoWriter(strcat(movie_name,'.avi'));
        open(writerObj);
        writeVideo(writerObj,movies);
        close(writerObj);
end

parameters = ["PoreTypes";"Type";"D";"pixelSize";"nFrames";"np";"dT";"stdGauss";"int_part";"bg";"scaler"];
value_parameters = [pore_type;type;D;pixelSize;nFrames;np;dT;stdGauss;int_part;bg;scaler];
table_parameter = table(parameters,value_parameters);
writetable(table_parameter, strcat(fdir,'/parameter-table.txt'),'Delimiter',' ');

testSOFI(double(convolve_images_bg), truth, poreMap, 1, fdir); %using simulated data to see the resolved image using AC2 and XC2 quickly
