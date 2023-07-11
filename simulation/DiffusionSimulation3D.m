%% LK 120914
% Create a 2D diffusion simulation  within a binary pore map 
% right now only use
%   - Brownian diffusion, single rate
%% SY 20200513
% added Levy flight
%% SY 20200529
% added two-component Brownian
%% SY

close all
clear;

%% User defined parameters
savedata = 0; %yes = 1;no = 0
fdir='C:\Users\User\Desktop\SY\Kisley-lab\Simulation Code\'; %directory to save data
snum='69'; % file number for saved file
createmovie = 0; %yes = 1;no = 0
moviename = 'P'; %save name for .avi file
showMovie = 0; % See movie frames, really long, Yes = 1

% %%% Simulation details %%% %
% image information
% load binary 2D pore map image
%load('C:\Users\User\Desktop\SY\Kisley-lab\Simulation Code\PAM_SEM_close2.mat')
scaler = 100;
poreMap = makeBinaryMap(1, [100, 100, 5, 3, 0.5] .* scaler); % Channels
%poreMap = makeBinaryMap(2, [100, 100, 5, 5, 1] .* scaler ); % Rings
%poreMap = makeBinaryMap(3, [100, 100, 200 / scaler, 3, 0.5] .* scaler); % Pores

% Cords are in (i, j, k), or (y, x, z)
minCords = [1, 1, -100000];
maxCords = [size(poreMap,1), size(poreMap,2), 0];

% "microscope" and sample parameters
dT = 0.002; % Time between frames in s
pixelSize = 47.6 / scaler; % 47.6; % pixel size in nm
nFrames = 2500;      % Number of frames
np = 375;    % Number of Particles
detectRange = round([-200, 0] ./ pixelSize); % Detection range in nm of microscope [low, high]

% PSF information
stdGauss = 2.5; % sigma of simulated Gaussian PSF in pixels
int_part = 150/50*3; % intensity of the PSF; taken from shot noise, Poisson rand
bg = 140/50*3*1.25; % max bg of image; bg/2 is average value (read noise)
sbr=int_part/(bg/2);
SNR = int_part / sqrt((int_part + ((2*bg)^2)/12)); % Signal to Noise (mean / std)

% %%% Diffusion parameters %%% %
% Brownian parameters
type = 0; %Brownian = 0; Levy Flight = 1; Two-Component = 2;
D = 1e6; % D in nm^2/s 
extraSteps = 500; % How many steps are calculated inbetween frames

% Two-Component Brownian parameters
D1 = 6e6;
D2 = 5e5; 
P1 = 0.9; %Percentage of particles w/ D1
P2 = 0.1; %Won't actually be used but here for clarity

% Anomalous parameters
alpha = .8; % For anomalous

% Blinking Settings
blink = 0; % Turn on blinking; 1 = yes, 0 = no
offOnTime = [0.300, 0.030]; % [off time, on time] in s

% NOT IMPLIMENTED Fluorophore parameters
addbleaching = 0; %yes = 1; no = 0
bleachtime = .1; %in seconds
tol = 10; %particle motion tolerance

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

% Iterate through each particle
for particle = 1:np
    
    % Random start position
    ystart = randi([minCords(1), maxCords(1)]);
    xstart = randi([minCords(2), maxCords(2)]);
    zstart = 0;% randi([-500, 0]);
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

        % %% Blinking %% %
        if blink
            intarr(particle, frame) = intarr(particle, frame) * (blinkInfo(particle, 3)-1);
    
            blinkInfo(particle, 4) = blinkInfo(particle, 4) + dT;
            % If off/on time length > current time
            if blinkInfo(particle, blinkInfo(particle, 3)) >= blinkInfo(particle, 4)
                blinkInfo(particle, 4) = 0; % Set Clock to zero
                blinkInfo(particle, blinkInfo(particle, 3)) = exprnd(offOnTime(blinkInfo(particle, 3))); % Set new off/on time
                blinkInfo(particle, 3) = mod(blinkInfo(particle, 3), 2) + 1; % Swap off/on state
            end
        end
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

%% Create Ground Truth %%

truth = zeros(maxCords(1), maxCords(2));

for frame = 1:nFrames
    for particle = 1:np
        truth(round(info(frame, particle, 1)), round(info(frame, particle, 2))) = truth(round(info(frame, particle, 1)), round(info(frame, particle, 2))) + 1;
    end
end

%% PSF Addition %%

% Set image size, frames for movie
movies = [];
maxCords = round(maxCords ./ scaler); % Scale everything back down
info = info ./ scaler;
images = (bg .* rand(maxCords(1), maxCords(2), nFrames)) - bg/2; %create background for movie

intensity = zeros(np, 1);
gauss = cell([np, 1]);
gaussR = round(stdGauss * 4); % Create a guassian to fit the whole range

% Add PSF on top of particle locations (includes 4 std of the guassian)
for frame = 1:nFrames
    for particle = 1:np
        intensity(particle) = poissrnd(intarr(particle, frame), 1, 1); % Shot noise variation in signal
        yloc = round(info(frame, particle, 1)); % Scales down and rounds location 
        xloc = round(info(frame, particle, 2));
        zloc = round(info(frame, particle, 3));
        yShift = info(frame, particle, 1) - yloc; % Center Gaussian PSF at correct subpixel position
        xShift = info(frame, particle, 2) - xloc;
        
        % Create guassian
        gauss{particle} = makeGauss(gaussR, [xShift, yShift], stdGauss, intensity(particle), zloc, detectRange);
        yLowLim = yloc - gaussR;
        yUpLim = yloc + gaussR;
        xLowLim = xloc - gaussR;
        xUpLim = xloc + gaussR;

        % Check if particle is near edge and adjust gaussian to stay in bounds
        if (yloc < minCords(1) + gaussR) % To Close to top
            gauss{particle}(1:gaussR-yloc+1, :) = [];
            yLowLim = minCords(1);
        end

        if (yloc > maxCords(1) - gaussR) % To Close to bottom
            gauss{particle}(2+gaussR+maxCords(1)-yloc:2*gaussR+1, :) = [];
            yUpLim = maxCords(1);
        end

        if (xloc < minCords(2) + gaussR) % To Close to left
            gauss{particle}(:, 1:gaussR-xloc+1) = [];
            xLowLim = minCords(2);
        end

        if (xloc > maxCords(2) - gaussR) % To Close to right
            gauss{particle}(:, 2+gaussR+maxCords(2)-xloc:2*gaussR+1) = [];
            xUpLim = maxCords(2);
        end

        images(yLowLim:yUpLim, xLowLim:xUpLim, frame) = (images(yLowLim:yUpLim, xLowLim:xUpLim, frame) + gauss{particle});

    end
end

fprintf("Finished \n")

if showMovie
    for frame = 1:nFrames
        imagesc(images(:,:,frame))
        colormap(gray)
        movies=[movies getframe];
    end
end

% Save
if savedata==1
    Data=images;
    save(strcat(fdir,'dataset',snum), 'Data', 'coords');
end
if createmovie==1
%     moviename=strcat(snum,'movie.avi');
    writerObj = VideoWriter(strcat(moviename,'.avi'));
    open(writerObj);
    writeVideo(writerObj,movies);
    close(writerObj);
end

images = double(images);

testSOFI(images, truth, poreMap, 1);

%{
Creates a 2D gaussian from a slice of a 3D gaussian.

gaussR: radius of matrix returned, ignores center pixel. Returns 2*radius+1
        square matrix
center: The center from which the gaussian will be computed. Must be in
        form [x, y]
sigma:  The standard deviation of the guassian returned
zLevel: Which z slice of the 3D guassian returned
%}
function gauss = makeGauss(gaussR, center, sigma, max, zLevel, detectRange)
    

    if (zLevel >= detectRange(1) && zLevel <= detectRange(2))
        [x, y] = meshgrid(-gaussR:gaussR);
        
        
        if zLevel < 0
            sigma = sigma * (1 + zLevel / detectRange(1) * 0.75);
        elseif zLevel > 0
            sigma = sigma * (1 + zLevel / detectRange(2) * 0.75);
        end
        
        xPart = ((x - center(1)) ./ sigma) .^ 2;
        yPart = ((y - center(2)) ./ sigma) .^ 2;

        gauss = max .* exp(-(xPart + yPart) ./ 2);
    else
        gauss = zeros(gaussR * 2 + 1);
    end
end
