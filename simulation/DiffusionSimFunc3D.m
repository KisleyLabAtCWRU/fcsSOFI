%% LK 120914
% Create a 2D diffusion simulation  within a binary pore map
% right now only use
%   - Brownian diffusion, single rate
%% SY 20200513
% added Levy flight
%% SY 20200529
% added two-component Brownian
%% SY

%{
Simulates diffusion on a random binary pore map


Inputes:
diffType - 
    Brownian = 0; Levy Flight = 1; Two-Component = 2
diffSettings - 
    If Brownian (0):
    [Extra Steps, D (nm^2/s)]
    Extra Steps - Diffusion steps calculated between frames
    D (nm^2/s) - Diffusion coefficient

    If Levy Flights (1):
    [Extra Steps, D (nm^2/s), alpha]
    Alpha - For anomalous
    
    If Two-Component (2):
    [Extra Steps, D1 (nm^2/s), D2 (nm^2/s), P1 (%)]
    D1 (nm^2/s) - Diffusion coefficient one
    D2 (nm^2/s) - Diffusion coefficient two
    P1 (%) - Percent of particles with first diffusion coefficient

(See more details in makeBinaryMap)
poreType -
    Channels = 1, Pores = 2
poreSettings - ALL PORE SETTINGS IN PIXELS
    If Channels (1):
    [height, width, thickness, seperation]
    height - Overall pixel height of map
    width - Overall pixel width of map
    thickness - How many pixels thick a channel is
    seperation - How mant pixels seperate each channel

    If Pores (2):
    [height, width, numPores, poreR, poreSigma]
    height: Overall pixel height of map
    width: Overall pixel width of map
    numPores: The number of pores to be generated in the map
    poreR: The mean radius of a pore
    poreSigma: The standard deviation used when generating a pores actual radius

micSettings - (Must be char array since detectRange is array)
    {zMin (pixels), zMax (pixels), dT (s), pixelSize (nm), nFrames, nParticles, detectRange (nm)}
    zMin (pixels) - Minimum z value a particle can diffuse to
    zMax (pixels) - Maximum z value a particle can diffuse to
    dT (s) - Time inbetween frames in seconds
    pixelSize (nm) - Size of pixels in nm
    nFrames - Number of frames generated
    nParticles - Number of particles simulated
    detectRange (nm) - [low, high], Detection range in nm of microscope (depth)

psfSettings - 
    [stdGauss (Pixels), int_part, bg]
    stdGauss (Pixels) - Standard deviation of Point Spread Function in pixels
    int_part - Intensity of particle PSF, used as lambda for a poisson
                random number (shot noise)
    bg - max background of image, bg/2 is average value (read noise)
%}
function [images, truth, poreMap] = DiffusionSimFunc3D(diffType, diffSettings, poreType, poreSettings, micSettings, psfSettings)

%% User defined parameter
% See movie frames, takes really long, but good for seeing sim settings
showMovie = 1; % Yes = 1

%% Set Up Variables

% Diffusion Info
type = diffType;
extraSteps = diffSettings(1);

switch type
    case 0
        D = diffSettings(2);
    case 1
        D = diffSettings(2);
        alpha = diffSettings(3);
    case 2
        D1 = diffSettings(2);
        D2 = diffSettings(3);
        P1 = diffSettings(4);
end

% Pore Map Info
poreMap = makeBinaryMap(poreType, poreSettings);

% Microscope Info
% Cords are in (i, j, k), or (y, x, z)
minCords = [1, 1, micSettings{1}];
maxCords = [size(poreMap, 1), size(poreMap, 2), micSettings{2}];
dT = micSettings{3};
pixelSize = micSettings{4};
nFrames = micSettings{5};
nParticles = micSettings{6};
detectRange = round(micSettings{7} ./ pixelSize);

% PSF Info
stdGauss = psfSettings(1);
int_part = psfSettings(2);
bg = psfSettings(3);
% sbr = int_part/(bg/2);

% NOT IMPLIMENTED Fluorophore parameters
%{
addbleaching = 0; %yes = 1; no = 0
bleachtime = .1; %in seconds
tol = 10; %particle motion tolerance
%}

% %%% End user input %%%% %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Iteration to store positions of particles

intarr = int_part*ones(nParticles, nFrames);
info = zeros(nFrames, nParticles, 3);

% fprintf("Simulating Particle Movement... ")

% Iterate through each particle
for particle = 1:nParticles
    
    % Random start position
    ystart = randi([minCords(1), maxCords(1)]);
    xstart = randi([minCords(2), maxCords(2)]);
    zstart = randi([-100, 0]);
    while poreMap(ystart, xstart) == 0
         ystart=randi([minCords(1), maxCords(1)]);
         xstart=randi([minCords(2), maxCords(2)]);
    end

    % Indexed (frame, particle, (y, x, or z))
    % For array indexing:       (i, j, or k)
    info(1, particle, 1) = ystart;
    info(1, particle, 2) = xstart;
    info(1, particle, 3) = zstart;
    
    % Iterate through each frame
    % static_counter = 0; Used for Bleaching
    % particle_off = 0; Used for Bleaching
    for frame = 2:nFrames

        % %% Distance of movement %% %
        % Brownian diffusion
        if type == 0
            stepSize = (sqrt(2*D*dT/extraSteps)*(randn(extraSteps, 3)))./pixelSize;
                
        % Levy flight
        elseif type == 1
            stepSize = abs(sqrt(2*D*dT)*((1/randn(1, 3)).^(-1/alpha)))./pixelSize;

        % Two-component brownian
        elseif type == 2
            if particle <= P1*nParticles
                stepSize = abs(sqrt(2*D1*dT)*randn(1, 3))./pixelSize;
            else
                stepSize = abs(sqrt(2*D2*dT)*randn(1, 3))./pixelSize;
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
            stepIncrement = 1;
            if stepSize(k) >= 1
                stepIncrement = 100;
            end
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

% fprintf("Finished \nAdding PSF... ")

%% PSF Addition %%

% Set image size, frames for movie
images = (bg .* rand(maxCords(1), maxCords(2), nFrames)) - bg/2; %create background for movie

intensity = zeros(nParticles, 1);
gauss = cell([nParticles, 1]);
gaussR = 10;

% Add PSF on top of particle locations (11 x 11 pixels in size)
for frame = 1:nFrames
    for particle = 1:nParticles
        intensity(particle) = poissrnd(intarr(particle, frame), 1, 1); % Shot noise variation in signal
        yloc = round(info(frame, particle, 1)); % Create integer values for emiter location
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
        if (yloc < minCords(1) + gaussR) % Close to top
            gauss{particle}(1:gaussR-yloc+1, :) = [];
            yLowLim = minCords(1);
        end

        if (yloc > maxCords(1) - gaussR) % Close to bottom
            gauss{particle}(2+gaussR+maxCords(1)-yloc:2*gaussR+1, :) = [];
            yUpLim = maxCords(1);
        end

        if (xloc < minCords(2) + gaussR) % Close to left
            gauss{particle}(:, 1:gaussR-xloc+1) = [];
            xLowLim = minCords(2);
        end

        if (xloc > maxCords(2) - gaussR) % Close to right
            gauss{particle}(:, 2+gaussR+maxCords(2)-xloc:2*gaussR+1) = [];
            xUpLim = maxCords(2);
        end

        images(yLowLim:yUpLim, xLowLim:xUpLim, frame) = (images(yLowLim:yUpLim, xLowLim:xUpLim, frame) + gauss{particle});

    end
end

% fprintf("Finished \n")

truth = zeros(maxCords(1), maxCords(2));

% Makes Ground Truth
for frame = 1:nFrames
    for particle = 1:nParticles
        truth(round(info(frame, particle, 1)), round(info(frame, particle, 2))) = truth(round(info(frame, particle, 1)), round(info(frame, particle, 2))) + 1;
    end
end


if showMovie
    movies = [];
    for frame = 1:nFrames
        imagesc(images(:,:,frame))
        colormap(gray)
        movies=[movies getframe];
    end
end

images = double(images);



end

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
