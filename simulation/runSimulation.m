%{
Script to run the DiffusionSimFunc3D with set settings
%}
%% Simulation Settings
% For more info, look at DiffusionSimFunc3D description

% Brownian = 0; Levy Flight = 1; Two-Component = 2
diffType = 0;

% If Brownian (0):
% [Extra Steps, D (nm^2/s)]
% If Levy Flights (1):
% [Extra Steps, D (nm^2/s), alpha]
% If Two-Component (2):
% [Extra Steps, D1 (nm^2/s), D2 (nm^2/s), P1 (%)]
diffSettings = [500, 1];
% Other Options: Levy - [1000, 1e6, 0.8], Two - [1000, 6e6, 5e5, 0.9]

% Channels = 1, Pores = 2
poreType = 1;

% All in pixels
% If Channels (1):
% [height, width, thickness, seperation]
% If Pores (2):
% [height, width, numPores, poreR, poreSigma]
% poreSettings = [100, 100, 200, 3, 0.5];
poreSettings = [100, 50, 50, 5, 5, 0];

% [zMin (pixels), zMax (pixels), dT (s), pixelSize (nm), nFrames, nParticles, detectRange (nm)]
micSettings = {-10000, 0, 0.002, 0.0467, 2000, 200, [-200, 0]};

% [stdGauss (Pixels), int_part, bg]
psfSettings = [2.7, 150/50*3, 140/50*3];

% Number of Runs
nRuns = 50;

%% Run Simulation

percents = zeros(nRuns, 10);
sigmas = zeros(nRuns, 1);

testSigmas = logspace(-5, 5, nRuns);

fprintf("Running... ");
tic
for run = 1:nRuns
    fprintf("%i ", run);
    % Simulate Code
    [images, truth, poreMap] = DiffusionSimFunc3D(diffType, diffSettings, poreType, poreSettings, micSettings, psfSettings);
    
    % Test It
    [percents(run, :), sigmas(run)] = testSOFI(images, truth, poreMap, 0);

    diffSettings = [diffSettings(1), testSigmas(run)];

end

time = toc;
fprintf("Finished in %i Minutes and %.2f Seconds \n", floor(time / 60), mod(time, 60));


%percentAvg = mean(percents);
sigmaAvg = mean(sigmas);

%{
fprintf('AC2 Accuracy: %f \n', percentAvg(1))
fprintf('AC3 Accuracy: %f \n', percentAvg(2))
fprintf('AC4 Accuracy: %f \n', percentAvg(3))
fprintf('AC2 Decon Accuracy: %f \n', percentAvg(4))
fprintf('AC3 Decon Accuracy: %f \n', percentAvg(5))
fprintf('AC4 Decon Accuracy: %f \n', percentAvg(6))
fprintf('XC2 Accuracy: %f \n', percentAvg(7))
fprintf('XC2 Decon Accuracy: %f \n', percentAvg(8))
fprintf('Average Accuracy: %f \n', percentAvg(9))
fprintf('Average Decon Accuracy: %f \n', percentAvg(10))
fprintf('PSF Estimate std: %f \n', sigmaAvg)
%}

plot(testSigmas, sigmas, 'o', 'MarkerSize', 12)
set(gca, 'XScale', 'log')

