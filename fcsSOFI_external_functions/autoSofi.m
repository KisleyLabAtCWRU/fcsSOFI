

function [AC2, AC3, AC4, AG4, AC2Corrected, AC3Corrected, AG4Corrected] = autoSofi(data)
    

    %% Variable Set Up
    sz = size(data);
    height = sz(1);
    width = sz(2);
    nframes = sz(3);
    tauMaster = [0, 2, 4, 6]; % Chosen Time Shifts
    data = double(data);
    average = mean(data, 3);
    delta = data - average;
    i = 1:height;
    j = 1:width;
    t = 1:(nframes-max(tauMaster));

    %% Auto Correlation Calculations
    % Second Order
    tauComb = nchoosek(tauMaster, 2);
    AC2All = zeros(height, width, size(tauComb, 1));
    for n = 1:size(tauComb, 1)
        tau = tauComb(n, :);
        AC2All(:, :, n) = abs(mean(delta(i,j,t+tau(1)) .* delta(i,j,t+tau(2)), 3));
    end
    AC2 = AC2All(:, :, 1);
    AC2CorrectedAll = AC2All + average.^2;
    AC2Corrected = AC2CorrectedAll(:, :, 1);

    % Third Order
    tauComb = nchoosek(tauMaster, 3);
    AC3All = zeros(height, width, size(tauComb, 1));
    for n = 1:size(tauComb, 1)
        tau = tauComb(n, :);
        AC3All(:, :, n) = abs(mean(delta(i,j,t+tau(1)) .* delta(i,j,t+tau(2)) .* delta(i,j,t+tau(3)), 3));
    end
    AC3 = AC3All(:, :, 1);
    AC3CorrectedAll = AC3All + (((AC2CorrectedAll(:, :, 1)+AC2CorrectedAll(:, :, 2)+AC2CorrectedAll(:, :, 4)).*average) - (2*average.^3));
    AC3Corrected = AC3CorrectedAll(:, :, 3);

    % Fourth Order
    AG4 = abs(mean(delta(i,j,t) .* delta(i,j,t+tau(1)) .* delta(i,j,t+tau(2)) .* delta(i,j,t+tau(3)), 3));
    AG4_01 = abs(mean(delta(i,j,t) .* delta(i,j,t+tau(1)), 3));
    AG4_02 = abs(mean(delta(i,j,t) .* delta(i,j,t+tau(2)), 3));
    AG4_03 = abs(mean(delta(i,j,t) .* delta(i,j,t+tau(3)), 3));
    AG4_12 = abs(mean(delta(i,j,t+tau(1)) .* delta(i,j,t+tau(2)), 3));
    AG4_13 = abs(mean(delta(i,j,t+tau(1)) .* delta(i,j,t+tau(3)), 3));
    AG4_23 = abs(mean(delta(i,j,t+tau(2)) .* delta(i,j,t+tau(3)), 3));

    AC4 = abs(AG4-AG4_01.*AG4_23-AG4_02.*AG4_13-AG4_03.*AG4_12);

    AG4Corrected = AG4 + (sum(AC3CorrectedAll, 3).*average) - (sum(AC2CorrectedAll, 3).*average.^2) + (3*average.^4);
    

    %{
    %% Normalize
    AC2 = rescale(AC2);
    AC3 = rescale(AC3);
    AC4 = rescale(AC4);
    %}
    % Old Sofi Implimentation
    %{

    %% Variable Set Up
    sz = size(data);
    height = sz(1);
    width = sz(2);

    % For xcorr, the corrlated values (values from different frames) must
    % be in the first index
    data = permute(data, [3, 1, 2]);

    % If no worker pool, start one
    p = gcp;
    if isempty(p)
        parpool('Processes');
    end


    %% Auto Correlation Calculations
    
    sofi = zeros(height, width);
    sofi = num2cell(sofi);
    % Must have a linear loop for parfor
    parfor k = 1:(height * width)
        [i, j] = ind2sub([height width], k) 
        [autoCorr, lags] = xcorr(data(:, i, j), data(:, i, j));

        %min_lags staring at zero
        min_lags = (length(lags) - 1) / 2 + 1;
        max_lags = length(lags);   

        % 2nd order correlation
        autoCorr = autoCorr(min_lags:max_lags);
        
        sofi{k} = autoCorr(2)

    end
    sofi = cell2mat(sofi);
    disp('Sofi Auto Corr 100% Done')

    %}
end
