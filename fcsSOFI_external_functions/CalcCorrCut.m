%% CalcCorr - LK111114
% Function that calculates autocorrelation 2nd order only
% Input:
%   - innerpts -
%   - DataVector - 1D vector of the image to be analyzed

function [ACXC_all] = CalcCorrCut(innerpts, DataVector)

    % i range accomodates time offset
    for i = 1:length(innerpts) - 2      
        %taking only indices we care about
        index = innerpts(i);
        %Fluorescence intensity calculations        
        %fluorescence fluctuation
        F = DataVector(:, index);
        dF = F - mean(F);

        %% Perform AC calculations
        %autocorrelating and cross-correlating with xcorr
        [AC, lags] = xcorr(dF, dF); 

        %min_lags staring at zero
        min_lags = (length(lags) - 1) / 2 + 1;
        max_lags = length(lags);   

        % 2nd order correlation
        AC = AC(min_lags:max_lags);

        %LK-add storing the AC and xcorr curves for each pixel, i
        ACXC_all(1, i) = struct('Order2', AC');
    end
end
