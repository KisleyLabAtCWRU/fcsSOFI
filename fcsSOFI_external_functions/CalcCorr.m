%% CalcCorr - LK111114
% Function that calculates autocorrelation 2nd order only
% Input:
%   - innerpts -
%   - c - scaling factor for 
%   - L -
%   - DataVector - 1D vector of the image to be analyzed

function [ACXC_all]=CalcCorr(innerpts,DataVector,c,L);

    %i range accomodates time offset
    i = 1;
    for i = 1:length(innerpts)-2        
    %taking only indices we care about
            index = innerpts(i);
    %Fluorescence intensity calculations        
            %fluorescence fluctuation
            F = DataVector(:,index);
            dF = F - mean(F);

            %dF of all surrounding points
            dFD = DataVector(:, index+1)- mean(DataVector(:, index+1));
            dFR = DataVector(:, index+L+1)- mean(DataVector(:, index+L+1));
            dFDR = DataVector(:, index+L+2)- mean(DataVector(:, index+L+2));

    %% Perform AC, XC calculations
            %autocorrelating and cross-correlating with xcorr
            [AC, lags] = xcorr(dF, dF); 

            %min_lags staring at zero
            min_lags = (length(lags)-1)/2+1;
            max_lags=length(lags);

            %fluctuations about time offset
            F2 = DataVector(:, index+1);
            dF2 = F2-mean(F2);
            F3 = DataVector(:, index+2);
            dF3 = F3- mean(F3);      

    % 2nd order correlation
            [corrCombined2]=rcSOFI_2ndOrder_AConly(AC,  min_lags,max_lags);        

            %LK-add storing the AC and xcorr curves for each pixel, i
            ACXC_all(1,i)=struct('Order2',corrCombined2);

            i = i+1;

    end
end