%{
Computes the cross correlation map for 2nd and 3rd order sofi of a given matrix
representing the video data.

Inputes:
data - the matrix of data to compute SOFI on

data must be in the format height by width by frames.
Ex: In a 5x10x30 each frame is 5 pixels tall and 10 pixels wide and there
are 30 frames.
%}

function [XC2, XC3, XC2Corrected, sigma] = crossSofi(data)

%% Variable Set Up
sz = size(data);
height = sz(1);
width = sz(2);
nframes = sz(3);
tau = [1, 4, 6]; % Chosen Time Shifts
data = double(data);
average = mean(data,3);
delta = data - average;

i = 2:height-1;
j = 2:width-1;
t = 1:(nframes-max(tau));
clear data

%% Cross Correlation Calculations
% Second Order
AC2 = abs(mean(delta(i,j,t) .* delta(i,j,t), 3));
XC2Vertical = abs(mean(delta(i,j,t) .* delta(i+1,j,t), 3));
XC2Horizontal = abs(mean(delta(i,j,t) .* delta(i,j+1,t), 3));
XC2Diagonal1 = abs(mean(delta(i,j,t) .* delta(i+1,j+1,t), 3));
XC2Diagonal2 = abs(mean(delta(i+1,j,t) .* delta(i,j+1,t), 3));
XC2Diagonal = (XC2Diagonal1 + XC2Diagonal2) / 2;

% Create XC2 matrix from XC2 claculations
XC2 = zeros(2*(height-2)-1, 2*(width-2)-1);
XC2(1:2:end, 1:2:end) = AC2; 
XC2(2:2:end, 1:2:end) = XC2Vertical(1:end-1, :); 
XC2(1:2:end, 2:2:end) = XC2Horizontal(:, 1:end-1);  
XC2(2:2:end, 2:2:end) = XC2Diagonal(1:end-1, 1:end-1);

% Diffusion Corrections
origCorrection = average(i, j).^2; 
vertCorrection = average(i, j) .* average(i+1, j);
horiCorrection = average(i, j) .* average(i, j+1); 
diagCorrection = (average(i,j) .* average(i+1,j+1) + average(i+1, j) .* average(i, j+1)) ./ 2;

XC2Corrected = zeros(2*(height-2)-1, 2*(width-2)-1);
XC2Corrected(1:2:end, 1:2:end) = AC2 + origCorrection;
XC2Corrected(2:2:end, 1:2:end) = XC2Vertical(1:end-1, :) + vertCorrection(1:end-1, :);
XC2Corrected(1:2:end, 2:2:end) = XC2Horizontal(:, 1:end-1) + horiCorrection(:, 1:end-1);
XC2Corrected(2:2:end, 2:2:end) = XC2Diagonal(1:end-1, 1:end-1) + diagCorrection(1:end-1, 1:end-1);

clear XC2Vertical XC2Horizontal XC2Diagonal1 XC2Diagonal2 XC2Diagonal
clear origCorrection vertCorrection horiCorrection diagCorrection

% Third Order
for k = 1:size(tau,1)
    AC3            = abs(mean(delta(i,j,t)   .* delta(i,j,t+tau(2))     .* delta(i,j,t+tau(3)), 3));
    XC3Horizontal1 = abs(mean(delta(i,j,t)   .* delta(i,j,t+tau(2))     .* delta(i,j+1,t+tau(3)), 3));
    XC3Horizontal2 = abs(mean(delta(i,j,t)   .* delta(i,j+1,t+tau(2))   .* delta(i,j+1,t+tau(3)), 3));
    XC3Vertical1   = abs(mean(delta(i,j,t)   .* delta(i,j,t+tau(2))     .* delta(i+1,j,t+tau(3)), 3));
    XC3Vertical2   = abs(mean(delta(i,j,t)   .* delta(i+1,j,t+tau(2))   .* delta(i+1,j,t+tau(3)), 3));
    XC3Diagonal1   = abs(mean(delta(i,j,t)   .* delta(i,j,t+tau(2))     .* delta(i+1,j+1,t+tau(3)), 3));
    XC3Diagonal2   = abs(mean(delta(i,j,t)   .* delta(i+1,j+1,t+tau(2)) .* delta(i+1,j+1,t+tau(3)), 3));
    XC3Diagonal3   = abs(mean(delta(i,j+1,t) .* delta(i,j+1,t+tau(2))   .* delta(i+1,j,t+tau(3)), 3));
    XC3Diagonal4   = abs(mean(delta(i,j+1,t) .* delta(i+1,j,t+tau(2))   .* delta(i+1,j,t+tau(3)), 3));
end
XC3 = zeros(3*(height-2)-2, 3*(width-2)-2);
XC3(1:3:end,1:3:end) = AC3(1:end,1:end);
XC3(1:3:end,2:3:end) = XC3Horizontal1(1:end,1:end-1);
XC3(1:3:end,3:3:end) = XC3Horizontal2(1:end,1:end-1);
XC3(2:3:end,1:3:end) = XC3Vertical1(1:end-1,1:end);
XC3(3:3:end,1:3:end) = XC3Vertical2(1:end-1,1:end);
XC3(2:3:end,2:3:end) = XC3Diagonal1(1:end-1,1:end-1);
XC3(3:3:end,3:3:end) = XC3Diagonal2(1:end-1,1:end-1);
XC3(2:3:end,3:3:end) = XC3Diagonal3(1:end-1,1:end-1);
XC3(3:3:end,2:3:end) = XC3Diagonal4(1:end-1,1:end-1);

fprintf('Finished.\n');
clear AC3 XC3Horizontal1 XC3Horizontal2...
    XC3Vertical1 XC3Vertical2 XC3Diagonal1...
    XC3Diagonal2 XC3Diagonal3 XC3Diagonal4;


%% Distance Factor Calculations
% Filter to only use important locations for calculation
averageBlur = imgaussfilt(average, 2);
averageBlur = averageBlur - min(averageBlur(:));
averageBlur = mat2gray(averageBlur);
filterGradient = bwconvhull(abs(gradient(averageBlur)) > 0.01, 'objects');
filterIntensity = (averageBlur > 0.25 & averageBlur < 0.9);
filter = logical(filterGradient .* filterIntensity);
filter = filter(i, j);

% Extra Cross Corrilations
XC2Vert = abs(mean(delta(i-1,j,t) .* delta(i+1,j,t+tau(1)), 3));
XC2Hor = abs(mean(delta(i,j-1,t) .* delta(i,j+1,t+tau(1)), 3));
XC2Diag1 = abs(mean(delta(i-1,j-1,t) .* delta(i+1,j+1,t+tau(1)), 3));
XC2Diag2 = abs(mean(delta(i+1,j-1,t) .* delta(i-1,j+1,t+tau(1)), 3));

% Calculate distance factors
distVert = XC2Vert ./ AC2;
distHor = XC2Hor ./ AC2;
distDiag1 = XC2Diag1 ./ AC2;
distDiag2 = XC2Diag2 ./ AC2;

% Distance factors must be between zero and one
distVert(distVert > 1) = 1;
distHor(distHor > 1) = 1;
distDiag1(distDiag1 > 1) = 1;
distDiag2(distDiag2 > 1) = 1;
distVert(distVert < 0) = 0;
distHor(distHor < 0) = 0;
distDiag1(distDiag1 < 0) = 0;
distDiag2(distDiag2 < 0) = 0;

% Find average distance value from important regions
distVert = mean(distVert(filter), "omitnan");
distHor = mean(distHor(filter), "omitnan");
distDiag1 = mean(distDiag1(filter), "omitnan");
distDiag2 = mean(distDiag2(filter), "omitnan");
distDiag = (distDiag1 + distDiag2) / 2;


%% Distance Factor Applications
XC2(2:2:end, 1:2:end) = XC2(2:2:end, 1:2:end) / distVert^(1/4);
XC2(1:2:end, 2:2:end) = XC2(1:2:end, 2:2:end) / distHor^(1/4);
XC2(2:2:end, 2:2:end) = XC2(2:2:end, 2:2:end) / distDiag^(1/4);

XC3(1:3:end,2:3:end) = XC3(1:3:end,2:3:end) / distHor^(2/6);
XC3(1:3:end,3:3:end) = XC3(1:3:end,3:3:end) / distHor^(2/6);
XC3(2:3:end,1:3:end) = XC3(2:3:end,1:3:end) / distVert^(2/6);
XC3(3:3:end,1:3:end) = XC3(3:3:end,1:3:end) / distVert^(2/6);
XC3(2:3:end,2:3:end) = XC3(2:3:end,2:3:end) / distDiag1^(2/6);
XC3(3:3:end,3:3:end) = XC3(3:3:end,3:3:end) / distDiag1^(2/6);
XC3(2:3:end,3:3:end) = XC3(2:3:end,3:3:end) / distDiag2^(2/6);
XC3(3:3:end,2:3:end) = XC3(3:3:end,2:3:end) / distDiag2^(2/6);

%% Normalize
%{
XC2 = rescale(XC2);
XC3 = rescale(XC3);
%}

%% Calculate Estimate PSF
sigmaX = sqrt(-1/(log(distHor)));
sigmaY = sqrt(-1/(log(distVert)));
sigmaDiag1 = sqrt(-2/(log(distDiag1)));
sigmaDiag2 = sqrt(-2/(log(distDiag2)));
sigma = (sigmaX + sigmaY + sigmaDiag1 + sigmaDiag2) / 4;

% Old Implementation
%{
    %% Cross Correlation Calculations
    if doNormalCross % Normal Cross Correlation
        sofi = zeros(height, width);
        for i = 2:height-1
            for j = 2:width-1
                AboveBelow = xcorr(data(:, i-1, j), data(:, i+1, j), 0);
                LeftRight = xcorr(data(:, i, j-1), data(:, i, j+1), 0);
                sofi(i, j) = (AboveBelow + LeftRight) / 2;
            end
            if i == round(height / 4)
                disp('Sofi No Virtual Cross Corr 25% Done')
            elseif i == round(height / 2)
                disp('Sofi No Virtual Cross Corr 50% Done')
            elseif i == round(3 * height / 4)
                disp('Sofi No Virtual Cross Corr 75% Done')
            elseif i == round(height - 1)
                disp('Sofi No Virtual Cross Corr 100% Done')
            end
        end
        sofi = sofi(2:height-2, 2:width-2);


    else % Virtual Cross Correlation
        sofi = zeros(height*2-1, width*2-1);
        for i = 2:height*2-2
            for j = 2:width*2-2
                if (mod(i,2) == 0)
                    if (mod(j,2) == 0) % i even, j even
                        downDiag = xcorr(data(:, convert(i-1), convert(j-1)), data(:, convert(i+1), convert(j+1)), 0);
                        upDiag = xcorr(data(:, convert(i-1), convert(j+1)), data(:, convert(i+1), convert(j-1)), 0);
                        sofi(i, j) = ((downDiag + upDiag) / 2);
                    else % i even, j odd
                        aboveBelow = xcorr(data(:, convert(i-1), convert(j)), data(:, convert(i+1), convert(j)), 0);
                        sofi(i, j) = aboveBelow;
                    end
                else
                    if (mod(j,2) == 0) % i odd, j even
                        leftRight = xcorr(data(:, convert(i), convert(j-1)), data(:, convert(i), convert(j+1)), 0);
                        sofi(i, j) = leftRight;
                    else % i odd, j odd
                        aboveBelow = xcorr(data(:, convert(i-2), convert(j)), data(:, convert(i+2), convert(j)), 0);
                        leftRight = xcorr(data(:, convert(i), convert(j-2)), data(:, convert(i), convert(j+2)), 0);
                        sofi(i, j) = ((aboveBelow + leftRight) / 2);
                    end
                end
            end
            if i == round(height / 2)
                disp('Sofi Cross Corr 25% Done')
            elseif i == round(height)
                disp('Sofi Cross Corr 50% Done')
            elseif i == round(3 * height / 2)
                disp('Sofi Cross Corr 75% Done')
            elseif i == round(height * 2 - 2)
                disp('Sofi Cross Corr 100% Done')
            end

        end
        
        
        %% Virtual Pixel Corrections
        horizPlusVirtMean = mean(reshape(sofi(3:2:height*2-2, 3:2:width*2-2), 1, []));
        horizPlusVirtStd = std(reshape(sofi(3:2:height*2-2, 3:2:width*2-2), 1, []));
        horizonMean = mean(reshape(sofi(3:2:height*2-2, 2:2:width*2-2), 1, []));
        horizonStd = std(reshape(sofi(3:2:height*2-2, 2:2:width*2-2), 1, []));
        virtMean = mean(reshape(sofi(2:2:height*2-2, 3:2:width*2-2), 1, []));
        virtStd = std(reshape(sofi(2:2:height*2-2, 3:2:width*2-2), 1, []));
        diagMean = mean(reshape(sofi(2:2:height*2-2, 2:2:width*2-2), 1, []));
        diagStd = std(reshape(sofi(2:2:height*2-2, 2:2:width*2-2), 1, []));
    
        aHorizon = horizPlusVirtStd / horizonStd;
        bHorizon = horizPlusVirtMean - aHorizon*horizonMean;
        aVert = horizPlusVirtStd / virtStd;
        bVert = horizPlusVirtMean - aVert*virtMean;
        aDiag = horizPlusVirtStd / diagStd;
        bDiag = horizPlusVirtMean - aDiag*diagMean;
    
        sofi(3:2:height*2-2, 2:2:width*2-2) = aHorizon * sofi(3:2:height*2-2, 2:2:width*2-2) + bHorizon;
        sofi(2:2:height*2-2, 3:2:width*2-2) = aVert * sofi(2:2:height*2-2, 3:2:width*2-2) + bVert;
        sofi(2:2:height*2-2, 2:2:width*2-2) = aDiag * sofi(2:2:height*2-2, 2:2:width*2-2) + bDiag;

        sofi = sofi(2:height*2-2, 2:width*2-2); 
    end

    crossSofi = sofi;
end

% Converts a virtual index into a real index
function [i] = convert(vi)
    i = (vi + 1) / 2;
end
%}

end