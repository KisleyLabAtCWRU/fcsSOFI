%{
Function to compute the auto correlation curves of a stack of data

Intput:
data - A 3D matrix conataining the data to compute the auto correlation on. Must be in the format
    height by width by frames

Output:
autoCorrelations - 2D matrix containing the auto correlation curves.
    Columns contain the autocorrelation curve for a single pizel starting 
    at time lag 0. Each column is a different pixel. Changing the linear
    index to the row, col equivilent with ind2sub will give the origonal
    postion back.
%}
function [autoCorrelations] = fcsCorrelate(data)

    data = double(data);
    delta = data - mean(data, 3);
    autoCorrelations = zeros(size(data, 3), numel(data(:, :, 1)));

    % Using a linear index loop to allow for parralization later
    for iLin = 1:(numel(data(:, :, 1)))

        [i, j] = ind2sub(size(data(:, :, 1)), iLin);
        
        [AC, lags] = xcorr(delta(i, j, :));

        AC = AC(lags >= 0);
        
        autoCorrelations(:, iLin) = AC;
    end
end
