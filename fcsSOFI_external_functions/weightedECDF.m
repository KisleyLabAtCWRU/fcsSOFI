%{
Computes the weighted empericle cumulative distribution values given a
empericle distribution. 

Inputes:
vals - The values recieved from the distribution (diffusion coeficients)
weights - The weighted importance of the value (sofi values)

vals and weights must be the same size, as each val must have a weight. 
vals and weights can be a matrix or a vector. Matrix's will just be 
linearly indexed, treating all values to be part of the same distribution.
The val and weight which share the same index are pared.

Output:
vals - a sorted version (low to high) of input vals with NaN's removed.
ecdfProb - The assiated ecdf probability for each value in vals in the same
            order as vals

To recive the ecdf plot, do something along the lines of:
plot(vals, ecdfProb, 'k.')
%}
function [vals, ecdfProb] = weightedECDF(vals, weights)

% Make linear
vals = vals(:);
weights = weights(:);

% Remove and NaNs from the data
weightNans = ~isnan(weights);
diffNans = ~isnan(vals);
nans = logical(weightNans .* diffNans); % Combines NaN Positions from both arrays
weights = weights(nans);
vals = vals(nans);

% Sorts values accending and brings weights with it
[vals, newIndex] = sort(vals);
weights = weights(newIndex);

% Normalize so all weights add to 1
weights = weights ./ sum(weights);

% Sum of all previouse weights is the value of the ecdf at that point
ecdfProb = cumsum(weights);

% Plot for testing
%{
figure
plot(vals, ecdfProb, 'k.')
ylim([0 1])
title("Weighted CDF")
%}
