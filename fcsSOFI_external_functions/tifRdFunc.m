%% Script to load a tiff file parralized

function [Data] = tifRdFunc(fileName, path, frames, tiffReadVol)

% If no worker pool, start one
p = gcp;
if isempty(p)
    parpool('Processes');
end

% Pre-allocate space for parralized future results
f(1:10) = parallel.FevalFuture;

% Open tiff using tiffreadVolume (requires matlab 2021b)
if (tiffReadVol)
    f(1) = parfeval(@tiffreadVolume, 1, fileName, 'PixelRegion', {[1, inf], [1, inf], [1, floor(frames/10)]});
    f(2) = parfeval(@tiffreadVolume, 1, fileName, 'PixelRegion', {[1, inf], [1, inf], [floor(frames/10)+1, floor(2*frames/10)]});
    f(3) = parfeval(@tiffreadVolume, 1, fileName, 'PixelRegion', {[1, inf], [1, inf], [floor(2*frames/10)+1, floor(3*frames/10)]});
    f(4) = parfeval(@tiffreadVolume, 1, fileName, 'PixelRegion', {[1, inf], [1, inf], [floor(3*frames/10)+1, floor(4*frames/10)]});
    f(5) = parfeval(@tiffreadVolume, 1, fileName, 'PixelRegion', {[1, inf], [1, inf], [floor(4*frames/10)+1, floor(5*frames/10)]});
    f(6) = parfeval(@tiffreadVolume, 1, fileName, 'PixelRegion', {[1, inf], [1, inf], [floor(5*frames/10)+1, floor(6*frames/10)]});
    f(7) = parfeval(@tiffreadVolume, 1, fileName, 'PixelRegion', {[1, inf], [1, inf], [floor(6*frames/10)+1, floor(7*frames/10)]});
    f(8) = parfeval(@tiffreadVolume, 1, fileName, 'PixelRegion', {[1, inf], [1, inf], [floor(7*frames/10)+1, floor(8*frames/10)]});
    f(9) = parfeval(@tiffreadVolume, 1, fileName, 'PixelRegion', {[1, inf], [1, inf], [floor(8*frames/10)+1, floor(9*frames/10)]});
    f(10) = parfeval(@tiffreadVolume, 1, fileName, 'PixelRegion', {[1, inf], [1, inf], [floor(9*frames/10)+1, frames]});
else % Open tiff using TiffReadRM (slower) (Must have TiffReadRM added to path)
    f(1) = parfeval(@TiffReadRM, 1, fileName, path, 1, floor(frames/10));
    f(2) = parfeval(@TiffReadRM, 1, fileName, path, floor(frames/10)+1, floor(2*frames/10));
    f(3) = parfeval(@TiffReadRM, 1, fileName, path, floor(2*frames/10)+1, floor(3*frames/10));
    f(4) = parfeval(@TiffReadRM, 1, fileName, path, floor(3*frames/10)+1, floor(4*frames/10));
    f(5) = parfeval(@TiffReadRM, 1, fileName, path, floor(4*frames/10)+1, floor(5*frames/10));
    f(6) = parfeval(@TiffReadRM, 1, fileName, path, floor(5*frames/10)+1, floor(6*frames/10));
    f(7) = parfeval(@TiffReadRM, 1, fileName, path, floor(6*frames/10)+1, floor(7*frames/10));
    f(8) = parfeval(@TiffReadRM, 1, fileName, path, floor(7*frames/10)+1, floor(8*frames/10));
    f(9) = parfeval(@TiffReadRM, 1, fileName, path, floor(8*frames/10)+1, floor(9*frames/10));
    f(10) = parfeval(@TiffReadRM, 1, fileName, path, floor(9*frames/10)+1, frames);
end

% Pre-allocate space for results from the future objects
results = cell(1,10);
% Fetch the results from the future object when they complete
for idx = 1:10
    [completedIdx, value] = fetchNext(f); % Fetch the result (returns which one and the result)
    results{completedIdx} = value; % Place in result cell in correct location 
    %fprintf('Got result with index: %d.\n', completedIdx);
end

% Combines all the results together back into one matrix
Data = cat(3, results{1}, results{2}, results{3}, results{4}, results{5}, results{6}, results{7}, results{8}, results{9}, results{10});