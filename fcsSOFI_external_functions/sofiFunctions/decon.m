%{
Function to take in SOFI AC and XC maps with an estimated sigma and return
maps after deconvblind has been used on them

Input:
    AC - Cell array containing all auto correlation maps
    XC - Cell array containing all cross correlation maps
    sigma - Scaler, the estimated standard deviation of the guassian psf
    average - 2D array containing the average image

    AC and XC must be formated in the following way: 
    AC = {AC2, AC3, AC4, ..., ACN}
    XC = {XC2, XC3, XC4, ..., XCN}
    AC and XC must start at 2nd order and increase by one each index.

Output:
    deconAC - Cell array containing all deconvoluted auto correlation maps
    deconXC - Cell array containing all deconvoluted cross correlation maps
    deconAvg - 2D array containing deconvoluted average image
    Both deconAC and deconAvg are retuned in same format as Input
%}
function [deconAC, deconXC, deconAvg] = decon(average, AC, XC, sigma)

    deconXC = cell(size(XC));
    deconAC = cell(size(AC));

    psf = fspecial('gaussian', [size(average, 1), size(average, 2)], double(sigma));
    deconAvg = deconvblind(average, psf);

    for order = 1:length(AC)
        % Create guassian psf with converted sigma and to correct power
        % Need power since PSF is raised to the nth power in theory
        psf = (fspecial('gaussian', [size(AC{order}, 1), size(AC{order}, 2)], double(sigma))).^(order+1);
        deconAC{order} = deconvblind(AC{order}, psf);
    end

    for order = 1:length(XC)
        % Need to convert sigma since extra pixels
        psf = (fspecial('gaussian', [size(XC{order}, 1), size(XC{order}, 2)], double((order+1)*sigma))).^(order+1);
        deconXC{order} = deconvblind(XC{order}, psf);
    end
end


