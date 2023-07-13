%{
A function used to run SOFI anylsis and use the results as you see fit.
There is a section with alot of plotting which can be turned on or off with
oneRun. There is also a section which compares all the SOFI images and find
the percent similarity to the origonal binary map, this section does not
work with current simulation code because teh binarly map is now scaled up

Inputes:
data - The image stack to compute the SOFI anylsis on
truth - The image stack containing all the true point source locations
poreMap - The 2D image showing the binary map which emmitters can diffuse in
oneRun - 0 or 1, Plot a bunch of images if true

Output:
percents - An array containing all the percent similariries to the true
sigma - The PSF sigma estimate from the SOFI calculations
%}
function [percents, sigma] = testSOFI(data, truth, poreMap, oneRun)

%fprintf("Calculating SOFI information... ")

data = double(data);

%% SOFI Calculations
average = mean(data, 3);
[AC2, AC3, AC4, AG4, AC2Corrected, AC3Corrected, AG4Corrected] = autoSofi(data);
[XC2, XC3, XC2Corrected, sigma] = crossSofi(data);
[deconAC, deconXC, deconAvg] = decon(average, {AC2, AC3, AC4}, {XC2, XC3}, sigma);

truthSat = satAdj(truth, 0, 0.05);

%{
AC2 = satAdj(AC2, 0, 0.5);
AC3 = satAdj(AC3, 0, 0.5);
AC2Corrected = satAdj(AC2Corrected, 0, 0.5);
AC3Corrected = satAdj(AC3Corrected, 0, 0.5);
%}

%fprintf("Finished \n")


%% Compare Images
truthBinary = double(imbinarize(truth, 0));
%{
AC2Diff = imabsdiff(truthBinary, AC2);
AC2Percent = 1 - sum(AC2Diff(:)) ./ numel(AC2);

AC3Diff = imabsdiff(truthBinary, AC3);
AC3Percent = 1 - sum(AC3Diff(:)) ./ numel(AC3);

AC4Diff = imabsdiff(truthBinary, AC4);
AC4Percent = 1 - sum(AC4Diff(:)) ./ numel(AC4);

deconAC2Diff = imabsdiff(truthBinary, deconAC{1});
deconAC2Percent = 1 - sum(deconAC2Diff(:)) ./ numel(deconAC{1});

deconAC3Diff = imabsdiff(truthBinary, deconAC{2});
deconAC3Percent = 1 - sum(deconAC3Diff(:)) ./ numel(deconAC{2});

deconAC4Diff = imabsdiff(truthBinary, deconAC{3});
deconAC4Percent = 1 - sum(deconAC4Diff(:)) ./ numel(deconAC{3});

XC2Diff = imabsdiff(imresize(truthBinary, size(XC2), "nearest"), XC2);
XC2Percent = 1 - sum(XC2Diff(:)) ./ numel(XC2);

deconXC2Diff = imabsdiff(imresize(truthBinary, size(deconXC{1}), "nearest"), deconXC{1});
deconXC2Percent = 1 - sum(deconXC2Diff(:)) ./ numel(deconXC{1});

averageDiff = imabsdiff(truthBinary, average);
averagePercent = 1 - sum(averageDiff(:)) ./ numel(average);

deconAverageDiff = imabsdiff(truthBinary, deconAvg);
deconAverageAC2Percent = 1 - sum(deconAverageDiff(:)) ./ numel(deconAvg);

percents = [AC2Percent, AC3Percent, AC4Percent, deconAC2Percent, ...
    deconAC3Percent, deconAC4Percent, XC2Percent, deconXC2Percent, ...
    averagePercent, deconAverageAC2Percent];
%}
if oneRun

    %% Print Everything
%{
    fprintf('AC2 Accuracy: %f \n', AC2Percent)
    fprintf('AC3 Accuracy: %f \n', AC3Percent)
    fprintf('AC4 Accuracy: %f \n', AC4Percent)
    fprintf('AC2 Decon Accuracy: %f \n', deconAC2Percent)
    fprintf('AC3 Decon Accuracy: %f \n', deconAC3Percent)
    fprintf('AC4 Decon Accuracy: %f \n', deconAC4Percent)
    fprintf('XC2 Accuracy: %f \n', XC2Percent)
    fprintf('XC2 Decon Accuracy: %f \n', deconXC2Percent)
    fprintf('Average Accuracy: %f \n', averagePercent)
    fprintf('Average Decon Accuracy: %f \n', deconAverageAC2Percent)
%}
    fprintf('PSF Estimate std: %f \n', sigma)
    
    %% Plot Everything
    
    % Average Images
    figure('Name', 'Average Images', 'Units', 'normalized', 'Position', [0.705208333333333,0.019444444444444,0.3125,0.314814814814815]);
    
    subplot(1, 2, 1)
    imagesc(average)
    colormap(gray)
    title("Average")
    axis image
    
    subplot(1, 2, 2)
    imagesc(deconAvg)
    colormap(gray)
    title("Average Decon")
    axis image
    
    % AC Images
    figure('Name', 'AC Images', 'Units', 'normalized', 'Position', [0.689583333333333,0.3,0.307291666666667,0.615740740740741])
    
    subplot(3, 2, 1)
    imagesc(AC2)
    colormap(gray)
    title("AC2")
    axis image
    
    subplot(3, 2, 2)
    imagesc(deconAC{1})
    colormap(gray)
    title("AC2 Decon")
    axis image
    
    subplot(3, 2, 3)
    imagesc(AC3)
    colormap(gray)
    title("AC3")
    axis image
    
    subplot(3, 2, 4)
    imagesc(deconAC{2})
    colormap(gray)
    title("AC3 Decon")
    axis image
    
    subplot(3, 2, 5)
    imagesc(AC4)
    colormap(gray)
    title("AC4")
    axis image
    
    subplot(3, 2, 6)
    imagesc(deconAC{3})
    colormap(gray)
    title("AC4 Decon")
    axis image
    
    % XC Images
    figure('Name', 'XC Images', 'Units', 'normalized', 'Position', [0.365625,0.002777777777778,0.3421875,0.585185185185185])
    
    subplot(2, 2, 1)
    imagesc(XC2)
    colormap(gray)
    title("XC2")
    axis image
    
    subplot(2, 2, 2)
    imagesc(deconXC{1})
    colormap(gray)
    title("XC2 Decon")
    axis image
    
    subplot(2, 2, 3)
    imagesc(XC3)
    colormap(gray)
    title("XC3")
    axis image
    
    subplot(2, 2, 4)
    imagesc(deconXC{2})
    colormap(gray)
    title("XC3 Decon")
    axis image

    % Line Sections
    scaler = length(poreMap) / length(AC2);
    
    XC2 = padarray(XC2, [2, 2]);
    XC3 = padarray(XC3, [3, 3]);
    deconXC{1} = padarray(deconXC{1}, [2, 2]);
    deconXC{2} = padarray(deconXC{2}, [3, 3]);
    posArrayXC2 = scaler+1:scaler/2:length(poreMap)+scaler/2;
    posArrayXC3 = scaler+1:scaler/3:length(poreMap)+scaler/3;
    posArrayAC = scaler+1:scaler:length(poreMap)+scaler;
    posArrayPore = 1:1:length(poreMap);

    XC2Line = mean(XC2, 1);
    XC3Line = mean(XC3, 1);
    AC2Line = mean(AC2, 1);
    AC3Line = mean(AC3, 1);
    AC4Line = mean(AC4, 1);
    AC2CorrectLine = mean(AC2Corrected, 1);
    AC3CorrectLine = mean(AC3Corrected, 1);
    AG4CorrectLine = mean(AG4Corrected, 1);
    avgLine = mean(average, 1);

    deconXC2Line = mean(deconXC{1}, 1);
    deconXC3Line = mean(deconXC{2}, 1);
    deconAC2Line = mean(deconAC{1}, 1);
    deconAC3Line = mean(deconAC{2}, 1);
   
    figure('Name', 'Line Sections', 'Units', 'normalized', 'Position', [-0.021875,0.001851851851852,0.4140625,0.614814814814815])
    
    subplot(2, 1, 1)
    hold on
    plot(posArrayPore, poreMap(25, :) ./ max(poreMap(25, :)), '-k', 'DisplayName', 'Binary Map')
    plot(posArrayAC, avgLine ./ max(avgLine), '--k.', 'DisplayName', 'Average')
    plot(posArrayXC2, XC2Line ./ max(XC2Line), '-b.', 'DisplayName', 'XC2')
    plot(posArrayXC3, XC3Line ./ max(XC3Line), '-c.', 'DisplayName', 'XC3')
    plot(posArrayAC, AC2Line ./ max(AC2Line), '-g.', 'DisplayName', 'AC2')
    plot(posArrayAC, AC3Line ./ max(AC3Line), '-r.', 'DisplayName', 'AC3')
    legend
    title('SOFI Line Sections')
    hold off

    subplot(2, 1, 2)
    hold on
    plot(posArrayPore, poreMap(25, :) ./ max(poreMap(25, :)), '-k', 'DisplayName', 'Binary Map')
    plot(posArrayAC, avgLine ./ max(avgLine), '--k.', 'DisplayName', 'Average')
    plot(posArrayXC2, deconXC2Line ./ max(deconXC2Line), '-b.', 'DisplayName', 'XC2')
    plot(posArrayXC3, deconXC3Line ./ max(deconXC3Line), '-c.', 'DisplayName', 'XC3')
    plot(posArrayAC, deconAC2Line ./ max(deconAC2Line), '-g.', 'DisplayName', 'AC2')
    plot(posArrayAC, deconAC3Line ./ max(deconAC3Line), '-r.', 'DisplayName', 'AC3')
    legend
    title('Decon SOFI Line Sections')
    hold off


    figure('Name', 'Corrected Line Sections')

    hold on
    plot(posArrayPore, poreMap(25, :) ./ max(poreMap(25, :)), '-k', 'DisplayName', 'Binary Map')
    plot(posArrayAC, avgLine ./ max(avgLine), '--k.', 'DisplayName', 'Average')
    plot(posArrayAC, AC2Line ./ max(AC2Line), '-g.', 'DisplayName', 'AC2')
    plot(posArrayAC, AC3Line ./ max(AC3Line), '-r.', 'DisplayName', 'AC3')
    plot(posArrayAC, AC4Line ./ max(AC4Line), '-b.', 'DisplayName', 'AC4')
    plot(posArrayAC, AC2CorrectLine ./ max(AC2CorrectLine), '--g.', 'DisplayName', 'AC2 Corrected')
    plot(posArrayAC, AC3CorrectLine ./ max(AC3CorrectLine), '--r.', 'DisplayName', 'AC3 Corrected')
    plot(posArrayAC, AG4CorrectLine ./ max(AG4CorrectLine), '--b.', 'DisplayName', 'AC4 Corrected')
    legend
    title('Corrected AC SOFI Line Sections')
    hold off
    %}

    % Real Images
    figure('Name', 'Real Images', 'Units', 'normalized', 'Position', [0.0546875, 0.600925925925926, 0.652083333333333, 0.316666666666667])
    
    subplot(1, 3, 1)
    imagesc(truthSat)
    colormap(gray)
    title("True Pixel Positions")
    axis image
    
    subplot(1, 3, 2)
    imagesc(truthBinary)
    colormap(gray)
    title("True Pixel Positions")
    axis image
    
    subplot(1, 3, 3)
    imagesc(poreMap)
    colormap(gray)
    title("Pore Map")
    axis image


    % Corrected AC Images
    figure('Name', 'Corrected AC Images')

    subplot(2, 4, 1)
    imagesc(AC2)
    colormap(gray)
    title("AC2")
    axis image

    subplot(2, 4, 2)
    imagesc(AC3)
    colormap(gray)
    title("AC3")
    axis image

    subplot(2, 4, 3)
    imagesc(AG4)
    colormap(gray)
    title("AG4")
    axis image

    subplot(2, 4, 4)
    imagesc(AC4)
    colormap(gray)
    title("AC4")
    axis image

    subplot(2, 4, 5)
    imagesc(AC2Corrected)
    colormap(gray)
    title("AC2 Corrected")
    axis image

    subplot(2, 4, 6)
    imagesc(AC3Corrected)
    colormap(gray)
    title("AC3 Corrected")
    axis image

    subplot(2, 4, 7)
    imagesc(AG4Corrected)
    colormap(gray)
    title("AG4 Corrected")
    axis image

    
    figure('Name', 'Corrected XC Images')

    subplot(2, 1, 1)
    imagesc(XC2)
    colormap(gray)
    title("XC2")
    axis image

    subplot(2, 1, 2)
    imagesc(XC2Corrected)
    colormap(gray)
    title("AX2 Corrected")
    axis image

end
end


% Adjust the sat max and min of an image
function imageSat = satAdj(image, satMin, satMax)

    image = rescale(image);
    image(image < satMin) = satMin;
    image(image > satMax) = satMax;
    imageSat = rescale(image);

end
