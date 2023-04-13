function [PSF] = AveragePSFperFrame(LocatStore)
avgPSFx1 = zeros(size(LocatStore, 2));
avgPSFy1 = zeros(size(LocatStore, 2));
for i = 1:size(LocatStore, 2)
     PSFx1 = LocatStore(i).PSFData(:,3);
     PSFy1 = LocatStore(i).PSFData(:,4);
     
     avgPSFx1(i) = mean(PSFx1, 'omitnan');
     avgPSFy1(i) = mean(PSFy1, 'omitnan');
end
totalAvgPSFx = mean(avgPSFx1, 'omitnan');
totalAvgPSFy = mean(avgPSFy1, 'omitnan');
PSF = (totalAvgPSFx + totalAvgPSFy) / 2;
end