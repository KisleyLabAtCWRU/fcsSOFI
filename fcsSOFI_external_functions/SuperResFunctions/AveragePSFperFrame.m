function [PSF] = AveragePSFperFrame(LocatStore)
avgPSFx1 = zeros(size(LocatStore));
avgPSFy1 = zeros(size(LocatStore));
for i = 1:size(LocatStore, 2)
     PSFx1 = LocatStore(i).PSFData(:,3);
     PSFy1 = LocatStore(i).PSFData(:,4);
     
     avgPSFx1(i) = mean(squeeze(PSFx1), 'omitnan');
     avgPSFy1(i) = mean(squeeze(PSFy1), 'omitnan');
end
totalAvgPSFx = mean(squeeze(avgPSFx1), 'omitnan');
totalAvgPSFy = mean(squeeze(avgPSFy1), 'omitnan');
PSF = (totalAvgPSFx + totalAvgPSFy) / 2;
end