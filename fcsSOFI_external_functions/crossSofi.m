function [crossSofiVirt, crossSofi] = crossSofi(data, doNormalCross, satMax)


    %% Variable Set Up
    crossSofi = 0;
    crossSofiVirt = 0;
    sz = size(data);
    height = sz(1);
    width = sz(2);
    data = permute(data, [3, 1, 2]);


    %% Cross Correlation Calculations

    % If no virtual pixels wanted
    if doNormalCross
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
    
    % Else do have virtual pixels
    else
        vSofi = zeros(height*2-1, width*2-1);
        for i = 2:height*2-2
            for j = 2:width*2-2
                if (mod(i,2) == 0)
                    if (mod(j,2) == 0) % i even, j even
                        downDiag = xcorr(data(:, convert(i-1), convert(j-1)), data(:, convert(i+1), convert(j+1)), 0);
                        upDiag = xcorr(data(:, convert(i-1), convert(j+1)), data(:, convert(i+1), convert(j-1)), 0);
                        vSofi(i, j) = ((downDiag + upDiag) / 2);
                    else % i even, j odd
                        aboveBelow = xcorr(data(:, convert(i-1), convert(j)), data(:, convert(i+1), convert(j)), 0);
                        vSofi(i, j) = aboveBelow;
                    end
                else
                    if (mod(j,2) == 0) % i odd, j even
                        leftRight = xcorr(data(:, convert(i), convert(j-1)), data(:, convert(i), convert(j+1)), 0);
                        vSofi(i, j) = leftRight;
                    else % i odd, j odd
                        aboveBelow = xcorr(data(:, convert(i-2), convert(j)), data(:, convert(i+2), convert(j)), 0);
                        leftRight = xcorr(data(:, convert(i), convert(j-2)), data(:, convert(i), convert(j+2)), 0);
                        vSofi(i, j) = ((aboveBelow + leftRight) / 2);
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
        horizPlusVirtMean = mean(reshape(vSofi(3:2:height*2-2, 3:2:width*2-2), 1, []));
        horizPlusVirtStd = std(reshape(vSofi(3:2:height*2-2, 3:2:width*2-2), 1, []));
        horizonMean = mean(reshape(vSofi(3:2:height*2-2, 2:2:width*2-2), 1, []));
        horizonStd = std(reshape(vSofi(3:2:height*2-2, 2:2:width*2-2), 1, []));
        virtMean = mean(reshape(vSofi(2:2:height*2-2, 3:2:width*2-2), 1, []));
        virtStd = std(reshape(vSofi(2:2:height*2-2, 3:2:width*2-2), 1, []));
        diagMean = mean(reshape(vSofi(2:2:height*2-2, 2:2:width*2-2), 1, []));
        diagStd = std(reshape(vSofi(2:2:height*2-2, 2:2:width*2-2), 1, []));
    
        aHorizon = horizPlusVirtStd / horizonStd;
        bHorizon = horizPlusVirtMean - aHorizon*horizonMean;
        aVert = horizPlusVirtStd / virtStd;
        bVert = horizPlusVirtMean - aVert*virtMean;
        aDiag = horizPlusVirtStd / diagStd;
        bDiag = horizPlusVirtMean - aDiag*diagMean;
    
        vSofi(3:2:height*2-2, 2:2:width*2-2) = aHorizon * vSofi(3:2:height*2-2, 2:2:width*2-2) + bHorizon;
        vSofi(2:2:height*2-2, 3:2:width*2-2) = aVert * vSofi(2:2:height*2-2, 3:2:width*2-2) + bVert;
        vSofi(2:2:height*2-2, 2:2:width*2-2) = aDiag * vSofi(2:2:height*2-2, 2:2:width*2-2) + bDiag;
    end


    %% Image Normailzation and Creation
    if doNormalCross
        sofi = sofi(2:height-2, 2:width-2);
        sofi = sofi - (min(min(sofi)));
        sofi = sofi ./ (max(max(sofi)));

        sofi(sofi > satMax) = satMax;
        sofi = sofi ./ (max(max(sofi)));
        
        crossSofi = sofi;
    else

    vSofi = vSofi(2:height*2-2, 2:width*2-2);
    vSofi = vSofi - (min(min(vSofi)));
    vSofi = vSofi ./ (max(max(vSofi)));

    vSofi(vSofi > satMax) = satMax;
    vSofi = vSofi ./ (max(max(vSofi)));

    crossSofiVirt = vSofi;
    end

end

% Converts a virtual index into a real index
function [i] = convert(vi)
    i = (vi + 1) / 2;
end
