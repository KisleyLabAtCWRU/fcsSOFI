function [crossSofi] = crossSofi(data, doNormalCross)


    %% Variable Set Up
    sz = size(data);
    height = sz(1);
    width = sz(2);
    data = permute(data, [3, 1, 2]);


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
