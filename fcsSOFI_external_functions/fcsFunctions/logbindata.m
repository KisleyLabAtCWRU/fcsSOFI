%{
logbindata.m

Function to bin data logarithmically

lags - old lags
new_lags - lags after binning
AC_aver - old AC values
new_AC - AC values after binning

Takes data over many time lags, and averages the data into bins for fewer
points. The Bins range from the smallest to largest log10 values. There are
10 bins per decade.

AC_aver can be a 2D array which contains multiple data sets. Different data
sets need to be in different columns. The binning will take place in the
1st dimension. Aka, AC_aver needs to be (points x datasets)

=========================================
Aug 2, 2007               Alexei Tcherniak
=========================================
% Modified to comply with NI Board
April 29, 2011            Sergio Dominguez
==========================================
% Modified to work in two dimensions
July 31, 2023            Benjamin Wellnitz
==========================================
%}
function [new_lags, new_AC] = logbindata(lags,AC_aver,ddwell,max_lag)

% determine the lowest and higher order
low_flag = 0;
hi_flag = 0;
    
for i=1:10 
    if (ddwell/10^(-i) >= 0.999) && low_flag == 0
        low_order = -i; % find the lowest order of magnitude in the time scale
        low_flag = 1;
    end
    
    if max_lag > 1
        if (max_lag/10^(i) <= 1) && hi_flag ==0
            hi_order = i; % find the highest order of magnitude in the time scale
            hi_flag = 1;
        end
    elseif max_lag < 1 % in case the max time is less than 1 s
        if (max_lag/10^(-i) >= 0.9999) && hi_flag ==0
            hi_order = -i+1; % find the highest order of magnitude in the time scale
            hi_flag = 1;
        end
    else
        'oops, something is wrong with max_lag'
        'exiting'
    end
end

% ['low_order', 'hi_order']
[low_order, hi_order];

% compute the number of bin before the next order
n_first_bins = int32((10^(low_order+1))/(lags(1)*ddwell)) - 1;
% n_first_bins = int32((10^(low_order+1))/(lags(1))) - 1;
% n_first_bins = 10;


% compute the total number of bins
nbins = n_first_bins + 9*(hi_order - (low_order + 1));

% return

new_lags = zeros(nbins,1);
new_AC = zeros(nbins, size(AC_aver, 2));
npoints_per_bin = zeros(hi_order-(low_order+1),1);

% fill the bins until the next order
for ipoint=1:n_first_bins
    new_lags(ipoint) = lags(ipoint);
    new_AC(ipoint, :) = AC_aver(ipoint, :);
end

flag_stop = 0; % is used to stop filling the bins once the end of lags array is reached


% start the loop to fill the rest of the bins
for iorder = 1:(hi_order-(low_order+1))
    order = double(low_order + iorder); % else it can't do 10^(-int)
    npoints_per_bin(iorder) = int32(10.0^order/(lags(1)*ddwell));
%     npoints_per_bin(iorder) = int32(10.0^order/(lags(1)));
    
    % loop over bins inside the order
    for ibin = 1:9
        bin_indx = n_first_bins + (iorder-1)*9 + ibin;

        % if there is not enough points to fill the last bin, average by
        % the number of points left to the end of 'lags'
%         norm = min((length(lags)-bin_indx),npoints_per_bin(iorder));
        
        % loop over the points for this bin
        for ipoint = 1:npoints_per_bin(iorder)
        
            tmp_indx = n_first_bins;
            for jorder = 1:(iorder-1)
                tmp_indx = tmp_indx + 9*npoints_per_bin(jorder);
            end
            tmp_indx = tmp_indx + (ibin-1)*npoints_per_bin(iorder) + ipoint;
            
%             if ipoint == 1
%                 [bin_indx, npoints_per_bin(iorder), tmp_indx]
%             end
            
            if (tmp_indx < length(lags))
                point_indx = tmp_indx;
            else
                flag_stop  = 1;
                break
            end
            
           
%             new_lags(bin_indx) = new_lags(bin_indx) + lags(point_indx)/norm;
%             new_AC(bin_indx) = new_AC(bin_indx) + AC_aver(point_indx)/norm;
% 'point_indx, tmp_indx'
% point_indx, tmp_indx
            new_lags(bin_indx) = new_lags(bin_indx) + lags(point_indx)/npoints_per_bin(iorder);
            new_AC(bin_indx, :) = new_AC(bin_indx, :) + AC_aver(point_indx, :)./npoints_per_bin(iorder);
        end %loop over points
        
        if (flag_stop == 1)
            break
        end
        
    end % loop over bins
    
    if (flag_stop == 1)
        break
    end
    
end % loop over orders


% % rid new_lags of zero values
new_lags_rep = repmat(new_lags, 1, size(AC_aver, 2)); % Make copy to logical index AC
new_AC = new_AC(new_lags_rep ~= 0); % Logical Inddex AC
new_lags = new_lags(new_lags ~= 0); % Logical Index lags
new_AC = reshape(new_AC, length(new_lags), size(AC_aver, 2)); %Logical Index leaves linear, must reshape



%  figure
%  hold on
%  %plot(new_lags*ddwell,new_AC,'o','MarkerSize',12) %%very important to multiply new_lags*ddwell !
%  %plot(new_lags,new_AC,'o','MarkerSize',12) %%very important to multiply new_lags*ddwell !
%  set(gca,'XScale','log')
% 
%  %% if you want to compare to previous data
% hold on
% plot(lags,AC_aver,'o','MarkerSize',10,'Color','r')
% plot(new_lags,new_AC,'o','MarkerSize',12, 'Color', 'b') %%very important to multiply new_lags*ddwell !
% set(gca, 'FontSize', 20)
% 
%  legend('log bins','default bin')


