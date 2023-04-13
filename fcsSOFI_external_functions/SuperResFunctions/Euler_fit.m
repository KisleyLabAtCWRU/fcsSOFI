function [xc, yc, varargout] = Euler_fit(im, x_max, y_max)
%% Euler gradient fitting
% im is the fitting region
% (x_max, y_max) is the inital guess of the center
%% calculate the gradient at each spot
[v h] = size(im);
dIdu = im(1:v-1,2:h)-im(2:v,1:h-1);
dIdv = im(1:v-1,1:h-1)-im(2:v,2:h);
flt = ones(3)/9;  % simple 3x3 averaging filter
fdu = conv2(dIdu, flt, 'same');
fdv = conv2(dIdv, flt, 'same');
m_cop = -(fdv+fdu)*1j -(fdv-fdu);
Am = (abs(m_cop));
phi = angle(m_cop);
xm_onerow = [1.5:h-0.5];
xm = xm_onerow(ones(v-1, 1), :);
ym_onecol = [1.5:v-0.5]';
ym = ym_onecol(:,ones(h-1,1));

%% Least Squares fit
    %options = optimset('MaxFunEvals',1000,'MaxIter',1000, 'TolX', 0.01);
    options = optimset('TolX', 0.001);
    % Optimize the LS fit
    coefvalues = fminsearch(@(coefs) diff_ang(Am, xm, ym, phi, coefs),[x_max, y_max],options);

%% the function to calculate the fitting potential
    function psi = diff_ang(Am, xm, ym, phi, coefs)
        xc = coefs(1);
        yc = coefs(2);
        m_c = (xc-xm) + (yc-ym)*1j;
        phi_c = angle(m_c);
        delta = -1./(pi+0.5-abs(abs(phi_c(:) - phi(:))-pi));
        psi = sum(sum(Am(:).*delta));
    end
       
    xc = coefvalues(1);
    yc = coefvalues(2);
    
%% The rest of the parameters
% use the standard methods, has nothing to do with this Euler_fit
% std of x and y are defined as the second momentum: sum((x-xc).^2.*I)/sum(I)
[X, Y] = meshgrid(1:h, 1:v);
I = im - min(im(:));
% not using Gaussian fitting, the the width(sig_x, sig_y) are not
% trustable, they could change a lot depending on the size of fitting
% region
try
sig_x = sum(sum((X-xc).^2.*I))/sum(sum(I));
sig_y = sum(sum((Y-yc).^2.*I))/sum(sum(I));
Imax = I(round(yc), round(xc)); % simply take the intensity of the center pixel
sig_I = std(I(:)); % the std of the intensity
varargout = {sig_x, sig_y, Imax, sig_I};
catch ME
end
end