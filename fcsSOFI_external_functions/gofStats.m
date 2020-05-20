%% Goodness of fit statistics and uncertainties for single pixel fits
function [rsq,chisq,J,MSE,ci] = gofStats(type,parameters,fitresult,x,y)

%% Residuals
residuals = y - fitresult;
a = (y - fitresult).^2./fitresult;
a(isinf(a)) = 0;

%% Chi-square
chisq = sum(a);

%% R-square
rsq = 1-sum(residuals.^2)/sum(y.^2);

%% Partial Derivatives and Jacobian Matrix
% Brownian
if type == 1
    % partial derivatives
    dGda = parameters(3)./(parameters(3)+x);
    dGdb = ones(size(x));
    dGdtau = (parameters(1).*x)./(parameters(3)+x).^2;
    
    % Jacobian
    J = [dGda dGdb dGdtau];J = reshape(J,[length(x),length(parameters)]);
    
% 2-Componenet Brownian
elseif type == 2
    % partial derivatives
    dGda1 = parameters(4)./(parameters(4)+x);
    dGda2 = parameters(5)./(parameters(5)+x);
    dGdb = ones(size(x));
    dGdtau1 = (parameters(1).*x)./(parameters(4)+x).^2;
    dGdtau2 = (parameters(2).*x)./(parameters(5)+x).^2;
    
    % Jacobian
    J = [dGda1 dGda2 dGdb dGdtau1 dGdtau2];J = reshape(J,[length(x),length(parameters)]);
    
% Anomalous
elseif type == 3
    %partial derivatives
    dGda = 1./((x./parameters(3)).^(parameters(4)+1));
    dGdb = ones(size(x));
    dGdtau = parameters(1)*parameters(4).*(x./parameters(3)).^(parameters(4))./(parameters(3).*((x/parameters(3)).^(parameters(4))).^2);
    dGdalpha = (parameters(1)*(x./parameters(3)).^(parameters(4)).*log(x./parameters(3)))./((x./parameters(3)).^(parameters(4)+1)).^2;

    % Jacobian
    J = [dGda dGdb dGdtau dGdalpha];J = reshape(J,[length(x),length(parameters)]);
    
end

%Mean squared error
MSE = sum((y - fitresult).^2)/length(y); 

%% Confidence intervals

ci = nlparci(parameters,residuals,'jacobian',J);

end