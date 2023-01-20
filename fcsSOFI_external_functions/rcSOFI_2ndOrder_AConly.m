%% rcSOFI - 2nd order correlation function

function [corrCombined]=rcSOFI_2ndOrder(AC,  min_lags,max_lags,lags)

% 2nd order correlation
        AC2 = fft(AC(min_lags:max_lags)); %2nd order
        
        %out of Fourier space
        AC2 = ifft(AC2);       

        % 4X499 correlation data
        corrCombined = [AC2'];

end
        