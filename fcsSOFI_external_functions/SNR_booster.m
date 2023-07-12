function im_out = SNR_booster(frame)
    [v, h] = size(frame);
    ex_w = 3;% we always use a 3X3 matrix to convolute with the raw image.
    % to use Fourier transform for convolution, the mask need to has the same size as the image.
    mask = zeros(v, h);
    mask(ceil(v/2):ceil(v/2)+ex_w-1,ceil(h/2):ceil(h/2)+ex_w-1) = ones(ex_w, ex_w);
    % Convolution
    ft_mask = fft2(mask);
    ft_im = fft2(frame);
    im_out = real(fftshift(ifft2(ft_mask .* ft_im))) / ex_w^2;
end