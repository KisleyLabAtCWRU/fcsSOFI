function thd_map = LRG_SuperRes_LocalThrMap(im, snrEnhance)
    im = double(im);
    if snrEnhance
        im = SNR_booster(im);
    end
    n = 3; % how many std to add up as a threshold
    w = 20; % usually 20X20 and shift by 10 is a good choice for a 200X200 image
    [v, h] = size(im);
    count = zeros(v, h);% count store the times each pixel has contributed in local
    % background calculation
    bg = count; % record the local background
    sd = count; % record the local standard deviation
    
    for i = 1 : ceil(v / w * 2) - 1
        for j = 1 : ceil(h / w * 2) - 1
            % 1, select the local region
            % 2, sort the pixels based on their intensities
            % 3, find the 50% and 75% point as discussed in the paper
            % 4, calculate local background(bg), standard deviation(sd) and
            % count
            im_local = im(1+w/2*(i-1):min(v,w/2*(i+1)), 1+w/2*(j-1):min(h,w/2*(j+1)));
            im_local = sort(im_local(:));
            n_loc = numel(im_local);
            bg(1+w/2*(i-1):min(v,w/2*(i+1)), 1+w/2*(j-1):min(h,w/2*(j+1))) =...
                bg(1+w/2*(i-1):min(v,w/2*(i+1)), 1+w/2*(j-1):min(h,w/2*(j+1))) +...
                im_local(round(n_loc/2));
            sd(1+w/2*(i-1):min(v,w/2*(i+1)), 1+w/2*(j-1):min(h,w/2*(j+1))) =...
                sd(1+w/2*(i-1):min(v,w/2*(i+1)), 1+w/2*(j-1):min(h,w/2*(j+1))) +...
                im_local(round(n_loc*0.5)) - im_local(round(n_loc*0.18));%sd = 0.82 to 0.5 of cumulative distribution
            count(1+w/2*(i-1):min(v,w/2*(i+1)), 1+w/2*(j-1):min(h,w/2*(j+1))) =...
                count(1+w/2*(i-1):min(v,w/2*(i+1)), 1+w/2*(j-1):min(h,w/2*(j+1))) + 1;
        end % for j
    end % for i
    bg = bg ./ count;
    sd = sd ./ count;
    thd_map = bg + n * sd;% determine the local threshold
    return
end