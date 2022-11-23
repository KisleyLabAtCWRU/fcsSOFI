function [Data1] = TiffReadRM(filekey, filepath, startframe, endframe)
    fname = strcat(filepath, filekey);
    i = 1; %counter
    for j = startframe:endframe
        Data1(:, :, i) = imread(fname, j);
        i = i + 1;
    end
    return
end