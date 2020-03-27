function [selectedpts] = getUpperLeft(length, width, startingpt)

selectedpts = [];
for n = 1:width-1
    newpts = startingpt+(n-1)*length:startingpt+n*length-2;
    
    selectedpts = [selectedpts newpts];
    n = n+1;
end



end