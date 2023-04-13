%% new super-resolution images 20130130
function [superdata] = LRG_SuperRes_GenerateSR(LocatStore,e)
rsigma=e.sigmarad;
% rsigma = 0 using Thompson equation for sigma radius. Otherwise use the
% value in nm as sigma radius for the shape.
nzoom = e.nzoom;
ymax=e.ymax;ymin=e.ymin;xmax=e.xmax;xmin=e.xmin;
% probeDistance = 2; % if the distance larger than 2 pixels, keep both.
% spotsigma = e.sr_sigmathr/e.pixelSize*nzoom; % if the event uncertainty is smaller than e.sr_sigmathr nm, keep it.
alllocat = [];  % get all identified locations in all frames from LocatStore with [y,x,a1], where a1 is the amplitude/intensity of the 2D peak

for f = 1:e.nframes
    if isempty(LocatStore(1,f).PSFfinal)==0
        for i = 1:size(LocatStore(1,f).PSFfinal(:,1),1)
%           if LocatStore(1,f).PSFDataReduced(i,5)>e.bkMean 
             alllocat = [alllocat;LocatStore(1,f).PSFfinal(i,1:6)];   
                %  [y,x,sy,sx,a1,superSigma, flag],  a1 is not used at 09/24/2012 but might be
                %  used later to determin the uncertainty of the center of
                %  each events, since delta_x is proportional to sqrt(a1).
%           end
        end
     end
end

% sls = [];

% for i = 1:size(alllocat(),1)
% %     slSigmax = (alllocat(i,3)+alllocat(i,4))/2;
% %     slN = alllocat(i,5)*2*pi*alllocat(i,3)*alllocat(i,4);
% %     sla = 1;
% %     slbkSigma = e.bkSigma;
% %     % slbkSigma = sla*sqrt(slN)/2/slSigmax; %2011 Nanolett Simanson,~50% smaller than the measured noise 
% %     % sls = [sls; slbkSigma];
% %     alllocat(i,6) = sqrt(slSigmax^2/slN+sla^2/12/slN+8*3.1415*slSigmax^4*slbkSigma^2/sla^2/slN^2); % 2002 Biophys.J. p2775 Thompson equation
% %     % for j = 1:size(alllocat(),1)
%         if alllocat(i,6)*nzoom<=spotsigma
%           %  if abs(alllocat(i,3)/alllocat(i,4)-1.0)<= e.xyratio % x,y ratio of original within 0.7 -1.3
%       %      if i~=j 
%                % if probeDistance > sqrt((alllocat(i,1)-alllocat(j,1))^2+(alllocat(i,2)-alllocat(j,2))^2) 
%                % locations that have neighbours within probeDistance pixels. 
%                     alllocat(i,7) = 1;
%                % end
%        %     end
%           %  end
%         end
%     % end
% end

% ratiohist =alllocat(:,3)./alllocat(:,4);
% %[Mean,Sigma] = JC_1DGaussianHist(data,histnum,height,center,sigma)
% [Mean,Sigma] = JC_1DGaussianHist(ratiohist,20,2500,1,0.12)
% figure; hist(ratiohist(:),20); title(['xy ratio, std=',num2str(Sigma)]);
% % alllocat2 = []; 
% % 
% % % This is to filter out places that only have one event that only last one frame.
% % for i = 1:size(alllocat(),1)
% %    if alllocat(i,7)== 1
% %        alllocat2 = [alllocat2; alllocat(i,:)];
% %    end
% % end

       
% Initialize a matrix with much higher pixel resolution. eg. nZoom =10, each old pixel is devided into 10x10 new pixels    
% nzoom =20; % ratio of pixel size between regular image and super-resolution image.
superdata= zeros((ymax-ymin+1)*nzoom,(xmax-xmin+1)*nzoom);  % size of the original image * nzoom.


% put down the 2D gaussian shape at each location. Let the close events
% overlaping with each other.
for i = 1:size(alllocat,1)
    newy0 = round(alllocat(i,1)*nzoom);  % new center y0
    newx0 = round(alllocat(i,2)*nzoom);  % new center x0
    
    % regenerage a 2D Gaussian shape for each event with sigma = number of pixel 
    % a1*exp(-(x-x0).^2/(2*sigmax^2)-(y-y0).^2/(2*sigmay^2))+b1
    radius = 1*nzoom; sigmax = rsigma*nzoom; sigmay = rsigma*nzoom; a1 = 1; b1 = 0; x0 = radius+1; y0 = radius+1;
    % radius is the size of the shape, should be larger than 3*sigma. 
    % If nzoom =10, the sigma is ~<2 pixels determined from a single event
    % dwelling for miltiple frames.  The amplitude of the shape a1 is set to 1 now.
    % So the whole peak intensity is sum(shape(:)) = 2*pi*a1*sigmax*sigmay.
    if rsigma ~= 0
       sigmax = rsigma/e.pixelSize*nzoom; sigmay = rsigma/e.pixelSize*nzoom;  %%%%%% the average size set to be 30 nm
    end
    % This overload the sigma calculated with the previous algorithm
    
    shape = zeros(2*radius+1);
    for i = 1:2*radius+1
        for j = 1:2*radius+1
            shape(i,j)= a1*exp(-(i-x0)^2/(2*sigmax^2)-(j-y0)^2/(2*sigmay^2))+b1;
        end
    end

    % superdata(x-radius:x+radius,y-radius:y+radius) = superdata(x-radius:x+radius,y-radius:y+radius)+shape.*alllocat(i,3);
    if newy0-radius>0 && newy0+radius < (e.ymax-e.ymin)*nzoom && newx0-radius>0 && newx0+radius<(e.xmax-e.xmin)*nzoom
        superdata(newy0-radius:newy0+radius,newx0-radius:newx0+radius) = superdata(newy0-radius:newy0+radius,newx0-radius:newx0+radius)+shape;
    end 
end
% figure; imagesc(superdata); colormap(jet);  title(['Super-resolution image of all events: ',e.filelist]); % make the 2D image of the super-resolution map.

%% Find an event
% obj_y = 205;
% obj_x = 92;
% obj_r = 10;
% for f = 1:e.nframes
%     if isempty(LocatStore(1,f).PSFDataReduced)==0  % start to consider the real intensity that it should be
%         for i = 1: size(LocatStore(1,f).PSFDataReduced(:,1),1)
%             r = 0;
%             r = (obj_y-LocatStore(1,f).PSFDataReduced(i,1))^2+(obj_x-LocatStore(1,f).PSFDataReduced(i,2))^2;
%             if r <= obj_r 
%                 %draw the image
% %                 figure;
% %                 imagesc(Data1(obj_y-obj_r:obj_y+obj_r,obj_x-obj_r:obj_x+o
% %                 bj_r, f)); hold on;
% %                 plot(LocatStore(1,f).PSFDataReduced(i,2)-obj_x+obj_r+1,LocatStore(1,f).PSFDataReduced(i,1)-obj_y+obj_r+1,'gx','MarkerSize',10,'LineWidth',2)  %Plot little, thick, blue circles around the ones kept
% %                 text(double(LocatStore(1,f).PSFDataReduced(i,2)-obj_x+obj_r+2),double(LocatStore(1,f).PSFDataReduced(i,1)-obj_y+obj_r),[num2str(round(LocatStore(1,f).PSFDataReduced(i,2))),' ',num2str(round(LocatStore(1,f).PSFDataReduced(i,1)))]);
% %                 title(['frame#',num2str(f),'/',num2str(size(LocatStore,2))]);
%                 
% %                 figure; imagesc(Data1(:,:,f)); hold on;
% %                 title(['frame#',num2str(f),'/',num2str(size(LocatStore,2))]);               
% %                 %plot(LocatStore(1,f).PSFDataReduced(i,2),LocatStore(1,f).PSFDataReduced(i,1),'gx','MarkerSize',10,'LineWidth',2)  %Plot little, thick, blue circles around the ones kept
% %                 text(double(LocatStore(1,f).PSFDataReduced(i,2)+2),double(LocatStore(1,f).PSFDataReduced(i,1)),[num2str(round(LocatStore(1,f).PSFDataReduced(i,2))),' ',num2str(round(LocatStore(1,f).PSFDataReduced(i,1)))]);
%             
%                 figure; 
%                 imagesc(Data1(196:216,81:103,f)); hold on;
%                 plot(LocatStore(1,f).PSFDataReduced(i,2)-81.0+1.0,LocatStore(1,f).PSFDataReduced(i,1)-196.0+1.0,'gx','MarkerSize',10,'LineWidth',2)  %Plot little, thick, blue circles around the ones kept
%                 title(['frame#',num2str(f),'/',num2str(size(LocatStore,2))]);
%                 text(double(LocatStore(1,f).PSFDataReduced(i,2)-81.0+2.0),double(LocatStore(1,f).PSFDataReduced(i,1)-196.0),[num2str(round(LocatStore(1,f).PSFDataReduced(i,2))),' ',num2str(round(LocatStore(1,f).PSFDataReduced(i,1)))]);
%                
%             end
%         end
%     end
% end 
return