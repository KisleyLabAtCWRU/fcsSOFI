This README gives a brief over view of how to use the fcsSOFI script.

Data Prepreation:

Your data must come in the form of tiff files or mat files. (If using tiff, must assign useTiffFile = 1)
You can use multiple files imaging the same point, the code will string them together.
If you have ran your data before and have the background corrected (BC) .mat files, this can also be used.
	If you with to run data with different setting, this will save a lot of time


User Input Section:


startloc = The file path to your data

Pixle Size: Size of the pixel (in nm) as defined by physical camera pixel length and objective magnification.
PSFsample: The FWHM (in pixels) of the sample's PSF.
dT: The time in seconds between each image of the tiff file.
sigma: Calculated from the PSFsample, used for deconvelution of sofi image.
numberFiles: The number of tiff files being combined together.
framesLength: The length of each of the tiff files being combined together. (Must be uniform)

Region of Interest Settings: The data will be truncated to only look at this portion.
	tmin and tmax are of all tiff files added together. (Three 10000 frames tiff files, tmax = 30000)

Type: The type of diffusion to fit too
	1 = Brownian, 2 = 2-Comp Brownian, 3 = Anomalous, ...
      4 = Brownian 1 Comp with tau, 5 = 1-comp Brownian with tau and A, ...
      6 = Anomalous with Tau and alpha)

A_stp: Starting fit value for amplitude
alpha_stp: Starting fit value for alpha
D_stp: Starting fit value for diffusion coefficient
D2_stp: Starting fit value for second diffusion coefficient 
alpha_min/max: alpha threshold where alpha_max is the maximum value of alpha allowed to appear on alpha map

diffusionMin/Max: Set oritifitail limits to the logrithmic diffusion values...
			 Used to exclude clearly wrong diffusion values.

number_fits: Number of fit iterations per pixel

plotFigures: Plots variouse figures
	Figure 1: Histogram of diffusion values
	Figure 2: 
		Top Left: Average image over time
		Top Right: sofi image with no deconvolution
		Bottom Left: sofi image with deconvolution
		Bottom Right: line sections at row 25 of all three plots
					Blue: Average
					Red: sofi no decon
					Black: sofi with decon
	Figure 3:
		Top Left: Map of tauD values
		Top Right: Diffusion Map
		Bottom Left: Diffusion Map with poor fits removed
		Bottom Right: Diffusion on a log10 scale
	Figure 4: If doing 2-comp diffusion, same as last figure for second diffusion
	Figure 5: A map of the fit R Squared results
	Figure 6: The log10 of the diffusion values
	Figure 7: The final sofi Image
	Figure 8: The combined fcsSOFI image
	Figure 9: The single pixle diffusion fit for the selected pixle

store_execution_times:
savethedata: Saves the folling files in a new folder: (the .mat convered file is always saved if using tiff file)
	The background corrected .mat file
	All the figures plotted in a .fig file
	A selection of variables in .mat form
		fit curves, fit parameters, sofi map, diffusion maps, r squared map, alpha map

examplecf: Plots a curve fit for a single chossen pixle. Also prints the values found for this pixle
doDecon: deconvelution on the sofi image can be skiped if it is found to remove too much infromation
satmin/max: Caps the min and max of the sofi image, then renormalized it...
	Focuses in on different sections if they are not bright enough or too bright
doFileConversionAndBackSubtraction: If useing the BC .mat file from a previouse run, change to 0
	This speeds the code up signifigantly.
Color selctions:
	This picks the color scale used for all the diffusion images.
  











