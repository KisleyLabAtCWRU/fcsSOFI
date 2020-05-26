# GPU-Accelerated fcsSOFI GUI

Version 1.9: 22 May 2020

Based on Kisley, L.; Higgins, D.; Weiss, S.; Landes, C.F.; et al. Characterization of Porous Materials by Fluorescence Correlation Spectroscopy Super-resolution Optical Fluctuation Imaging. ACS Nano 2015, 9, 9158–9166. doi:10.1021/acsnano.5b03430. PMID 26235127.

Gpufit is an open source software library for GPU-parallelized curve fitting developed by Mark Bates, Adrian Przybylski, Björn Thiel, and Jan Keller-Findeisen. The relevant 2017 Scientific Reports publication describing Gpufit can be found here: https://www.nature.com/articles/s41598-017-15313-9#Ack1

Gpufit homepage: https://github.com/gpufit/Gpufit

**Instructions for Use**

1) Prepare Data for Analysis

* For now, the fcsSOFI GUI only accepts data packaged in .mat files. I recommend manipulating and saving your data
in MATLAB and not some other data processing environment (Excel for instance). 

* Your data must be in the format of a 3-dimensional single or double array (i.e. 512x512x1000 single). This array
should form a "movie" which fcsSOFI can analyze pixel-by-pixel and frame-by-frame over time.
The first two dimensions of your array should countain the spatial information of a given frame (the image)
and the last dimension should order each frame in time. Each frame should be separated by the same time step dT.

NOTE: To be analyzed by this preliminary GUI, your 3D array must be named 'Data', the default name for an imported array in MATLAB.

*I am working to make the code agnostic to the specific array name, but for now, the array must have the name
'Data'. If this is not the case, redefine your movie as 'Data' in MATLAB and save your .mat file again. Your
.mat file can contain other arrays/components/data than just your 'Data' array - the GUI extracts the array 'Data' automatically from any .mat file.*


2) Open GUI

* Simply double click on the 'fcsSOFI_GUI_WSYYMMDD' MATLAB App file. Assuming you have MATLAB 
installed on your computer, the GUI should open and be immediately ready for use.

NOTE: Make sure the GUI (fcsSOFI_GUI_WSYYMMDD.mlapp), the custom Gpufit library folder (Gpufit_build-64_20190709), and your 
data exist in the same directory, and make that you are working in that directory in MATLAB.

3) Load Data 

* Data can be loaded in two ways: either with the "Browse" button or the "Manual Load" text-input boxes.
The Browse button opens your file explorer, from which you can select a .mat data file. If you know the 
file name (and the file is in the same directory as the fcsSOFI GUI), you can manually input that 
name as a combination of the "File name ext." and "Trial number(s)" inputs. 
For example, the inputs
	File name ext. = 'dataset'
	Trial number(s) = 4
Loads the file
	'dataset4.mat'
The manual input is intended to allow the analysis of multiple data sets at once. This functionality
is limited/full of bugs as of March 2020 and is not reccomended.

* After you load the data with either 'Browse' or 'Load Data' buttons, the first frame of your 'Data1' array 
should appear on the GUI axis. From here you can choose a specific region of interest (ROI) with the input boxes
under 'Region of Interest in Pixels.' The x and y inputs crop the image on a specific range of pixels, and 
the t inputs select specific frames. Click 'Display ROI' to display the ROI on the first frame.

* The 32x32x5000 simulated data file 'dataset77.mat' is included in this repository. The default inputs to
'File name ext.' and 'Trial number(s)' are already set to load 'dataset77.mat'. As long as your MATLAB path is 
set to the extracted zip folder, you can just press 'Run' right away and the GUI will load and analyze the array
contained in 'dataset77.mat'.

4) Run fcsSOFI analysis

* Once data is loaded, simply press the 'Run' button and the fcsSOFI analysis will commence. Once complete, the
fcsSOFI GUI will print execution times in the top white panel and the combined fcsSOFI image will display on the
GUI axes. Toggle through the figures with the 'Display Figure' dropdown menu below the axes.

**GUI Inputs/Components**

* Load Data
'Browse' button: opens file explorer and allows user to select a .mat data file

* Manual Load
'File name ext.' text input: specifies generic file name common to your data files (i.e 'dataset')
'Trial number(s) text input: specifies data file corresponding to specific trial number, will soon be able to handle multiple trials at once

* Region of Interest in Pixels
'xmin' text input: minimum horizontal position/pixel of ROI
'xmax' text input: maximum horizontal position/pixel of ROI
'ymin' text input: minimum vertical position/pixel of ROI
'ymax' text input: maximum vertical position/pixel of ROI
'tmin' text input: first frame to analyze
'tmax' text input: last frame to analyze

* Microscopy Parameters
'Pixel size (nm)' text input: physical pixel size of microscope in nm
'dT (s)' text input: time interval between each captured frame in seconds

* C-axis Scale (for Simulated Data)
'Min scale' text input: minimum scale for C-axis
'Max scale' text input: maximum scale for C-axis

* 'Diffusion Type' button group: select diffusion curve-fitting model
	'Brownian' button: single-component brownian diffusion
	'2-Comp Brownian' button: two-component brownian diffusion
	'Anomalous' button: anomalous diffusion 

* 'Alpha start point' text input: choose start point for anomalous stretch factor alpha (anomalous diffusion model only)

* 'Save figures?' button group: select yes to save figures as .png files

* 'Save data?' button group: select yes to save analyzed data in a .mat file

* 'Run' button: run fcsSOFI analysis for current text/button inputs

* Top text area: prints completed actions, curve fitting progress (25% to 100%), and execution times

* Axes: displays figures

* 'Display Figure' dropdown menu: once fcsSOFI analysis is complete, toggle between result figures

* Single-Pixel Fit Results
'Row Index' text input: select the row index of the desired pixel
'Column Index' text input: select the column index of the desired pixel
'Display' button: display visual curve fit results on GUI axis and displays curve-fit diffusion coefficient estimates, 
relevant parameters, and g.o.f indicators

* Top right text area: displays curve fit diffusion coefficient estimates, relevant parameters, and g.o.f indicators
