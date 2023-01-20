# fcsSOFI

This repository hosts and tracks current progress on the Kisley Lab fcsSOFI update project. Please cite: doi:10.1021/acsnano.5b03430.

fcsSOFI combines the correlation techniques of fluorescence correlation spectroscopy (FCS) and super-resolution optical fluctuation imaging (SOFI) to produce to produce super-resolution spatial maps of diffusion coefficients. This has been applied thus far to understand diffusion and spatial properties of porous and complex materials, including agarose and PNIPAM hydrogels, liquid crystals, and liquid-liquid phase-separated polymer and biomolecule condensates. 

More details can be found at:
Kisley, L.; Higgins, D.; Weiss, S.; Landes, C.F.; et al. Characterization of Porous Materials by Fluorescence Correlation Spectroscopy Super-resolution Optical Fluctuation Imaging. ACS Nano 2015, 9, 9158–9166. doi:10.1021/acsnano.5b03430. PMID 26235127.

**Files and folders maintained on this repository:**
* README: Description of the fcsSOFI repository (README.md - the file you are reading now!)
* fcsSOFI User Guide - a short guide for the fcsSOFI script explaing user inputes (fcsSOFI_user_guide.pdf)
* GPU-accelerated fcsSOFI script (fcsSOFI_script.m)
* Test data set: simulation of two pores with emitters undergoing Brownian diffusion (dataset77.mat)
* Folder containing external functions called in script (fcsSOFI_External_Functions)
* Customized Gpufit library (Gpufit_build-64-YYYYMMDD)

The GPU-accelerated portions of the GUI and script use Gpufit, an open source software library for GPU-parallelized curve fitting developed by Mark Bates, Adrian Przybylski, Björn Thiel, and Jan Keller-Findeisen. The relevant 2017 Scientific Reports publication describing Gpufit can be found here: https://www.nature.com/articles/s41598-017-15313-9#Ack1 and the Gpufit homepage: https://github.com/gpufit/Gpufit
