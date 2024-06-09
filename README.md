# insar_surface-def_tools
Random post-processing scripts for loading and manipulating InSAR data with python


The data and scripts in this directory are designed to deal with SAR offset and InSAR data in Python

The main functions used are in surfacedef2los.py
Example Scripts are loose as jupyter notebooks (.ipynb) in the main dir. 

A brief description of each directory 
archived_scripts : can ignore    
contrib_burgi : Sentinel-2 optical pixel offset tiffs provided by Paula   
contrib_melgar_2023_stations : station locations of the data used in Melgar et al., 2023 inversion 
contrib_usgs-products : usgs slip inversion products	    
cutde : this dir is a bit disorganized but contains the scripts and inputs used to get from a slip model to a surface deformation (.disp) flat file
observation-files : this direcotry contains all of the observation files used
optical-pixel-offsets.ipynb
plots : this is where plots are written to. Naming convention is chaotic. Sorry!
surface-displacement-files : these are the surface displacement files calculated from various slip models using cutde
topo : some dems used for the phasegradient base maps in pygmt
