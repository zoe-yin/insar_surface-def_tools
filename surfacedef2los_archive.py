#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15, 2023

@author: hyin

Description: This file contains functions used to deal with InSAR data (primarily files produced using ISCE2) and modeled surface deformation data.

Notes: This notebook was created and tested in the conda environment 'wasp' on Zoe's USGS laptop

"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pygmt
import os
import xarray as xr
import rioxarray

def surfacedefplot(disp_file,region=None,cpt=None):
    '''
    Parameters
    ----------
    disp_file : string
        Full path to the .disp file
    directory : string, optional
        location of model. The default is None.
    Returns
    -------
    Returns Pygmt figure object
    '''

    ##########################
    #### DISPLACEMENTS     ###
    ##########################
    # Read in cutde displacement file
    #Longitude, Latitude, Elevation, Easting Displacement (m), Northing Displacement (m), Vertical Displacement (m)
    modeled_disp = pd.read_csv(disp_file, sep='\t',header=0, names=['lon','lat', 'elev','disp_e','disp_n','disp_u'])

    
    ##########################
    ####     PLOT     ###
    ##########################

    # Plot modeled surface deformations as a single plot w/ 3 subplots

    ## SET UP  ##
    fig = pygmt.Figure()
    pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT="11p")
    if region == None:
        region = [35.5, 38.5, 36, 38.5]
    if cpt == None:
        cpt = [-3,3,0.1]
    # set the projection to mercator 0/0 projection with 20cm height
    projection = 'M0/0/20c'

    ### Plot the Observed data 
    # obs_da = los_ds.los
    # Plot hillshade
    gradientgrd = '/Users/hyin/yin_usgs/turkiye_2023/surfacedef2LOS/topo/gradients.grd'
    pygmt.makecpt(cmap="gray", series=[-2,0.1, 0.1])
    fig.grdimage(region=region, projection=projection,grid=gradientgrd, cmap=True, nan_transparent=True)

    pygmt.makecpt(cmap="vik", series=cpt)
    fig.basemap(region=region, projection=projection, frame=["a0.5f0.1"])

    fig.plot(
        x=modeled_disp['lon'],
        y=modeled_disp['lat'],
        fill=modeled_disp['disp_e'],
        cmap=True,
        style="c0.3c",
        pen=None,
        transparency=30,
    )

    fig.coast(shorelines=False, region=region, projection=projection, water='204/212/219')
    fig.colorbar(frame="af+lLOS (m)",transparency=30)

    ## MOVE TO NEXT SUBPLOT ##
    fig.shift_origin(xshift="w+6c")

    # ### Plot the Modeled projected data 
    # model_da = a2_a184_mod_proj

    # Plot hillshade
    gradientgrd = '/Users/hyin/yin_usgs/turkiye_2023/surfacedef2LOS/topo/gradients.grd'
    pygmt.makecpt(cmap="gray", series=[-2,0.1, 0.1])
    fig.grdimage(region=region, projection=projection,grid=gradientgrd, cmap=True, nan_transparent=True)

    pygmt.makecpt(cmap="vik", series=cpt)
    fig.basemap(region=region, projection=projection, frame=["a0.5f0.1"])

    fig.plot(
        x=modeled_disp['lon'],
        y=modeled_disp['lat'],
        fill=modeled_disp['disp_n'],
        cmap=True,
        style="c0.3c",
        pen=None,
        transparency=30,
    )

    fig.coast(shorelines=False, region=region, projection=projection, water='204/212/219')
    fig.colorbar(frame="af+lLOS (m)",transparency=30)

    ## MOVE TO NEXT SUBPLOT ##
    fig.shift_origin(xshift="w+6c")

    ### Plot the residual 
    # residual_da = a2_a184_mod_proj_residual
    # Plot hillshade
    gradientgrd = '/Users/hyin/yin_usgs/turkiye_2023/surfacedef2LOS/topo/gradients.grd'
    pygmt.makecpt(cmap="gray", series=[-2,0.1, 0.1])
    fig.grdimage(region=region, projection=projection,grid=gradientgrd, cmap=True, nan_transparent=True)
    pygmt.makecpt(cmap="vik", series=cpt)
    fig.basemap(region=region, projection=projection, frame=["a0.5f0.1"])
    fig.plot(
        x=modeled_disp['lon'],
        y=modeled_disp['lat'],
        fill=modeled_disp['disp_u'],
        cmap=True,
        style="c0.3c",
        pen=None,
        transparency=30,
    )
    fig.coast(shorelines=False, region=region, projection=projection, water='204/212/219')
    fig.colorbar(frame="af+lLOS (m)",transparency=30)    
    
    return fig

    # pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")
    # region = [35.5, 38.5, 36, 38.5]
    # projection = 'M0/0/20c'

    # fig = pygmt.Figure()

    # pygmt.makecpt(cmap="vik", series=[-3, 3, 0.1])
    # fig.basemap(region=region, projection=projection, frame=["a0.5f0.1","+tNorth component"])
    # fig.plot(
    #     x=modeled_disp['lon'],
    #     y=modeled_disp['lat'],
    #     fill=modeled_disp['disp_n'],
    #     cmap=True,
    #     style="c0.3c",
    #     pen=None,
    # )
    # fig.coast(shorelines=True, region=region, projection=projection, water="white")

    # # Shift plot origin of the second map by "width of the first map + 2 cm"
    # # in x direction
    # fig.shift_origin(xshift="w+2c")

    # # Plot 
    # fig.basemap(region=region, projection=projection, frame=["a0.5f0.1","+tEast component"])
    # fig.plot(
    #     x=modeled_disp['lon'],
    #     y=modeled_disp['lat'],
    #     fill=modeled_disp['disp_e'],
    #     cmap=True,
    #     style="c0.3c",
    #     pen=None,
    # )
    # fig.coast(shorelines=True, region=region, projection=projection, water="white")

    # # Shift plot origin of the third map by "width of the second map + 2 cm"
    # # in x direction
    # fig.shift_origin(xshift="w+2c")

    # # Plot 
    # fig.basemap(region=region, projection=projection, frame=["a0.5f0.1","+tVertical component"])
    # fig.plot(
    #     x=modeled_disp['lon'],
    #     y=modeled_disp['lat'],
    #     fill=modeled_disp['disp_u'],
    #     cmap=True,
    #     style="c0.3c",
    #     pen=None,
    # )
    # fig.coast(shorelines=True, region=region, projection=projection, water="white")
    # fig.colorbar(position="JMR+o1c/0c+w20c",frame="af+lDeformation (m)")

    # fig.show()
    # fig.savefig('modeled_disp.jpg', dpi=720)

def residual_plot(obs_da, model_da, residual_da, region=None,cpt=None):
    '''
    Parameters
    ----------
    obs_da : xarray data array object
    model_da : xarray data array object 
    residual_da : xarray data array object
    region : array
    cpt=None
    
    Returns
    -------
    fig : Pygmt figure object 
    '''

    ## SET UP  ##
    fig = pygmt.Figure()
    pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT="11p")
    if region == None:
        region = [35.5, 38.5, 36, 38.5]
    if cpt == None:
        cpt = [-3,3,0.1]
    # set the projection to mercator 0/0 projection with 20cm height
    projection = 'M0/0/20c'

    ### Plot the Observed data 
    # obs_da = los_ds.los
    # Plot hillshade
    gradientgrd = '/Users/hyin/yin_usgs/turkiye_2023/surfacedef2LOS/topo/gradients.grd'
    pygmt.makecpt(cmap="gray", series=[-2,0.1, 0.1])
    fig.grdimage(region=region, projection=projection,grid=gradientgrd, cmap=True, nan_transparent=True)

    pygmt.makecpt(cmap="vik", series=cpt)
    fig.basemap(region=region, projection=projection, frame=["a0.5f0.1"])

    fig.grdimage(
        grid=obs_da,
        projection=projection,
        cmap=True,
        transparency=30
    )

    fig.coast(shorelines=False, region=region, projection=projection, water='204/212/219')
    fig.colorbar(frame="af+lLOS (m)",transparency=30)

    ## MOVE TO NEXT SUBPLOT ##
    fig.shift_origin(xshift="w+6c")

    # ### Plot the Modeled projected data 
    # model_da = a2_a184_mod_proj

    # Plot hillshade
    gradientgrd = '/Users/hyin/yin_usgs/turkiye_2023/surfacedef2LOS/topo/gradients.grd'
    pygmt.makecpt(cmap="gray", series=[-2,0.1, 0.1])
    fig.grdimage(region=region, projection=projection,grid=gradientgrd, cmap=True, nan_transparent=True)

    pygmt.makecpt(cmap="vik", series=cpt)
    fig.basemap(region=region, projection=projection, frame=["a0.5f0.1"])

    fig.grdimage(
        grid=model_da,
        projection=projection,
        cmap=True,
        transparency=30
    )

    fig.coast(shorelines=False, region=region, projection=projection, water='204/212/219')
    fig.colorbar(frame="af+lLOS (m)",transparency=30)

    ## MOVE TO NEXT SUBPLOT ##
    fig.shift_origin(xshift="w+6c")

    ### Plot the residual 
    # residual_da = a2_a184_mod_proj_residual
    # Plot hillshade
    gradientgrd = '/Users/hyin/yin_usgs/turkiye_2023/surfacedef2LOS/topo/gradients.grd'
    pygmt.makecpt(cmap="gray", series=[-2,0.1, 0.1])
    fig.grdimage(region=region, projection=projection,grid=gradientgrd, cmap=True, nan_transparent=True)
    pygmt.makecpt(cmap="vik", series=cpt)
    fig.basemap(region=region, projection=projection, frame=["a0.5f0.1"])
    fig.grdimage(
        grid=residual_da,
        projection=projection,
        cmap=True,
        transparency=30)
    fig.coast(shorelines=False, region=region, projection=projection, water='204/212/219')
    fig.colorbar(frame="af+lLOS (m)",transparency=30)    
    
    return fig

def surfacedef2los(disp_file,look):
    '''
    Parameters
    ----------
    disp_file : string
        Full path to the .disp file
    look : tuple
        center lontiude of region of interest (for simplicity, hypocentral longitude)
        ex: (-0.9845004498094028, 0.17523543456162644, 0.0)
    directory : string, optional
        location of model. The default is None.
    Returns
    -------
    Xarray dataset with all variables, 'lon','lat', 'elev','disp_e','disp_n','disp_u', 'los'
    '''

    ##########################
    #### DISPLACEMENTS     ###
    ##########################
    # Read in cutde displacement file
    #Longitude, Latitude, Elevation, Easting Displacement (m), Northing Displacement (m), Vertical Displacement (m)
    df = pd.read_csv(disp_file, sep='\t',header=0, names=['lon','lat', 'elev','disp_e','disp_n','disp_u'])

    modeled_disp = df.to_array()
    
    # Project the surface def vector (modeled_disp) onto the look vector (e,n,u)
    # Manually calculating the dot product for sanity
    proj = modeled_disp['disp_e']*look[0] + modeled_disp['disp_n']*look[1] + modeled_disp['disp_u']*look[2]

    # Add the projected LOS data to a copy of the modeled_disp dataframe
    modeled_disp_new = modeled_disp.copy()
    modeled_disp_new.insert(6,'los',proj)

    # Convert the dataframe with modeled displacements to xarray
    ds = modeled_disp_new.set_index(["lat", "lon"]).to_xarray()
    # da = ds.to_array()[4]# da.set_coords(("lat", "lon"))
    return(ds)

def plot_grd(da,region=None, cpt=None):
    '''
    Parameters
    ----------
    da : xarray data array or pandas dataframe
        Name of a PyGMT compatible data object (Xarray data array or Pandas dataframe)

    region : array with 4 floats 
        [lon_min, lon_max, lat_min, lat_max]
        Default region is [35.5, 38.5, 36, 38.5] if none is specified
        i.e. region = [35.5, 38.5, 36, 38.5]

    cpt : array with 3 floats 
        [min_value, max_value, step]
        Default cpt is [-3,3,0.1] if none is specified
        i.e. cpt = [-3,3,0.1]
    Returns
    -------
    PyGMT figure object
    
    '''

    # # ##########################
    # # ####     PLOT     ###
    # # ##########################

    # Plot projected synthetic surface deformations
    # set the region
    if region == None:
        region = [35.5, 38.5, 36, 38.5]

    if cpt == None:
        cpt = [-3,3,0.1]
    # set the projection to mercator 0/0 projection with 20cm height
    projection = 'M0/0/20c'
    
    # Plot
    fig = pygmt.Figure()
    pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT="11p")

    # Plot hillshade
    gradientgrd = '/Users/hyin/yin_usgs/turkiye_2023/surfacedef2LOS/topo/gradients.grd'
    pygmt.makecpt(cmap="gray", series=[-2,0.1, 0.1])
    fig.grdimage(region=region, projection=projection,grid=gradientgrd, cmap=True, nan_transparent=True)
    
    pygmt.makecpt(cmap="vik", series=cpt)
    fig.basemap(region=region, projection=projection, frame=["a0.5f0.1"])
    
    fig.grdimage(
        grid=da,
        projection=projection,
        cmap=True,
        transparency=30
    )

    fig.coast(shorelines=False, region=region, projection=projection, water='204/212/219')
    fig.colorbar(frame="af+lLOS (m)",transparency=30)
    return fig

def geotiff2xarray(geotiff_file):
    '''
    Parameters
    ----------
    geotiff_file : string
        full path to .tif file that you want to convert
        This code was built to handle the geotiffs produced by ISCE for range or azimuth SAR offsets but might be able to handle other types of geotiffs
    Returns
    -------
    Returns an xarray dataset. This must be flattened to a dataarray to be compatible with pygmt

    NOTES: Might need to be updated to handle LOS datasets
    '''
    da = rioxarray.open_rasterio(geotiff_file)          # Open the data array with rioxarray
    observed_ds = da.to_dataset('band')          # Take the band object and set it as a variable
    observed_ds = observed_ds.rename({1: 'los_obs','x': 'lon','y': 'lat'})          # Rename the variable to a more useful name

    # NaNs are read in as -140000 by default so let's convert all of these to NaNs 
    observed_ds = observed_ds.where(observed_ds != -140000)          # replace all values equal to -140000 with NaNs 
    observed_ds = observed_ds.where(observed_ds != -23000)          # replace all values equal to -140000 with NaNs 
    return observed_ds

def geotiff2look(geotiff_file):
    '''
    Parameters
    ----------
    geotiff_file : string
        full path to .tif Look file. 
        This should be a *los.tif file which contains look vectors as bands 1
        This code was built to handle the los (look) geotiffs produced by ISCE
    Returns
    -------
    look_ds : xarray dataset object
        With additional data variables:
            * The LOS look vector (los_e, los_n, los_u)
            * The azimuth offset look vector (az_e, az_n, az_u)
    Ex: look_ds = geotiff2look('s1_2023018-20230209_p14_los.tif')
    '''

    # ## Calculate Average Incidence Angle
    # ## Calculate Average Incidence Angle
    da = rioxarray.open_rasterio(geotiff_file)          # Open the dataset with rioxarray
    look_ds = da.to_dataset('band')                  # Covert our xarray.DataArray into a xarray.Dataset
    look_ds = look_ds.rename({1: 'incidence',2:'azimuth'})          # Rename the variable to a more useful name
    look_ds = look_ds.where(look_ds != 0)          # replace zeros with NaNs 

    # Print mean incidence and azimuth angles for user
    print('Mean incidence angle (deg): ', np.nanmean(look_ds.incidence))
    print('Mean azimuth angle (deg): ', np.nanmean(look_ds.azimuth))

    # Calculate LOS Look vector

    # Calculate East component and write it to our dataset as a new variable
    look_ds = look_ds.assign(los_e=lambda look_ds: np.cos(np.radians(look_ds.azimuth)+np.radians(450))*np.sin(np.radians(look_ds.incidence)))
    # Calculate North component and write it to our dataset as a new variable
    look_ds = look_ds.assign(los_n=lambda look_ds: np.sin(np.radians(look_ds.azimuth)+np.radians(450))*np.sin(np.radians(look_ds.incidence)))
    # Calculate Vertical component and write it to our dataset as a new variable
    look_ds = look_ds.assign(los_u=lambda look_ds: np.cos(np.radians(look_ds.incidence)))
    print('Mean LOS look vector (e, n, u): ( ',np.nanmean(look_ds.los_e),',' ,np.nanmean(look_ds.los_n),',',np.nanmean(look_ds.los_u),')')
    print('**Note: Positive values indicate motion in the direction of the satellite')
    print('This is the same as the range offset look vector')

    # Calculate Azimuth offset Look vector
    # Calculate East component and write it to our dataset as a new variable
    look_ds = look_ds.assign(az_e=lambda look_ds: np.cos(np.radians(look_ds.azimuth)))
    look_ds = look_ds.assign(az_n=lambda look_ds: np.sin(np.radians(look_ds.azimuth)))
    look_ds = look_ds.assign(az_u=lambda look_ds: (look_ds.azimuth)*0)
    print('Mean Azimuth offset look vector (e, n, u): ( ',np.nanmean(look_ds.az_e),',' ,np.nanmean(look_ds.az_n),',',np.nanmean(look_ds.az_u),')')              
    print('**Note: Positive values indicate motion in the along-track direction of the satellite (i.e. the direction that the satellite is moving is positive)')

    # Reformat the x and y data coordinates to lat and lon
    look_ds = look_ds.rename({'x': 'lon','y': 'lat'})          # Rename the variable to a more useful name

    return look_ds

def merge_data(observed_ds, synthetic_da):
    '''
    Description
    ----------
    This function takes two overlappping xarray datasets, one with ovservations, and one with modeled surface deformation
    This upsamples the modeled surface deforamtions to match the observation sampling and writes them to a merged dataset
    Example usage: ds_merge = merge_data(observed_ds, synthetic_da)

    Parameters
    ----------
    observed_ds : xarray dataset
        xarray dataset object with some observed dataset
        Should have a data variable named 'los_obs'
    synthetic_da : xarray dataarray object
        xarray data array object with some synthetic dataset, projected into the same LOS as observation dataset
    Returns
    -------
    ds_merge : xarray dataset object 
        Function returns xarray dataset object with data variables 'los_synth' and 'los_obs', each in meters. 
    '''
    print('Original synthetic dataset has dimensions: ', synthetic_da.shape)

    # Upsample the synthetic data to match the observed data
    # write the results to a new data array
    interp_synth = synthetic_da.interp_like(observed_ds)
    # Check that the shape matches the observed data now
    print('Upsampled synthetic dataset has dimensions: ', interp_synth.shape)

    print('Original observed dataset has dimensions: ', observed_ds.los_obs.shape)

    ## Join the datasets
    print('Merging syntheic and observed data... ')
    # Use xarrays to do an outer join of the two datasets
    # This combines them, keeping all of the coordinate values used by either dataset
    # Fills in NaNs for any coordinates where data doesn't exist
    ds_merge = xr.combine_by_coords([interp_synth,observed_ds], join='outer')

    # check that the shape is reasonable
    print('Mean of merged Synthetic data = ',ds_merge.los_synth.mean(skipna=True).values)
    print('Mean of merged Observed data = ',ds_merge.los_obs.mean(skipna=True).values)

    ## Take the residual
    # Take the observed data minus the synthetic data 
    # Write the result as a new data variable, 'residual' which is in meters
    ds_merge = ds_merge.assign(residual=lambda ds_merge: ds_merge.los_obs - ds_merge.los_synth)
    print('Mean of merged Residuals = ',ds_merge.residual.mean(skipna=True).values)

    return ds_merge

def read_iscedata(scene,tiffdir=None):
    '''
    Description
    ----------
    This function takes 3 ISCE-produced geotiff files (range.tif, azimuth.tif, and los.tif) and merges them into a single xarray dataset on the same coordinate system
    Example usage:

    Parameters
    ----------
    scene : string
        string with the name of the .tif files in the format 'platform_firstdate-seconddate_path'
        i.e. 's1_20230129-20230210_p21'
    tiffdir : string
        location of all .tif files (range.tif, azimuth.tif, and los.tif)
    Returns
    -------
    ds_merge : xarray dataset object 
        Function returns xarray dataset object with data variables:  
           * incidence: ISCE output incidence angle ""
           * azimuth: ISCE output azimuth angle ""
           * los_e
           * los_n
           * los_u
           * az_e
           * az_n
           * az_u
           * az_offset
           * rng_offset
    '''
    if tiffdir==None:
        tiffdir = os.getcwd()
    
    # Read in the observed data Range offset data
    geotiff_file = tiffdir + scene + '_range.tif'
    # Read it in as an xarray dataset
    range_offsets_ds = geotiff2xarray(geotiff_file)
    # rename data variable to something unique
    range_offsets_ds = range_offsets_ds.rename({'los_obs':'rng_offset'})

    ## Read in the observed data Azimuth offset data
    geotiff_file = tiffdir + scene + '_azimuth.tif'
    # Read it in as an xarray dataset
    azimuth_offsets_ds = geotiff2xarray(geotiff_file)
    azimuth_offsets_ds = azimuth_offsets_ds.rename({'los_obs':'az_offset'})

    ## Read in the Look vectors for products
    # Set the geotiff path to LOS tif
    geotiff_file = tiffdir + scene + '_los.tif'
    # Calculate the look vectors at each point and write to a dataset
    look_ds = geotiff2look(geotiff_file)

    ## Merge all observations (Range offsets, Azimuth offsets, and Look vectors) into a single dataset
    merged_ds = xr.combine_by_coords([look_ds,azimuth_offsets_ds, range_offsets_ds], join='outer')
    return merged_ds

