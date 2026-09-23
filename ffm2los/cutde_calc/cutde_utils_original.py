#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:01:29 2023

@author: degoldberg
Original script from Dara Goldberg, summer 2023
"""

from mpl_toolkits.mplot3d import Axes3D
from cutde.halfspace import disp
from numpy import arange, linspace, meshgrid, array, c_, deg2rad, sin, cos, zeros, ones, shape, tile, delete, mean, rad2deg
from matplotlib import pyplot as plt
from math import asin, atan
import os
import pyproj


def cutde_Okada_displacements(frame=100,points=10, directory=None):
    print('Writing Okada Displacement File...')

    if directory==None:
        directory = os.getcwd()

    ##########################
    #### FAULT INFORMATION ###
    ##########################
    fsp = open(directory+'/fsp_sol_file.txt','r') #Get hypocenter info from fsp file
    for line in fsp:
        if line.startswith('% Loc'):
            hypo_lat = float(line.split()[5])
            hypo_lon = float(line.split()[8])
    fsp.close()
    sol = open(directory+'/Solucion.txt','r') #Get subfault size information from param file
    for line in sol:
        if line.startswith('#Fault_segment'):
            sf_length_as = float(line.split()[7].split('km')[0])
            sf_length_ad = float(line.split()[12].split('km')[0])
    sol.close()
    nu = 0.25 #POISSON'S RATIO
    #########################################
    ### COORDINATES OF OBSERVATION POINTS ###
    #########################################
    lowerbound = -abs(frame)
    upperbound = frame
    x = linspace(lowerbound,upperbound,points) #in km
    y = linspace(lowerbound,upperbound,points) #in km
    # x = linspace(-170,170,50) #in km
    # y = linspace(-170,170,50) #in km
    g = pyproj.Geod(ellps='WGS84') # Use WGS84 Ellipsoid
    # Convert to grid of lon/lat around epicenter
    # Remember pyproj assumes things are in meters, so multiple km by 1000
    _,xLats,_ = pyproj.Geod.fwd(g, hypo_lon*ones(len(x)), hypo_lat*ones(len(x)), 0*ones(len(x)), x*1000)
    xLons,_,_ = pyproj.Geod.fwd(g, hypo_lon*ones(len(y)), hypo_lat*ones(len(y)), 90*ones(len(y)), y*1000)
    # Make Longitude values in range from 0 to 360 (not -180 to 180) to avoid plotting issues
    for kLon in range(len(xLons)):
        if xLons[kLon] < 0:
            xLons[kLon] = 360 + xLons[kLon]

    gridLon, gridLat = meshgrid(xLons,xLats)
    # grid_pts = array([gridLon, gridLat, 0*gridLon]).reshape((3,-1)).T.copy()

    # initiate subfault arrays. These will eventually be the length of the number of subfaults
    LAT=[]; LON=[]; DEP=[]; SLIP=[]; RAKE=[]; STRIKE=[]; DIP=[]; T_RUP=[]; T_RIS=[]; T_FAL=[]; MO=[]

    sol = open(directory+'/Solucion.txt')
    for line in sol:
        if '#' in line: #HEADER LINES, skip over these
            continue
        if len(array(line.split())) < 4: #FAULT BOUNDARY LINES, skip over these
            continue
        else: #ACTUAL SUBFAULT DETAILS
            lat, lon, dep, slip, rake, strike, dip, t_rup, t_ris, t_fal, mo = line.split()
            # Make rake be -180 to 180 (not 0-360)
            if float(rake) > 180:
                rake = float(rake) - 360
            # Make Lon 0 to 360 (not -180 to 180)
            if float(lon) < 0:
                lon = float(lon) + 360
            LAT.append(float(lat)); LON.append(float(lon)); DEP.append(float(dep)); SLIP.append(float(slip)); #Now you have arrays of all the subfault info
            RAKE.append(float(rake)); STRIKE.append(float(strike)); DIP.append(float(dip));
    sol.close()
    ###############################
    ### CALCULATE DISPLACEMENTS ###
    ###############################
    Nfaults = len(LAT)  # number of subfaults
    ### Initialize northing, easting, and vertical displacements, ux, uy, uz ###
    ux = zeros(len(xLats)*len(xLons))
    uy = zeros(len(xLats)*len(xLons))
    uz = zeros(len(xLats)*len(xLons))

#    for ksub in range(40):
    for ksub in range(Nfaults): # for every subfault... 
        if ksub % 100 == 0:
            print('...Subfault: '+str(ksub))
        # Set variables for each subfault 
        sf_lon = LON[ksub]
        sf_lat = LAT[ksub]
        strike = STRIKE[ksub]
        dip = DIP[ksub]
        slip = SLIP[ksub]   #cm
        rake = RAKE[ksub]
        depth = DEP[ksub]
        ss_slip = (slip/100.) * cos(deg2rad(rake)) #slip in the strike-direction, convert from cm to m
        ds_slip = (slip/100.) * sin(deg2rad(rake)) #slip in the dip direction, convert from cm to m
        os_slip = 0 # Opening slip (assume none)

        ### Make new x,y vectors, grid of distances in km from current subfault to lon/lat grid of observation points defined earlier ###
        fwd_az,b_az,distance_m = pyproj.Geod.inv(g,sf_lon*ones(shape(gridLon)),sf_lat*ones(shape(gridLat)),gridLon,gridLat)
        distance_km = distance_m/1000.
        x = distance_km*sin(deg2rad(fwd_az))
        y = distance_km*cos(deg2rad(fwd_az))
        ### Turn observation points into three-column variable, with z-coordinate = 0 (surface)
        obs_pts = array([x, y, 0*y]).reshape((3,-1)).T.copy()
        
        ### Find corners of subfault in question, starting with N-striking, vertical dipping fault with same dimmensions as model subfaults ###
        fault_corners = array([
                        [0, -sf_length_as/2, -DEP[ksub] - sf_length_ad/2],
                        [0, -sf_length_as/2, -DEP[ksub] + sf_length_ad/2],
                        [0, sf_length_as/2, -DEP[ksub] + sf_length_ad/2],
                        [0, sf_length_as/2, -DEP[ksub] - sf_length_ad/2]
                        ])
        
        ### Then we will rotate that N-striking, vertically dipping subfault into the correct strike/dip ###
        ### Rotation matrix for strike angle
        theta_stk = deg2rad(strike)
        R_stk = zeros((3,3))
        R_stk[0,0] = cos(theta_stk)
        R_stk[0,1] = -sin(theta_stk)
        R_stk[1,0] = sin(theta_stk)
        R_stk[1,1] = cos(theta_stk)
        R_stk[2,2] = 1
        ### Rotation matrix for dip angle
        theta_dip = -deg2rad(90-dip) # Rotate by the complement of the dip, since dip is measured from horizontal plane
        R_dip = zeros((3,3))
        R_dip[0,0] = cos(theta_dip)
        R_dip[0,2] = -sin(theta_dip)
        R_dip[1,1] = 1
        R_dip[2,0] = sin(theta_dip)
        R_dip[2,2] = cos(theta_dip)
        ### Apply rotations:
        fault_corners = fault_corners.dot(R_dip)
        fault_corners = fault_corners.dot(R_stk)
        
        # ## 3D scatter plot of subfault, for sanity check
        # if ksub==0:
        #     #print(strike, dip)
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection = '3d')
        #     ax.scatter(fault_corners[:,0],fault_corners[:,1],fault_corners[:,2])
        #     ax.plot(fault_corners[:,0],fault_corners[:,1],fault_corners[:,2])
        #     set_axes_equal(ax)
        #     #print(fault_corners)
        #     plt.show()

        # ## Sanity check-- does this subfault have the same str/dip it came in with?? ###
        # check_dip = rad2deg(asin((fault_corners[1,2] - fault_corners[0,2]) / sf_length_ad))
        # check_str = rad2deg(atan((fault_corners[2,0] - fault_corners[1,0])/(fault_corners[2,1]-fault_corners[1,1])))
        # if (abs(check_dip-dip)) > 0.05 or (abs(check_str-strike)) > 0.05: # are check_str/check_dip equal to subfaul str/dip?
        #     print('STRIKE/DIP Inconsistency!')
        # print('Strike Check:', check_str)
        # print('Strike:', strike)
        # print('Dip values:', check_dip)
        # print('Dip:', dip)

        ### Split subfault corners into 2 triangels for calculation
        triangle1 = tile(fault_corners[0:3,:],(len(obs_pts),1,1)) # corners of first triangle are corners 0,1,2
        corners2 = delete(fault_corners, 1, axis=0) # corners of second triangle are corners 0,2,3
        triangle2 = tile(corners2,(len(obs_pts),1,1))
        
        slips = zeros((len(obs_pts),3))
        slips[:,0] = ss_slip # strike-slip slip
        slips[:,1] = ds_slip # dip-slip slip
        slips[:,2] = os_slip # opening slip
        
        disp_triangle1 = disp(obs_pts, triangle1, slips, nu)
        disp_triangle2 = disp(obs_pts, triangle2, slips, nu)
        
        disp_total = disp_triangle1 + disp_triangle2

        # Add contribution of this subfault to northing, easting, vertical displacements
        ux = ux + disp_total[:,0]
        uy = uy + disp_total[:,1]
        uz = uz+disp_total[:,2]


    DISPout = open(directory+'/surface_deformation.disp','w')
    DISPout.write('#Longitude, Latitude, Elevation, Easting Displacement (m), Northing Displacement (m), Vertical Displacement (m)\n')
    ko=0

    for klat in range(len(xLats)):
        for klon in range(len(xLons)):
            DISPout.write('%10.4f \t %10.4f \t %10.4f \t %10.4f \t %10.4f \t %10.4f \n' % (xLons[klon], xLats[klat], 0, ux[ko], uy[ko], uz[ko]))
            ko+=1
            
    ############
    ### PLOT ###
    ###########
    
    # Horizontal def plot
    plt.figure(figsize=(8,6))
    horizontal = (ux**2 + uy**2) **0.5
    horizontal_grid = horizontal.reshape((shape(gridLon)))
    horiz = abs(horizontal)
    horiz.sort()
    minmax = horiz[-2]
    plt.scatter(gridLon,gridLat,marker='s',c=horizontal*100, lw=0, s=100, vmin=0.0, vmax=minmax*100, cmap='pink_r')
    cb = plt.colorbar()
    cb.set_label('Horizontal Surface Displacement (cm)')
    
    for klat in range(len(xLats)):
        for klon in range(len(xLons)):
            if klat % 5 == 0 and klon % 10 == 0:
                i = klat * len(xLons) + klon  # Compute the flat index for ux, uy, horizontal
                plt.quiver(
                    xLons[klon], xLats[klat],
                    # ux[i]/horizontal[i], uy[i]/horizontal[i],
                    ux[i], uy[i],
                    pivot='mid', linewidths=0.01, width=.005, color='k', scale=30
                )

    plt.xlim([xLons.min(),xLons.max()])
    plt.ylim([xLats.min(),xLats.max()])
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    plt.savefig(directory+'/Horizontal_Surface_Displacement.png', dpi=300)
    #plt.close()

    # Vertical def plot
    plt.figure(figsize=(8,6))
    vertical_grid = uz.reshape((shape(gridLon)))
    abs_uz = abs(uz)
    abs_uz.sort()
    minmax = abs_uz[-2]
    plt.scatter(gridLon,gridLat, marker ='s', c=uz*100,lw=0, s=100, vmin=-minmax*100, vmax=minmax*100,cmap='coolwarm')
    cb = plt.colorbar()
    cb.set_label('Vertical Surface Displacement (cm)')
    plt.xlim([xLons.min(),xLons.max()])
    plt.ylim([xLats.min(),xLats.max()])
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    plt.savefig(directory+'/Vertical_Surface_Displacement.png', dpi=300)
    #plt.close()
    
    
    
# The following function is only used for the quick 3d plot of a subfault as a sanity check that it's oriented correctly
def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    