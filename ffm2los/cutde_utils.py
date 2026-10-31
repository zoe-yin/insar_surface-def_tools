#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31, 2026

@author: hyin (based upon work from dgoldberg)
"""

from mpl_toolkits.mplot3d import Axes3D
from cutde.halfspace import disp
from numpy import arange, linspace, meshgrid, array, c_, deg2rad, sin, cos, zeros, ones, shape, tile, delete, mean, rad2deg
from matplotlib import pyplot as plt
from math import asin, atan
import os
import pyproj
import numpy as np
from matplotlib.colors import TwoSlopeNorm
import re
from dataclasses import dataclass
from pyproj import Transformer

# Listed point -> cell centre, as fractions of (Dx along strike, Dy down dip).
# 1/3 each reproduces the cell centres in the matching .fsp file.
CENTER_OFFSET = (1 / 3, 1 / 3)

_HEADER = re.compile(r"nx\(Along-strike\)=\s*(\d+)\s+Dx\s*=\s*([\d.]+)km\s+ny\(downdip\)=\s*(\d+)\s+Dy\s*=\s*([\d.]+)km")


@dataclass
class CutdeSources:
    tris: np.ndarray        # (2N, 3, 3) vertices, km, (East, North, Up); UTM grid axes, relative to origin_m
    slips: np.ndarray       # (2N, 3) strike-slip, dip-slip, tensile/opening (always 0 here), metres
    centers: np.ndarray     # (N, 3) subfault centres, same frame, for plotting / checks
    epsg: int               # UTM zone the frame is built on, e.g. 32617
    origin_m: tuple         # (E0, N0) UTM metres subtracted from every coordinate

    # Row k and row k+N of tris/slips are the two triangles of subfault k (use row % N to get the subfault).

    def to_xy(self, lon, lat):
        """Degrees -> local km, in the same frame as tris (use for observation points)."""
        e, n = Transformer.from_crs(4326, self.epsg, always_xy=True).transform(np.asarray(lon, float), np.asarray(lat, float))
        return (e - self.origin_m[0]) / 1e3, (n - self.origin_m[1]) / 1e3

    def to_lonlat(self, x_km, y_km):
        """Local km -> degrees."""
        return Transformer.from_crs(self.epsg, 4326, always_xy=True).transform(
            np.asarray(x_km, float) * 1e3 + self.origin_m[0], np.asarray(y_km, float) * 1e3 + self.origin_m[1])


def read_solution_sources(path):
    # 1. Read: every 11-column line is a subfault; the latest segment header gives its Dx, Dy.
    rows, dx, dy = [], None, None
    for line in open(path):
        m = _HEADER.search(line)
        if m:
            dx, dy = float(m[2]), float(m[4])
        elif len(line.split()) == 11 and not line.lstrip().startswith("#"):
            rows.append([float(v) for v in line.split()] + [dx, dy])
    lat, lon, depth, slip_cm, rake, strike, dip, *_, dx, dy = np.array(rows).T

    # 2. Project to a local UTM frame (km), origin at the mean subfault position.
    zone = int((lon.mean() + 180) // 6) + 1
    epsg = (32600 if lat.mean() >= 0 else 32700) + zone
    fwd = Transformer.from_crs(4326, epsg, always_xy=True)
    e0, n0 = fwd.transform(lon.mean(), lat.mean())
    e, n = fwd.transform(lon, lat)
    x, y = (e - e0) / 1e3, (n - n0) / 1e3
    # strike is measured from true north; UTM's grid north is a hair different (grid convergence)
    strike = strike - np.degrees(np.arctan(np.tan(np.radians(lon - (6 * zone - 183))) * np.sin(np.radians(lat))))

    # 3. Two unit vectors describe each plane: along strike (s) and down dip (d), as (E, N, Up).
    az, dp = np.radians(strike), np.radians(dip)
    s = np.column_stack([np.sin(az), np.cos(az), 0 * az])
    d = np.column_stack([np.cos(dp) * np.cos(az), -np.cos(dp) * np.sin(az), -np.sin(dp)])

    # 4. Cell centres, then slide any cell that pokes above the surface down its own plane until it touches.
    center = np.column_stack([x, y, -depth]) + CENTER_OFFSET[0] * dx[:, None] * s + CENTER_OFFSET[1] * dy[:, None] * d
    above = np.clip(center[:, 2] + 0.5 * dy * np.sin(dp), 0, None)     # height of the top edge above z = 0
    center += (above / np.sin(dp))[:, None] * d

    # 5. Corners: deep/start, shallow/start, shallow/end, deep/end.  Split into triangles (0,1,2) and (0,2,3).
    hs, hd = 0.5 * dx[:, None] * s, 0.5 * dy[:, None] * d
    corners = np.stack([center - hs + hd, center - hs - hd, center + hs - hd, center + hs + hd], axis=1)
    tris = np.ascontiguousarray(np.concatenate([corners[:, [0, 1, 2]], corners[:, [0, 2, 3]]]))

    # 6. Slip: cm -> m, split into strike-slip and dip-slip components; no opening.
    m = slip_cm / 100
    slips = np.tile(np.column_stack([m * np.cos(np.radians(rake)), -m * np.sin(np.radians(rake)), 0 * m]), (2, 1))

    return CutdeSources(tris, slips, center, epsg, (float(e0), float(n0)))

def plot_surface_deformation(path, save_as=None, show=True, quiver_stride=0,
                             wrap_lon=True, cmap="RdBu_r"):
    """Plot east, north and vertical surface displacement from a .disp file.

    Parameters
    ----------
    path : str
        Path to the displacement file.
    save_as : str, optional
        If given, the figure is saved to this path.
    show : bool
        Call ``plt.show()`` at the end.
    quiver_stride : int
        Draw a horizontal-displacement arrow every N grid nodes on the
        vertical panel (0 disables arrows).
    wrap_lon : bool
        Convert longitudes in 0..360 to -180..180 (e.g. 275.6 -> -84.4).
    cmap : str
        Diverging colormap; each panel is centered on zero.

    Returns
    -------
    fig, axes
    """
    data = np.loadtxt(path, comments="#")
    lon, lat = data[:, 0], data[:, 1]
    east, north, up = data[:, 3], data[:, 4], data[:, 5]

    if wrap_lon:
        lon = np.where(lon > 180, lon - 360, lon)

    # The points form a regular lon/lat grid, so reshape for pcolormesh.
    lons, lats = np.unique(lon), np.unique(lat)
    ny, nx = len(lats), len(lons)
    if nx * ny != len(data):
        raise ValueError(f"Points do not form a regular grid ({nx}x{ny} != {len(data)})")
    ix = np.searchsorted(lons, lon)
    iy = np.searchsorted(lats, lat)

    def to_grid(v):
        g = np.full((ny, nx), np.nan)
        g[iy, ix] = v
        return g

    panels = [
        ("Easting (m)", to_grid(east)),
        ("Northing (m)", to_grid(north)),
        ("Vertical (m)", to_grid(up)),
    ]
    e_grid, n_grid = panels[0][1], panels[1][1]
    horiz = np.hypot(e_grid, n_grid)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, (title, grid) in zip(axes, panels):
        vmax = np.nanmax(np.abs(grid))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        mesh = ax.pcolormesh(lons, lats, grid, cmap=cmap, norm=norm, shading="auto")
        fig.colorbar(mesh, ax=ax, shrink=0.85, label="Displacement (m)")
        ax.set_title(title)
        ax.set_xlabel("Longitude (°)")
        ax.set_aspect("equal")

    axes[0].set_ylabel("Latitude (°)")

    # Horizontal displacement vectors on the vertical panel.
    if quiver_stride:
        s = quiver_stride
        X, Y = np.meshgrid(lons, lats)
        q = axes[2].quiver(X[::s, ::s], Y[::s, ::s], e_grid[::s, ::s], n_grid[::s, ::s],
                           color="0.15", angles="xy", scale_units="xy",
                           scale=np.nanmax(horiz) / (2 * (lons[1] - lons[0]) * s),
                           width=0.003)
        ref = float(np.round(np.nanmax(horiz) / 2, 1)) or float(np.nanmax(horiz) / 2)
        axes[2].quiverkey(q, 0.2, 1.1, ref, f"{ref:g} m horizontal", labelpos="E",
                          coordinates="axes", labelsep=0.05)

    fig.suptitle("Surface displacement", fontsize=14)

    if save_as:
        fig.savefig(save_as, dpi=200)
    if show:
        plt.show()
    return fig, axes


def surface_displacement(src, lon, lat, nu=0.25, max_pairs=2_000_000):
    """East, North, Up displacement (metres) at the surface, summed over every subfault.

    src   : CutdeSources from read_solution_sources()
    lon, lat : observation points in degrees; scalars, 1-D, or a 2-D grid (any shape)
    nu    : Poisson's ratio
    max_pairs : observation-source pairs per cutde call (memory knob; ~2M is a few hundred MB)
    disp  : the cutde function disp(obs_pts, tris, slips, nu); defaults to cutde.halfspace.disp

    Returns an array of shape (*lon.shape, 3).  East/North follow the UTM grid axes (they differ from
    true east/north by the grid convergence, a few hundredths of a degree for a fault near its zone centre).
    """
    lon = np.asarray(lon, float)
    x, y = src.to_xy(lon.ravel(), np.asarray(lat, float).ravel())
    obs = np.column_stack([x, y, np.zeros_like(x)])            # on the free surface, z = 0

    n_src = len(src.tris)                                      # 2 triangles per subfault
    step = max(1, max_pairs // n_src)                          # observation points per cutde call
    u = np.empty((len(obs), 3))
    for a in range(0, len(obs), step):
        o = obs[a:a + step]
        n = len(o)
        # every observation point paired with every triangle, then summed over the triangles
        d = disp(np.repeat(o, n_src, axis=0), np.tile(src.tris, (n, 1, 1)), np.tile(src.slips, (n, 1)), nu)
        u[a:a + n] = np.asarray(d).reshape(n, n_src, 3).sum(axis=1)

    return u.reshape(*lon.shape, 3)


 
def make_grid(west, east, south, north, res):
    """Grid of `res`-degree pixels covering the box. Returns (lon, lat, geotransform).
 
    lon, lat      : (ny, nx) arrays of pixel-centre coordinates
    geotransform  : GDAL order (west edge, res, 0, north edge, 0, -res)
 
    Edges are snapped outward to multiples of `res`, so grids built for different events line up.
    """
    i0, i1 = int(np.floor(west / res)), int(np.ceil(east / res))
    j0, j1 = int(np.floor(south / res)), int(np.ceil(north / res))
    w, n = round(i0 * res, 10), round(j1 * res, 10)
    lon1d = np.round(w + (np.arange(i1 - i0) + 0.5) * res, 10)    # west -> east
    lat1d = np.round(n - (np.arange(j1 - j0) + 0.5) * res, 10)    # north -> south
    lon, lat = np.meshgrid(lon1d, lat1d)
    return lon, lat, (w, res, 0.0, n, 0.0, -res)
 
 
def grid_around(src, pad_km=100, res=0.02):
    """Same as make_grid, for a box around the fault plus `pad_km` on every side (src = CutdeSources)."""
    x, y = src.centers[:, 0], src.centers[:, 1]
    xs = [x.min() - pad_km, x.max() + pad_km, x.min() - pad_km, x.max() + pad_km]
    ys = [y.min() - pad_km, y.min() - pad_km, y.max() + pad_km, y.max() + pad_km]
    lons, lats = src.to_lonlat(xs, ys)
    return make_grid(min(lons), max(lons), min(lats), max(lats), res)


# #### OLD CODE BELOW 
# def cutde_Okada_displacements(frame=100,points=10, directory=None):
#     print('Writing Okada Displacement File...')

#     if directory==None:
#         directory = os.getcwd()

#     ##########################
#     #### FAULT INFORMATION ###
#     ##########################
#     fsp = open(directory+'/fsp_sol_file.txt','r') #Get hypocenter info from fsp file
#     for line in fsp:
#         if line.startswith('% Loc'):
#             hypo_lat = float(line.split()[5])
#             hypo_lon = float(line.split()[8])
#     fsp.close()
#     sol = open(directory+'/Solucion.txt','r') #Get subfault size information from param file
#     for line in sol:
#         if line.startswith('#Fault_segment'):
#             sf_length_as = float(line.split()[7].split('km')[0])
#             sf_length_ad = float(line.split()[12].split('km')[0])
#     sol.close()
#     nu = 0.25 #POISSON'S RATIO
#     #########################################
#     ### COORDINATES OF OBSERVATION POINTS ###
#     #########################################
#     lowerbound = -abs(frame)
#     upperbound = frame
#     x = linspace(lowerbound,upperbound,points) #in km
#     y = linspace(lowerbound,upperbound,points) #in km
#     # x = linspace(-170,170,50) #in km
#     # y = linspace(-170,170,50) #in km
#     g = pyproj.Geod(ellps='WGS84') # Use WGS84 Ellipsoid
#     # Convert to grid of lon/lat around epicenter
#     # Remember pyproj assumes things are in meters, so multiple km by 1000
#     _,xLats,_ = pyproj.Geod.fwd(g, hypo_lon*ones(len(x)), hypo_lat*ones(len(x)), 0*ones(len(x)), x*1000)
#     xLons,_,_ = pyproj.Geod.fwd(g, hypo_lon*ones(len(y)), hypo_lat*ones(len(y)), 90*ones(len(y)), y*1000)
#     # Make Longitude values in range from 0 to 360 (not -180 to 180) to avoid plotting issues
#     for kLon in range(len(xLons)):
#         if xLons[kLon] < 0:
#             xLons[kLon] = 360 + xLons[kLon]

#     gridLon, gridLat = meshgrid(xLons,xLats)
#     # grid_pts = array([gridLon, gridLat, 0*gridLon]).reshape((3,-1)).T.copy()

#     # initiate subfault arrays. These will eventually be the length of the number of subfaults
#     LAT=[]; LON=[]; DEP=[]; SLIP=[]; RAKE=[]; STRIKE=[]; DIP=[]; T_RUP=[]; T_RIS=[]; T_FAL=[]; MO=[]

#     sol = open(directory+'/Solucion.txt')
#     for line in sol:
#         if '#' in line: #HEADER LINES, skip over these
#             continue
#         if len(array(line.split())) < 4: #FAULT BOUNDARY LINES, skip over these
#             continue
#         else: #ACTUAL SUBFAULT DETAILS
#             lat, lon, dep, slip, rake, strike, dip, t_rup, t_ris, t_fal, mo = line.split()
#             # Make rake be -180 to 180 (not 0-360)
#             if float(rake) > 180:
#                 rake = float(rake) - 360
#             # Make Lon 0 to 360 (not -180 to 180)
#             if float(lon) < 0:
#                 lon = float(lon) + 360
#             LAT.append(float(lat)); LON.append(float(lon)); DEP.append(float(dep)); SLIP.append(float(slip)); #Now you have arrays of all the subfault info
#             RAKE.append(float(rake)); STRIKE.append(float(strike)); DIP.append(float(dip));
#     sol.close()
#     ###############################
#     ### CALCULATE DISPLACEMENTS ###
#     ###############################
#     Nfaults = len(LAT)  # number of subfaults
#     ### Initialize northing, easting, and vertical displacements, ux, uy, uz ###
#     ux = zeros(len(xLats)*len(xLons))
#     uy = zeros(len(xLats)*len(xLons))
#     uz = zeros(len(xLats)*len(xLons))

# #    for ksub in range(40):
#     for ksub in range(Nfaults): # for every subfault... 
#         if ksub % 100 == 0:
#             print('...Subfault: '+str(ksub))
#         # Set variables for each subfault 
#         sf_lon = LON[ksub]
#         sf_lat = LAT[ksub]
#         strike = STRIKE[ksub]
#         dip = DIP[ksub]
#         slip = SLIP[ksub]   #cm
#         rake = RAKE[ksub]
#         depth = DEP[ksub]
#         ss_slip = (slip/100.) * cos(deg2rad(rake)) #slip in the strike-direction, convert from cm to m
#         ds_slip = (slip/100.) * sin(deg2rad(rake)) #slip in the dip direction, convert from cm to m
#         os_slip = 0 # Opening slip (assume none)

#         ### Make new x,y vectors, grid of distances in km from current subfault to lon/lat grid of observation points defined earlier ###
#         fwd_az,b_az,distance_m = pyproj.Geod.inv(g,sf_lon*ones(shape(gridLon)),sf_lat*ones(shape(gridLat)),gridLon,gridLat)
#         distance_km = distance_m/1000.
#         x = distance_km*sin(deg2rad(fwd_az))
#         y = distance_km*cos(deg2rad(fwd_az))
#         ### Turn observation points into three-column variable, with z-coordinate = 0 (surface)
#         obs_pts = array([x, y, 0*y]).reshape((3,-1)).T.copy()
        
#         ### Find corners of subfault in question, starting with N-striking, vertical dipping fault with same dimmensions as model subfaults ###
#         fault_corners = array([
#                         [0, -sf_length_as/2, -DEP[ksub] - sf_length_ad/2],
#                         [0, -sf_length_as/2, -DEP[ksub] + sf_length_ad/2],
#                         [0, sf_length_as/2, -DEP[ksub] + sf_length_ad/2],
#                         [0, sf_length_as/2, -DEP[ksub] - sf_length_ad/2]
#                         ])
        
#         ### Then we will rotate that N-striking, vertically dipping subfault into the correct strike/dip ###
#         ### Rotation matrix for strike angle
#         theta_stk = deg2rad(strike)
#         R_stk = zeros((3,3))
#         R_stk[0,0] = cos(theta_stk)
#         R_stk[0,1] = -sin(theta_stk)
#         R_stk[1,0] = sin(theta_stk)
#         R_stk[1,1] = cos(theta_stk)
#         R_stk[2,2] = 1
#         ### Rotation matrix for dip angle
#         theta_dip = -deg2rad(90-dip) # Rotate by the complement of the dip, since dip is measured from horizontal plane
#         R_dip = zeros((3,3))
#         R_dip[0,0] = cos(theta_dip)
#         R_dip[0,2] = -sin(theta_dip)
#         R_dip[1,1] = 1
#         R_dip[2,0] = sin(theta_dip)
#         R_dip[2,2] = cos(theta_dip)
#         ### Apply rotations:
#         fault_corners = fault_corners.dot(R_dip)
#         fault_corners = fault_corners.dot(R_stk)
        
#         ### 3D scatter plot of subfault, for sanity check
#         # if ksub==0:
#         #     #print(strike, dip)
#         #     fig = plt.figure()
#         #     ax = fig.add_subplot(111, projection = '3d')
#         #     ax.scatter(fault_corners[:,0],fault_corners[:,1],fault_corners[:,2])
#         #     ax.plot(fault_corners[:,0],fault_corners[:,1],fault_corners[:,2])
#         #     set_axes_equal(ax)
#         #     #print(fault_corners)
#         #     plt.show()

#         ### Sanity check-- does this subfault have the same str/dip it came in with?? ###
#         # check_dip = rad2deg(asin((fault_corners[1,2] - fault_corners[0,2]) / sf_length_ad))
#         # check_str = rad2deg(atan((fault_corners[2,0] - fault_corners[1,0])/(fault_corners[2,1]-fault_corners[1,1])))
#         # if (abs(check_dip-dip)) > 0.05 or (abs(check_str-strike)) > 0.05: # are check_str/check_dip equal to subfaul str/dip?
#             # print('STRIKE/DIP Inconsistency!')
#         # print('Strike values:', check_str)
#         # print('Dip values:', check_dip)

#         ### Split subfault corners into 2 triangels for calculation
#         triangle1 = tile(fault_corners[0:3,:],(len(obs_pts),1,1)) # corners of first triangle are corners 0,1,2
#         corners2 = delete(fault_corners, 1, axis=0) # corners of second triangle are corners 0,2,3
#         triangle2 = tile(corners2,(len(obs_pts),1,1))
        
#         slips = zeros((len(obs_pts),3))
#         slips[:,0] = ss_slip # strike-slip slip
#         slips[:,1] = ds_slip # dip-slip slip
#         slips[:,2] = os_slip # opening slip
        
#         disp_triangle1 = disp(obs_pts, triangle1, slips, nu)
#         disp_triangle2 = disp(obs_pts, triangle2, slips, nu)
        
#         disp_total = disp_triangle1 + disp_triangle2

#         # Add contribution of this subfault to northing, easting, vertical displacements
#         ux = ux + disp_total[:,0]
#         uy = uy + disp_total[:,1]
#         uz = uz+disp_total[:,2]


#     DISPout = open(directory+'/surface_deformation.disp','w')
#     DISPout.write('#Longitude, Latitude, Elevation, Easting Displacement (m), Northing Displacement (m), Vertical Displacement (m)\n')
#     ko=0

#     for klat in range(len(xLats)):
#         for klon in range(len(xLons)):
#             DISPout.write('%10.4f \t %10.4f \t %10.4f \t %10.4f \t %10.4f \t %10.4f \n' % (xLons[klon], xLats[klat], 0, ux[ko], uy[ko], uz[ko]))
#             ko+=1
            
#     ############
#     ### PLOT ###
#     ###########
    
#     # Horizontal def plot
#     plt.figure(figsize=(8,6))
#     horizontal = (ux**2 + uy**2) **0.5
#     horizontal_grid = horizontal.reshape((shape(gridLon)))
#     horiz = abs(horizontal)
#     horiz.sort()
#     minmax = horiz[-2]
#     plt.scatter(gridLon,gridLat,marker='s',c=horizontal*100, lw=0, s=100, vmin=0.0, vmax=minmax*100, cmap='rainbow')
#     cb = plt.colorbar()
#     cb.set_label('Horizontal Surface Displacement (cm)')
    
#     for klat in range(len(xLats)):
#         for klon in range(len(xLons)):
#             if klat % 5 == 0 and klon % 10 == 0:
#                 i = klat * len(xLons) + klon  # Compute the flat index for ux, uy, horizontal
#                 plt.quiver(
#                     xLons[klon], xLats[klat],
#                     ux[i]/horizontal[i], uy[i]/horizontal[i],
#                     pivot='mid', linewidths=0.01, width=.005, color='k', scale=30
#                 )

#     plt.xlim([xLons.min(),xLons.max()])
#     plt.ylim([xLats.min(),xLats.max()])
#     plt.ylabel('Latitude')
#     plt.xlabel('Longitude')
#     plt.savefig(directory+'/Horizontal_Surface_Displacement.png', dpi=300)
#     #plt.close()

#     # Vertical def plot
#     plt.figure(figsize=(8,6))
#     vertical_grid = uz.reshape((shape(gridLon)))
#     abs_uz = abs(uz)
#     abs_uz.sort()
#     minmax = abs_uz[-2]
#     plt.scatter(gridLon,gridLat, marker ='s', c=uz*100,lw=0, s=100, vmin=-minmax*100, vmax=minmax*100,cmap='coolwarm')
#     cb = plt.colorbar()
#     cb.set_label('Vertical Surface Displacement (cm)')
#     plt.xlim([xLons.min(),xLons.max()])
#     plt.ylim([xLats.min(),xLats.max()])
#     plt.ylabel('Latitude')
#     plt.xlabel('Longitude')
#     plt.savefig(directory+'/Vertical_Surface_Displacement.png', dpi=300)
#     #plt.close()
    
    
    
# # The following function is only used for the quick 3d plot of a subfault as a sanity check that it's oriented correctly
# def set_axes_equal(ax):
#     """
#     Make axes of 3D plot have equal scale so that spheres appear as spheres,
#     cubes as cubes, etc.

#     Input
#       ax: a matplotlib axis, e.g., as output from plt.gca().
#     """

#     x_limits = ax.get_xlim3d()
#     y_limits = ax.get_ylim3d()
#     z_limits = ax.get_zlim3d()

#     x_range = abs(x_limits[1] - x_limits[0])
#     x_middle = mean(x_limits)
#     y_range = abs(y_limits[1] - y_limits[0])
#     y_middle = mean(y_limits)
#     z_range = abs(z_limits[1] - z_limits[0])
#     z_middle = mean(z_limits)

#     # The plot bounding box is a sphere in the sense of the infinity
#     # norm, hence I call half the max range the plot radius.
#     plot_radius = 0.5*max([x_range, y_range, z_range])

#     ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
#     ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
#     ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    