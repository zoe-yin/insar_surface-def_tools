#!/usr/bin/env python

import xarray as xr
import rioxarray
import os
import argparse

# merge_dir = '/Volumes/T9_InSAR/2026-04-14_nevada/S1_A064_gmtsar_stack/merge/'
filetypes = ["corr_ll", "los_ll", "los_ll_dtr", "xphase_mask_ll", "yphase_mask_ll", "phasefilt_ll","phasefilt_mask_ll", "xphase_ll","yphase_ll","phase_ll"]    # "los_ll", "xphase_mask_ll", "yphase_mask_ll", "phasefilt_ll", "imagfilt_ll", "realfilt_ll"


parser = argparse.ArgumentParser(
    description="Convert GMTSAR-produced grid files (.grd) to geotiffs" \
    "Example usage: python grd2geotiff.py /Volumes/T9_InSAR/2026-04-14_nevada/S1_A064_gmtsar_stack/merge/" \
    "Looks for file types: corr_ll,los_ll, los_ll_dtr, xphase_mask_ll, yphase_mask_ll, phasefilt_ll,phasefilt_mask_ll, xphase_ll,yphase_ll"
)

# Add an argument with NO flags (positional argument)
parser.add_argument("mergedir", help="Full path to a directory containing grd files to convert (usually a 'merge' dir)")

args = parser.parse_args()

merge_dir = args.mergedir
for file in filetypes:
    file_path = merge_dir + file + '.grd'
    if os.path.exists(file_path):
        ds = xr.open_dataset(file_path)
        # --- Normalize longitude to -180..180 ---
        if "lon" in ds.coords:
            lon = ds.lon.values
            # Wrap into [-180, 180]
            lon_wrapped = ((lon + 180) % 360) - 180
            ds = ds.assign_coords(lon=lon_wrapped)

        ds = ds.rename({"lon": "x", "lat": "y"})

        ds.rio.write_crs("EPSG:4326", inplace=True)  # WGS 84
        out_file = os.path.join(merge_dir, file + '.tiff')
        ds.rio.to_raster(out_file)

        print(f"Wrote {out_file}")
