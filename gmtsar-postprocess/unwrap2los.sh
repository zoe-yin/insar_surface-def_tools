#!/bin/csh

#  script to convert to LOS and geocode

# unwrap_mask.grd should be in radar coordinates

# Convert from Phase to Line-of-Sight

# grab the wavelength from the .PRM master file
# then do grdmath to get from wavelength to LOS 
set wavel = `grep wavelength *.PRM | awk '{print($3)}' | head -1 ` 
echo 'wavelength is set to ' $wavel
gmt grdmath unwrap.grd $wavel MUL -79.58 MUL = los.grd
gmt grdgradient los.grd -Nt.9 -A0. -Glos_grad.grd
# get the max and min values of LOS to make a color palette

# set limitU = `echo $tmp | awk '{printf("%5.1f", $12+$13*2)}'`
# set limitL = `echo $tmp | awk '{printf("%5.1f", $12-$13*2)}'`
# gmt makecpt -Cpolar -Z -T"-$limitU"/"$limitU"/1 -D > los.cpt
gmt makecpt -Cvik -Z -T-300/300/1 -D > los.cpt
gmt grdimage los.grd -Ilos_grad.grd -Clos.cpt -Bxaf+lRange -Byaf+lAzimuth -BWSen -JX6.5i -X1.3i -Y3i -P -K > los.ps
gmt psscale -Rlos.grd -J -DJTC+w5i/0.2i+h+e -Clos.cpt -Bxaf+l"LOS displacement [range decrease @~\256@~]" -By+lmm -O >> los.ps
gmt psconvert -Tf -P -A -Z los.ps
echo "Line-of-sight map: los.pdf"

# convert los to lat-lon
proj_ra2ll.csh trans.dat los.grd los_ll.grd
# set BT = `gmt grdinfo -C los.grd | awk '{print $7}'`
# set BL = `gmt grdinfo -C los.grd | awk '{print $6}'`
gmt makecpt -Cvik -T-900/900/1 -Z > los.cpt
grd2kml.csh los_ll los.cpt

# detrend the data
gmt grdtrend los_ll.grd -N3r -Dlos_ll_dtr.grd

# # # mask out the water
# # make a landmask
# gmt grdlandmask -Rlos_ll.grd -Df -Glandmask_ll.grd -NNaN/1
# # apply it to the los grd
# gmt grdmath los_ll.grd landmask_ll.grd MUL = los_mask_ll.grd
# grd2kml.csh los_mask_ll los.cpt

gmt makecpt -Cvik -T100/200/1 -Z > los.cpt
grd2kml.csh los_ll los.cpt