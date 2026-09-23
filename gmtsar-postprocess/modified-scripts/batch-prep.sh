#!/bin/sh


# symlink files into F* directories
# # prep for baseline plot

## ASCENDING

BASEDIR='/Volumes/T9_InSAR/2026-04-14_nevada/S1_D144_batch/'

## Set up directories 
mkdir -p ${BASEDIR}/F1/raw ${BASEDIR}/F1/topo
mkdir -p ${BASEDIR}/F2/raw ${BASEDIR}/F2/topo
mkdir -p ${BASEDIR}/F3/raw ${BASEDIR}/F3/topo

# F1
cd ${BASEDIR}F1/raw
ln -s ${BASEDIR}data/*.SAFE/*/*iw1*vv*xml .
ln -s ${BASEDIR}data/*.SAFE/*/*iw1*vv*tiff .
ln -s ${BASEDIR}data/*EOF .
ln -s ${BASEDIR}topo/dem.grd ${BASEDIR}F1/topo/
ln -s ${BASEDIR}topo/dem.grd ${BASEDIR}F1/raw/
prep_data.csh
chmod ugo=rwx data.in
preproc_batch_tops.csh data.in dem.grd 1 >& log_preproc1
mv baseline_table.dat ../
# ln -s ${BASEDIR}data/orbits.list ${BASEDIR}F1/raw

# F2
cd ${BASEDIR}F2/raw
ln -s ${BASEDIR}data/*.SAFE/*/*iw2*vv*xml .
ln -s ${BASEDIR}data/*.SAFE/*/*iw2*vv*tiff .
ln -s ${BASEDIR}data/*EOF .
ln -s ${BASEDIR}topo/dem.grd ${BASEDIR}F2/topo/
ln -s ${BASEDIR}topo/dem.grd ${BASEDIR}F2/raw/
prep_data.csh
chmod ugo=rwx data.in
preproc_batch_tops.csh data.in dem.grd 1 >& log_preproc1
mv baseline_table.dat ../
# ln -s ${BASEDIR}data/orbits.list ${BASEDIR}F2/raw


# F3
cd ${BASEDIR}F3/raw
ln -s ${BASEDIR}data/*.SAFE/*/*iw3*vv*xml .
ln -s ${BASEDIR}data/*.SAFE/*/*iw3*vv*tiff .
ln -s ${BASEDIR}data/*EOF .
ln -s ${BASEDIR}topo/dem.grd ${BASEDIR}F3/topo/
ln -s ${BASEDIR}topo/dem.grd ${BASEDIR}F3/raw/
prep_data.csh
chmod ugo=rwx data.in
preproc_batch_tops.csh data.in dem.grd 1 >& log_preproc1
mv baseline_table.dat ../
# ln -s ${BASEDIR}data/orbits.list ${BASEDIR}F3/raw

