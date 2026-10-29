#!/bin/bash
# ll2ra_bounds.sh
#
# Convert a lat/lon bounding box (lower-left and upper-right corners) into
# a GMTSAR region_cut string (range_min/range_max/azimuth_min/azimuth_max)
# using SAT_llt2rat. No DEM or trans.dat needed -- this back-geocodes
# directly from the orbit state vectors in the PRM file.
#
# Usage:
#   ll2ra_bounds.sh master.PRM lat_ll lon_ll lat_ur lon_ur [elevation] [precise]
#
# Arguments:
#   master.PRM  - PRM file for the reference image (must have a valid orbit
#                 pointer)
#   lat_ll      - latitude of the lower-left (SW) corner
#   lon_ll      - longitude of the lower-left (SW) corner
#   lat_ur      - latitude of the upper-right (NE) corner
#   lon_ur      - longitude of the upper-right (NE) corner
#   elevation   - elevation in meters to use for all corners (default 0)
#   precise     - 0 (fast/approximate, default) or 1 (slower polynomial
#                 refinement)
#
# Example:
#   ll2ra_bounds.sh NSR_20260406A.PRM 39.27340 -119.15735 39.42459 -118.90963

set -euo pipefail

if [ $# -lt 5 ]; then
    echo "Usage: $0 master.PRM lat_ll lon_ll lat_ur lon_ur [elevation=0] [precise=0]" >&2
    exit 1
fi

PRM=$1
LAT_LL=$2
LON_LL=$3
LAT_UR=$4
LON_UR=$5
ELEV=${6:-0}
PRECISE=${7:-0}

if [ ! -f "$PRM" ]; then
    echo "Error: PRM file '$PRM' not found." >&2
    exit 1
fi

TMP_LLT=$(mktemp)
TMP_RAT=$(mktemp)
trap 'rm -f "$TMP_LLT" "$TMP_RAT"' EXIT

# SAT_llt2rat expects: lon lat elevation (per corner)
cat > "$TMP_LLT" <<EOF
$LON_LL $LAT_LL $ELEV
$LON_LL $LAT_UR $ELEV
$LON_UR $LAT_LL $ELEV
$LON_UR $LAT_UR $ELEV
EOF

SAT_llt2rat "$PRM" "$PRECISE" < "$TMP_LLT" > "$TMP_RAT"

if [ ! -s "$TMP_RAT" ]; then
    echo "Error: SAT_llt2rat produced no output. Check that the box falls" >&2
    echo "within the swath covered by $PRM." >&2
    exit 1
fi

echo "Corner points (range azimuth lon lat elev):"
cat "$TMP_RAT"
echo

# output columns: range azimuth lon lat elevation
RNG_MIN=$(awk 'NR==1{m=$1} {if($1<m)m=$1} END{print int(m)}' "$TMP_RAT")
RNG_MAX=$(awk 'NR==1{m=$1} {if($1>m)m=$1} END{print int(m)+1}' "$TMP_RAT")
AZI_MIN=$(awk 'NR==1{m=$2} {if($2<m)m=$2} END{print int(m)}' "$TMP_RAT")
AZI_MAX=$(awk 'NR==1{m=$2} {if($2>m)m=$2} END{print int(m)+1}' "$TMP_RAT")

echo "region_cut = ${RNG_MIN}/${RNG_MAX}/${AZI_MIN}/${AZI_MAX}"