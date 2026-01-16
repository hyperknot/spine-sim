#!/bin/bash

python extract_opensim_masses.py \
    --model "/Users/user/Documents/life/paragliding/2025/harness/opensim/models/Fullbody_TLModels_v2.0_OS4x/MaleFullBodyModel_v2.0_OS4.osim" \
    --geom "/Users/user/Documents/life/paragliding/2025/harness/opensim/models/Fullbody_TLModels_v2.0_OS4x/Geometry" \
    --out fullbody.json
