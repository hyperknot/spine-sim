#!/bin/bash

python extract_opensim_masses.py \
    --model "models/Fullbody_TLModels_v2.0_OS4x/MaleFullBodyModel_v2.0_OS4.osim" \
    --geom "models/Fullbody_TLModels_v2.0_OS4x/Geometry" \
    --out fullbody.json \
    --debug-disks
