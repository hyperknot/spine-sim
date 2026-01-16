#!/bin/bash
# OpenSim environment setup using micromamba
# Requires: brew install micromamba
# Then run: /opt/homebrew/opt/micromamba/bin/micromamba shell init --shell bash --root-prefix ~/micromamba
# Restart terminal after init

# Remove any broken environment
micromamba deactivate
micromamba env remove -n opensim_scripting

# Create and configure the environment
micromamba create -n opensim_scripting python=3.11 numpy -c conda-forge
micromamba activate opensim_scripting
micromamba install conda-forge::libcxx
micromamba install -c opensim-org -c conda-forge opensim
