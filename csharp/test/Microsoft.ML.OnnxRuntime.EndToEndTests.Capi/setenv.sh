#!/bin/bash

################################################################################
# Ensure script is only run sourced
if [[ $_ == $0 ]]; then
    echo "!!!ERROR!!! - PLEASE RUN WITH 'source'"
    exit 1
fi

################################################################################
# Set environment variables
export LD_LIBRARY_PATH=/home/brandon/work/ecplr/onnxruntime/build/Linux/RelWithDebInfo/
