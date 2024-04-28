#!/usr/bin/env bash

conda env create -f environment.yaml
echo "Please close and reopen your shell. Activate your new environment with 'conda activate tgm', then run 'pip install -e .'"
