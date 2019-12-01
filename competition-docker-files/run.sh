#!/usr/bin/env bash
pip install --upgrade tensorflow-hub
python run_local_test.py -dataset_dir=./datasets/2 -code_dir=./model
