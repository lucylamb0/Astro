#!/bin/bash

export PYTHONPATH=/user/HS103/md0046/.local/lib/python3.5/site-packages/:/user/HS103/md0046/tools/photutils/install/lib/python3.5/site-packages/:/user/HS103/md0046/tools/astropy/install/lib/python3.5/site-packages/:/user/HS103/md0046/tools/lmfit-py/install/lib/python3.5/site-packages/:$PYTHONPATH

ipython3 -i astro_lab.py
