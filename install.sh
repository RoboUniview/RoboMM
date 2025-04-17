#!/bin/bash
# create conda environment
conda env create -f environment.yml
conda activate RBMM214
# install the Deformable-DETR
cd ..
git clone https://github.com/fundamentalvision/Deformable-DETR.git
cd Deformable-DETR/models/ops
pip install .
pip install setuptools==57.5.0
pip install pyhash==0.9.3
# install RLbench
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

cd ../../../
wget --no-check-certificate https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
pip install git+https://github.com/stepjam/RLBench.git
pip install pytorch-lightning==1.8.6

