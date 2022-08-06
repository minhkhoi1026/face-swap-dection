conda install -y -c conda-forge cudatoolkit=10.1 cudnn=7.6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install -r requirement.txt
