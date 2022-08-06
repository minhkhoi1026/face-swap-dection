conda install -y -c conda-forge cudatoolkit=10.1 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install -r requirement.txt
