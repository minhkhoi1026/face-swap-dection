conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
python3 -m pip install -r requirements.txt
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
python3 -m pip install tensorflow==2.11
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
