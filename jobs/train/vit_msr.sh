conda activate fsd

pip install .

python src/train_test.py -c configs/train/double_head_vit_msr_hybrid.yml
