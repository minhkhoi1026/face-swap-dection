# FACE SWAP DETECTION BASED ON LANDMARK FOCUSING FEATURE

## Installation
The instruction assume you are on a linux-based operating system. Using other OS will have to incompatible and need to modify.

Prerequisites:
- Conda: https://docs.conda.io/en/latest/miniconda.html
- Python 3.8.16
```
conda create -n fsd python=3.8.16
```

Run this command in your bash terminal:
```
bash install_dep.sh
```
Then, install the source package using the following command in root folder:
```
pip install .
```

## Data
Training and testing images are stored in test and train folders respectively. The folder will be organised as such:
videos

```
dataset_v1/
|-- videos/
    |-- deepfaker/
        |-- real/
        |-- fake/
    |-- roop/
        |-- real/
        |-- fake/
|-- imgs/
    |-- deepfaker/
        |-- landmark/
            |-- real/
            |-- fake/
        |-- face/
            |-- real/
            |-- fake/
    |-- roop/
        |-- landmark/
            |-- real/
            |-- fake/
        |-- face/
            |-- real/
            |-- fake/
```

## Extract data
cmd:
```
$ python scripts/extract_all.py  --source <> --dest <> --sampling-ratio <> --extract-type <>
```
params:
- source: path of source data
- dest: path of destination image store
- sampling-ratio: specify a ratio x for frame sampling (0 < x <= 1)
- extract-type: choices in {all, frame, face}, default=all

For example:
```
$ python scripts/extract_all.py  --source ./data/ --dest ./data_extract/ --sampling-ratio 0.5 --extract-type all
```
If file structer of source folder '''./data/''' is as such:
```
    data
	    └── real
	    	    └── real1.mp4
					.
					.
	    └── fake
	    	    └── fake1.jpg
					.
					.
```
Then file structer of source folder '''./data_extract/''' is as such:
```
    data
		└── face_crop
					└── real
							└── real1_0000.png
								real1_0005.png
								.
								.
					└── fake
							└── fake1_0000.png
								fake1_0005.png
								.
								.
		└── landmark
					└── real
							└── real1_0000_landmark.png
								real1_0005_landmark.png
								.
								.
					└── fake
							└── fake1_0000_landmark.png
								fake1_0005_landmark.png
								.
								.
```


## Split dataset to train/test/val
```
python scripts/dataset_split.py --src <> --type <[normal,variant]> --sampling-ratio <>
```
For example:
```
python scripts/dataset_split.py --src fsd-deepfakerapp/dfapp_plus_landmark/train/ --type variant --sampling-ratio 0.5
```

## Train
Choose appropriate config in `configs` folder, then run this command:
```
python src/train.py -c configs/your_config.yml
```

## Test
Choose appropriate config in `configs` folder, then run this command:
```
python src/test.py -c configs/your_config.yml
```
User need to adjust these below fields:
- `dataset.args.test`
- `data_loader.test`
- `global.resume`: checkpoint path
- `metric` [optional]: change it whether you want different metrics with training stage
"""

## Demo application
This command will invoke new web application at `localhost:20626`:
```
streamlit run demo/lkt2.py --server.port 20626
```

Note: the first time the program process query, it need some warm-up time.

If you want to change the list of detector, you can edit the information of `configs/detector_registry_config.yml`.
