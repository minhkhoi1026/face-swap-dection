import pandas as pd
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", help="path to log file", required=True)
    
    return parser.parse_args()

args = parse_args()
df = pd.read_csv(args.log_path)

# create plot frame
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))

# plot accuray
axes[0].plot(df['accuracy'])
axes[0].plot(df['val_accuracy'])
axes[0].set_title('model accuracy')
axes[0].set_ylabel('accuracy')
axes[0].set_xlabel('epoch')
axes[0].legend(['train', 'validation'], loc='upper right')

# plot loss
axes[1].plot(df['loss'])
axes[1].plot(df['val_loss'])
axes[1].set_title('model loss')
axes[1].set_ylabel('loss')
axes[1].set_xlabel('epoch')
axes[1].legend(['train', 'validation'], loc='upper right')

fig.tight_layout()

plt.show()
