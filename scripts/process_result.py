import os
import pandas as pd


# Read the CSV file into a DataFrame
df = pd.read_csv("result/wandb_export_2023-06-07T17_16_11.520+07_00.csv")
d = dict()
for index, row in df.iterrows():
    filepath = row['img_path']
    pred = row['pred']
    label = row['target']

    file = os.path.basename(filepath)[:-9]

    if file not in d.keys():
        d[file] = (0,0,label)
    
    x,y,l = d[file]
    d[file] = (x+pred, y+1, l)

list = [(file,x/y,l) for file,(x,y,l) in d.items()]

list.sort(key=lambda item: abs(item[2]-item[1]))

for item in list:
    print(item[1], "\t", item[0])