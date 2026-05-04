import pandas as pd 
import matplotlib.pyplot as plt
import os 
import glob

path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))

for file in csv_files[:2]:
    data = pd.read_csv(file)
    ids = data["id"].unique()
    for id in ids:
        if data.loc[data["id"] == id, "classifications_car"].to_list()[0] > 0.5 or data.loc[data["id"] == id, "classifications_van"].to_list()[0] > 0.5 or data.loc[data["id"] == id, "classifications_truck"].to_list()[0] > 0.5:
            x = data.loc[data["id"] == id, "center_easting"]
            y = data.loc[data["id"] == id, "center_northing"]
            plt.plot(x, y)

plt.plot([604719, 604797, 604819, 604736, 604719], [5792819, 5792846, 5792763, 5792741, 5792819], color="r")
plt.show()