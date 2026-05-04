# Generative AI for Trajectory Prediction
This repository provides code for training a model that predicts vehicle trajectories. The model takes as input an image representing the previous one-second trajectory and generates an output image depicting the predicted trajectories for the next second in the traffic scenario. The model can also predict further seconds ahead, receiving as input its output.

## Installation and Setup
To install and run this project locally, follow these steps:

### 1. Clone the repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/lucegi/gen_ai_trajectory_prediction.git
cd gen_ai_trajectory_prediction
```

### 2. Install the requirements
Install all the required dependencies listed in the requirements.txt:
```bash
pip install -r requirements.txt
```

### 3. Download the DLR UT dataset
Navigate to the /trajectories folder and download the DLR UT trajectory dataset (in .csv format) from the [official site](https://zenodo.org/records/14773161):
```bash
cd trajectories

python3 download_trajectory_data.py
```
This downloads and unzips the data,

### 4. Convert the dataset
The dataset in .csv files needs to be converted in images, representing a frame in the traffic scenario:
```bash
python3 rasterize_data.py
```
After running this command, the folder /images_boxes_800 will be populated by images representing frames of the traffic scenario.
From the dataset generated, is necessary to infer the map, by running the command:
```bash
python3 raster_map.py
```
This script analyzes the images in the folder /images_boxes_800 and generates lane_topology.png, an image showing the topology of the road:
![topology](media/lane_topology.png)

### 5. Train
#### Black and White representation:
If you want to learn black and white representation, with three timesteps encoded in three gray scales +  the topology of the road, run:
```bash
python3 convert_gray_timesteps.py
```
This command generates the dataset of black and white images in the folder /images_boxes_BW_800. 
Then to train the Unet model:
```bash
python3 train_prediction_image_space_multistep.py --image bw
```
The model will learn this representation:
![BW_traffic](media/gray_tsteps_54788.png)

#### RGB representation:
If you want to learn a RGB representation, in which each timestep is encoded in each RGB channel, run:
```bash
python3 convert_multistep_multichannel_dataset.py
```
This command generates the dataset of RGB images in the folder /images_boxes_RGB_800. 
Then to train the Unet model:
```bash
python3 train_prediction_image_space_multistep.py --image rgb
```
The model will learn this representation:
![BW_traffic](media/rgb_54788.png)