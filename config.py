# config.py

# Image properties
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 800
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
MARGIN = 10

# Output directories (auto-named based on resolution)
RASTER_OUTPUT_DIR = f"images_boxes_{IMAGE_WIDTH}"
BW_OUTPUT_DIR = f"images_boxes_BW_{IMAGE_WIDTH}"
RGB_OUTPUT_DIR = f"images_boxes_RGB_{IMAGE_WIDTH}"
LANE_TOPOLOGY_IMAGE = "lane_topology.png"

# Input directory
TRAJECTORY_CSV_DIR = "trajectories"

# Lane map processing
LANE_THRESHOLD = 0.5
LANE_MASK_INTENSITY = 0.05

# Temporal parameters
FRAMES_PER_SEQUENCE = 3
FRAME_STEP = 2  # Overlapping frames

# Rendering control
SAMPLE_RATE = 10  # Every 10th frame 
BASE_DT = 0.05                       # one original sensor tick
DT_FRAME = SAMPLE_RATE * BASE_DT    # e.g. 0.5 s between RGB channelss

GRAY_WEIGHTS = [0.3, 0.6, 1.0]
BOUNDS_FILE = "global_bounds.json"

CACHED_TRAJECTORY_PICKLE = "cached_trajectories.pkl"
CACHE_IF_MISSING = True  # Set False to force reload
CSV_FILE_PATTERN = "trajectories_*.csv"

BOX_SCALE = 2.0                  # m ➜ px for box renderer
GAUSS_GAIN = 1.5                 # m ➜ px gain for gaussian renderer
GAUSS_MIN_PX = 2.0