import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

NKP = {0 : 18,
       1 : 24,
       2 : 20,
       3 : 18,
       4 : 6,
       5 : 22,
       6 : 21,
       7 : 9,
       8 : 9,
       9 : 6,
       10 : 6,
       11 : 14,
       12 : 11,
       13 : 10,
       14 : 10,
       15 : 24}

ID2NAMES = {"02691156": "airplane",
            "02808440": "bathtub",
            "02818832": "bed",
            "02876657": "bottle",
            "02954340": "cap",
            "02958343": "car",
            "03001627": "chair",
            "03467517": "guitar",
            "03513137": "helmet",
            "03624134": "knife",
            "03642806": "laptop",
            "03790512": "motorcycle",
            "03797390": "mug",
            "04225987": "skateboard",
            "04379243": "table",
            "04530566": "vessel",}
NAMES2ID = {v: k for k, v in ID2NAMES.items()}

AVAILABLE_DATASETS = [
    "modelnet10", "modelnet40", "modelnet10_zrotated",
    "shapenetsem", "shapenet", "shapenetsvr", "shapenetkey",
    "abc"
]

#Plot 3D
PLOT_MAX_3D_POINTS = 2048
DEFAULT_3D_VIEW = (30, -45)
TOP_3D_VIEW = (60, -45)
PLOT_3D_ALPHA = 1.
PLOT_3D_LIM = .75
CMAP = "viridis"
CMAP_SEG = "hsv"
COLOR_POINTS = 3 * [v for _, v in mcolors.TABLEAU_COLORS.items()]

#Tensor
COPY_NOISE = 0.0001

#Sampling OBJs
N_MAX_POINTS = 2**12

#Projection and alinement networks initialization
DECODER_INIT_MEAN = 0.
DECODER_INIT_STD = .01

#Copying linear shape models
COPY_NOISE_SCALE = .0001