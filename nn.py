########### CARE MODEL
import tensorflow as tf
import keras

# config = tf.ConfigProto(device_count={"CPU": 32})
# keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, IsotropicCARE,CARE
from csbdeep.utils import download_and_extract_zip_file, axes_dict, plot_some, plot_history

from PIL import Image


import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics, preprocessing
from sklearn import pipeline, model_selection

# from keras import backend as K
from scipy.stats import pearsonr
# from sklearn import svm, linear_model
# import numpy as np
# import matplotlib.pyplot as plt
# import microscPSF.microscPSF as msPSF
# import PIL
import scipy

# from scipy import matrix
# from scipy.sparse import coo_matrix
# import time
# from scipy import linalg
# from skimage import color, data, restoration
# from skimage.transform import rescale, resize, downscale_local_mean
# from scipy.signal import convolve2d as conv2
# import matlab.engine
# import pandas as pd

# import keras
from keras import metrics
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense, Dropout, Activation, Convolution1D, Flatten, Conv1D, UpSampling1D, InputLayer, UpSampling2D, Conv2D, Reshape, Input, LeakyReLU, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate
from keras.optimizers import SGD
from keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler


# from sklearn.preprocessing import Imputer
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# from sklearn.experimental import enable_iterative_imputer
# from scipy import signal
# from sklearn.impute import SimpleImputer

# import os

# x,y,image_x,image_y
