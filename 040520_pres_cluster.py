# %% markdown
# # A general algorithm for microscope image deconvolution
# %% codecell
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL
import scipy
from IPython.display import Video
from tqdm import tqdm
import multiprocessing
import os,sys
import skimage
from enum import Enum,auto

# os.getcwd()
if(sys.platform=="linux"):
    os.chdir("/homes/ctr26/gdrive/+projects/2019_jrl/2019_jRL_impute")

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer
from sklearn.metrics import mean_squared_error


# from sklearn import neural_network, metrics, gaussian_process, preprocessing, svm, neighbors
# from sklearn import pipeline, model_selection

# from keras import metrics
# from keras import backend as K
from scipy.stats import pearsonr
from sklearn import svm, linear_model
# import microscPSF.microscPSF as msPSF
from skimage.metrics import structural_similarity as ssim

from scipy import matrix
from scipy.sparse import coo_matrix
import time
from scipy import linalg
from skimage import color, data, restoration, exposure
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.signal import convolve2d as conv2
# import matlab.engine

# from sklearn.preprocessing import Imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# from sklearn.experimental import enable_iterative_imputer
from scipy import signal
from sklearn.impute import SimpleImputer

class psf_switch_enum(Enum):
    STATIC,VAR_PSF,VAR_ILL = auto(),auto(),auto()

SAVE_IMAGES = 0
# %% markdown
# # Image formation
# %% markdown
# Define constants: psf height width and image rescaling factor
# %% codecell
psf_w,psf_h,scale = 64,64,4 # Define constants: psf height width and image rescaling factor
psf_window_w, psf_window_h = round(psf_w/scale), round(psf_h/scale)
sigma = 1
# %% markdown
# Define approximate PSF function
# %% codecell
def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def psf_guass(w=psf_w, h=psf_h, sigma=3):
    # blank_psf = np.zeros((w,h))
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    psf = gaussian(xx, 0, sigma) * gaussian(yy, 0, sigma)
    return  psf/psf.sum() # Normalise PSF "energy"

static_psf = psf_guass(w=round(psf_window_h), h=round(psf_window_w), sigma=1 / scale); plt.imshow(static_psf)
# %% codecell
Video("eiffel_smlm.mp4") # Credit: Ricardo Henriques
# %% markdown
# # Deconvolution
# %% codecell
astro = rescale(color.rgb2gray(data.astronaut()), 1.0 / scale)
astro_blur = conv2(astro, static_psf, 'same') # Blur image
astro_corrupt = astro_noisy = astro_blur + (np.random.poisson(lam=25, size=astro_blur.shape) - 10) / 255. # Add Noise to Image
deconvolved_RL = restoration.richardson_lucy(astro_corrupt, static_psf, iterations=30)  # RL deconvolution

fig, ax = plt.subplots(nrows=1, ncols=4,figsize=(16,7))
ax[0].imshow(astro);ax[0].set_title('Truth')
ax[1].imshow(astro_blur);ax[1].set_title('Blurred')
ax[2].imshow(astro_noisy);ax[2].set_title('Blurred and noised')
ax[3].imshow(deconvolved_RL, vmin=astro_noisy.min(), vmax=astro_noisy.max());ax[3].set_title('Deconvolved');
# %% markdown
# # Richardson Lucy Deconvolution
# %% markdown
# We can achieve a model for image formation of an ideal system by generating a measurement matrix $\mathbf{H}$ that acts on the structure $\mathbf{g}$ we are trying to find (assuming the structure is quantised) to produce a pixelated image $\mathbf{f}$ (n-dimensional):
#
# <!-- %%latex -->
# \begin{align*}
# \underbrace{\mathbf{f}}_\text{Image} &= \overbrace{\mathbf{H}}^\text{Measurement matrix} \underbrace{\mathbf{g}}_\text{Object}\\
# \end{align*}
#
# In summation form:
#
# \begin{align*}
# f_{N_p}&=\sum_{N_v} H_{N_p, N_v} g_{N_v}\\
#  \overbrace{
# \begin{bmatrix}
#     f_{11} \\
#     \vdots  \\
#     f_{N_p}
#     \end{bmatrix}
# }^{N_p \times 1} \quad &= \overbrace{
# \begin{bmatrix}
#     H_{11} & H_{12} & \dots \\
#     \vdots & \ddots & \\
#     H_{N_v1} &        & H_{N_v N_p}
#     \end{bmatrix}
# }^{N_p \times N_v}  \overbrace{
# \begin{bmatrix}
#     g_{11} \\
#     \vdots  \\
#     g_{N_v}
#     \end{bmatrix}
# }^{N_v \times 1}\\
# \end{align*}
# %% codecell
N_v = np.ma.size(astro);N_v
N_p = np.ma.size(astro_blur);N_p
measurement_matrix = matrix(np.zeros((N_p, N_v)))
# %% markdown
# However, the system will be corrupted by noise such that:
#
# \begin{align*}
# \mathbf{f}= \mathbf{H} (\mathbf{g}+\mathbf{b})\\
# \end{align*}
#
# Assuming $\mathbf{b}$ as being a Poissonian noise distribution we can begin solve the inverse problem of finding $\mathbf{g}$ using maximum liklihood:
#
# \begin{align*}
# \operatorname{Pr}(\widehat{\mathbf{f}} | \mathbf{g}, \mathbf{b}) &=\prod_{i}\left(\frac{(H \mathbf{g}+\mathbf{b})_{i}{\widehat{\mathbf{f}}_{i}} \exp \left(-(H \mathbf{g}+\mathbf{b})_{i}\right)}{\widehat{\mathbf{f}}_{i} !}\right)
# \end{align*}
#
# It is then possible to solve for $\mathbf{g}$ iteratively giving the iterative Richardson lucy deconvolution algorithm in matrix form:
#
# \begin{align*}
# \mathbf{g}^{(k+1)}&=\operatorname{diag}\left(H^{T} \mathbf{1}\right)^{-1} \operatorname{diag}\left(H^{T} \operatorname{diag}\left(H \mathbf{g}^{(k)}+\mathbf{b}\right)^{-1} \mathbf{f}\right) \mathbf{g}^{(k)}
# \end{align*}
#
# In convolution notation with a spatially invariant point spread function (P, where P* is the flipped PSF) this can be compressed to:
#
# <!-- %%latex -->
# \begin{align*}
# \hat{g}^{(t+1)} & =\hat{g}^{(t)} \cdot\left(\frac{f}{\hat{g}^{(t)} \otimes P} \otimes P^{*}\right)
# \end{align*}
#
# So, if we know $\mathbf{H}$ and be extension $P$ we can deconvolve any image to retrieve a good approximation of an imaged object
#
# # Knowing $\mathbf{H}$
#
# Knowing $P$ is straightforward either experimentally or theoretically:
#
# For simple optical systems the Point Spread Function can be derived i.e for a perfect lens in a microscope with a glass slide and a liquid interface there is a closed form expression for each of the field components:
# \begin{align*}
# \begin{array}{l}
# h(x, y, z)=\left|I_{0}\right|^{2}+2\left|I_{1}\right|^{2}+\left|I_{2}\right|^{2} \\
# I_{0}(x, y, z)=\int_{0}^{\alpha} B_{0}(\theta, x, y, z)\left(t_{s}^{(1)} t_{s}^{(2)}+t_{p}^{(1)} t_{p}^{(2)} \frac{1}{n_{s}} \sqrt{n_{s}^{2}-n_{i}^{2} \sin ^{2} \theta}\right) d \theta \\
# I_{1}(x, y, z)=\int_{0}^{\alpha} B_{1}(\theta, x, y, z)\left(t_{p}^{(1)} t_{p}^{(2)} \frac{n_{i}}{n_{s}} \sin \theta\right) d \theta \\
# I_{2}(x, y, z)=\int_{0}^{\alpha} B_{2}(\theta, x, y, z)\left(t_{s}^{(1)} t_{s}^{(2)}+t_{p}^{(1)} t_{p}^{(2)} \frac{1}{n_{s}} \sqrt{n_{s}^{2}-n_{i}^{2} \sin ^{2} \theta}\right) d \theta \\
# B_{m}(\theta, x, y, z)=\sqrt{\cos \theta} \sin \theta J_{m}\left(k \sqrt{x^{2}+y^{2}} n_{i} \sin \theta\right) e^{j W(\theta)} \\
# W(\theta)=k\left\{t_{s} \sqrt{n_{s}^{2}-n_{i}^{2} \sin ^{2} \theta}+t_{i} \sqrt{n_{i}^{2}-n_{i}^{2} \sin ^{2} \theta}-t_{i}^{*} \sqrt{n_{i}^{* 2}-n_{i}^{2} \sin ^{2} \theta_{t}}\right. \\
# \left.+t_{g} \sqrt{n_{g}^{2}-n_{i}^{2} \sin ^{2} \theta}-t_{g}^{*} \sqrt{n_{g}^{* 2}-n_{i}^{2} \sin ^{2} \theta}\right\}
# \end{array}
# \end{align*}
#
# Knowing $P$ experimentally is relies on capturing images of bright objects that are smaller than the resolution of the instrument.
#
# We then **align and average** these multiple samplings of the PSF to approximate $P$
#
# However, $P$ is known to vary through lens imperfections causing optical abberations, meaning $\mathbf{H}$ is once again useful.
#
# $\mathbf{H}$ can also be written in terms of points spread functions:
# \begin{align*}
# \begin{bmatrix}
#     f_{1} \\
#     \vdots  \\
#     f_{N_p}
#     \end{bmatrix} \quad =
# \begin{bmatrix}
#     P_{1} \\
#     \vdots \\
#     P_{N_v}
#     \end{bmatrix}
# \begin{bmatrix}
#     g_{1} \\
#     \vdots  \\
#     g_{N_v}
#     \end{bmatrix}
# \\
# \end{align*}
#
# Where $P_n$ is a serialised Point Spread Function at the $n^\text{th}$ serialed pixel poisition. It's also possible to do this with tensors, but serialising is as functional
#
#
#
# Now, the difficulty therin lies that we do not know $P_n$ at every $n$; experimentally we know P_n at *most* positions but some form of **interpolation** is needed.
#
# It's fair to assume that the PSF varies smoothly for all $P_n$, but, there are several fringe cases of imaging system where this assumption falls flat and so interpolation alone would not produce a completely general deconvolution algorithm.
#
# <p float="center">
#     <img src="moire.png" width="200"/>
#     <img src="lightfield.png" width="200"/>
# </p>
#
# - **Structured illumination microscopy (SIM)** uses sinusoidally patterned light to increase image resolution
# - **Lightfield microscopy** uses an array of microlenes to record a 3D image on a 2D camera
#
# Both have funky spatially varying point spread functions.
# %% markdown
# # Building $\mathbf{H}$ from simulation
# %% markdown
# Set up arrays for generating H
# %% codecell
zero_image = np.zeros_like(astro)
psf_window_volume = np.full((N_v,psf_window_w, psf_window_h), np.NaN)

x_astro, y_astro = astro_blur.shape
xx_astro, yy_astro = np.meshgrid(np.linspace(-1, 1, x_astro),
                                    np.linspace(-1, 1, y_astro))
# %% markdown
# Store sinusoidal illumination incase things go well:
# %% codecell
illumination = np.cos(64 / 2 * np.pi * xx_astro)
plt.imshow(illumination)
# %% markdown
# Define a function that scales the PSF as a function of radial distance
# %% codecell
def psf_vary(psf_window_h,psf_window_w,radius,scale):
    return psf_guass(w=round(psf_window_h), h=round(psf_window_w),sigma=(1 / scale)*abs((-0.4*radius))+0.1)
# %% markdown
# Make the PSF vary across the image (as a function of radius)
# %% codecell
r_map = np.sqrt(xx_astro**2 + yy_astro**2)
radius_samples = np.linspace(-1,1,5)
fig,ax = plt.subplots(nrows=1,ncols=len(radius_samples),figsize=(16,7))

for i,radius in enumerate(np.linspace(-1,1,5)):
    psf_current = psf_vary(psf_window_h,psf_window_w,radius,scale)
    ax[i].imshow(psf_current);ax[i].set_title("Radius: " + str(radius))
plt.show()
# %% codecell
psf_switch = psf_switch_enum.VAR_PSF
if(psf_switch==psf_switch_enum.STATIC): filename = "H_staticpsf";
if(psf_switch==psf_switch_enum.VAR_PSF): filename = "H_varpsf";
if(psf_switch==psf_switch_enum.VAR_ILL): filename = "H_varill";
# %% markdown
# Loop over each row of H and insert a flattened PSF that is the same shape as the input image
# %% codecell
for i in tqdm(np.arange(N_v)):
    coords = np.unravel_index(i, np.array(astro.shape))  # Get the xy coordinates of the ith pixel in the original image
    r_dist = round(r_map[coords])                        # Convert to radius
    if(psf_switch==psf_switch_enum.STATIC):              # Select mode for generating H, i.e. static/varying psf etc.
            psf_current = static_psf
    if(psf_switch==psf_switch_enum.VAR_PSF):
        psf_current = psf_vary(psf_window_h,psf_window_w,radius,scale);
    if(psf_switch==psf_switch_enum.VAR_ILL):
        psf_current = static_psf* illumination[coords]

    psf_window_volume[i, :, :] = psf_current
    delta_image = np.zeros_like(astro)
    delta_image[np.unravel_index(i, astro.shape)] = 1
    delta_PSF = scipy.ndimage.convolve(delta_image, psf_current) # Convolve PSF with a image with a single 1 at coord

    measurement_matrix[i, :] = delta_PSF.flatten()
    if(SAVE_IMAGES):plt.imshow(psf_current);plt.imsave(f'./output/psfs/{str(i).zfill(6)}.png',psf_window_volume[:,:,i])
# %% codecell
# np.save(filename,measurement_matrix)
# %% codecell
# measurement_matrix = np.load(filename+".npy")
# %% markdown
# The resultant measurement matrix, $\mathbf{H}$.
#
# An ideal measurement matrix would have perfect transfer, i.e. be an identity matrix with leading 1s
# %% codecell
plt.figure(figsize=(18,7))
plt.imshow(exposure.equalize_hist(measurement_matrix))
# %% markdown
# Import custom RL algorithm, which uses sparse matrices to speedup the matrix calculations
# %% codecell
import richardson_lucy
H = scipy.sparse.linalg.aslinearoperator(measurement_matrix);
f = np.matrix(astro_blur.flatten()).transpose()
g = richardson_lucy.matrix_reconstruction(H,f,max_iter = 20)
# %% codecell
fig,ax = plt.subplots(ncols=3,nrows=1,figsize=(16,7))

ax[0].imshow(astro_blur);ax[0].set_title("Blurred")
ax[1].imshow(astro);ax[1].set_title("Original")
ax[2].imshow(g.reshape(astro_blur.shape));ax[2].set_title("RL")
plt.show()
# %% markdown
# # Learning $H$
# %% markdown
# Assuming we have a decently sized dataset of images of point images we can *try* to fill in the missing rows of $\mathbf{H}$
# %% markdown
# ### Matrix imputation
# Known as the netflix problem, attempts to fill in voids in matrices; fails miserably in this case though as entire rows are missing.
# %% codecell
# %% Begin RL matrix deconvolvution - Nuke beads
# %% markdown
# Remove majority of data randomly
#
# List 1000 random positions
# %% codecell
psfs = 1000

rows_to_nuke = np.random.choice(
    np.arange(measurement_matrix.shape[0]), measurement_matrix.shape[0] - psfs,replace=False);rows_to_nuke.shape
# %% markdown
# Remove rows
# %% codecell
psf_window_volume_nuked = psf_window_volume.copy()
psf_window_volume_nuked[rows_to_nuke,:, :] = np.NaN

H_nuked = measurement_matrix.copy()
H_nuked[rows_to_nuke,:] = np.NaN
# %% codecell

# %% codecell
# imp = IterativeImputer(missing_values=np.nan,verbose=2);
imp = SimpleImputer(missing_values=np.nan, strategy='mean',verbose=1);
imp.fit(H_nuked)
H_fixed = imp.transform(H_nuked)
# %% codecell
h_mse = mean_squared_error(measurement_matrix,H_fixed);h_mse
# %% markdown
# Save data for machine learning
# %% codecell
# # image_width,image_height = np.sqrt(measurement_matrix.shape).astype(np.int)
# image_width,image_height = astro.shape

# H_size_4d = [image_width,image_height,image_width,image_height]

# measurement_matrix_4d_nuked = np.reshape(np.array(H_nuked),array_size_4d)
# measurement_matrix_4d = np.reshape(np.array(measurement_matrix),array_size_4d)

# np.save('data/measurement_matrix_4d_nuked',measurement_matrix_4d_nuked)
# np.save('data/measurement_matrix_4d',measurement_matrix_4d)

# np.save('data/psf_window_volume_nuked',psf_window_volume_nuked)
# np.save('data/psf_window_volume',psf_window_volume)
# psf_window_volume_nuked.shape

# # plt.imsave('./output/H_nuked.png', H_nuked)
# %% markdown
# ### Machine learning
