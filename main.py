# Can use Richardson Lucy to deconvolve with a spatially varying PSF
# 1) Use spatially invariant PSF for RL on skimage image
# 2) Use spatially varying PSF for RL on skimage
#   How do abberate  - skip
# 3) Put NaNs in matrix and impute

# https://scikit-learn.org/stable/modules/impute.html
# https://en.wikipedia.org/wiki/Matrix_completion
# https://en.wikipedia.org/wiki/Netflix_Prize
# https://stackoverflow.com/questions/17982931/matrix-completion-in-python

# https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.richardson_lucy

'''
Richardson-Lucy algorithm

Ported from the RestoreTools MATLAB package available at:
http://www.mathcs.emory.edu/~nagy/RestoreTools/

Input: A  -  object defining the coefficient matrix.
       b  -  Right hand side vector.

 Optional Intputs:

       x0      - initial guess (must be strictly positive); default is x0 = A.T*b
       sigmaSq - the square of the standard deviation for the
                 white Gaussian read noise (variance)
       beta    - Poisson parameter for background light level
       max_iter - integer specifying maximum number of iterations;
                  default is 100
       Rtol    - stopping tolerance for the relative residual,
                 norm(b - A*x)/norm(b)
                 default is 1e-6
       NE_Rtol - stopping tolerance for the relative residual,
                 norm(A.T*b - A.T*A*x)/norm(A.T*b)
                 default is 1e-6

Output:
      x  -  solution

Original MATLAB code by J. Nagy, August, 2011

References:
[1]  B. Lucy.
    "An iterative method for the rectication of observed distributions."
     Astronomical Journal, 79:745-754, 1974.
[2]  W. H. Richardson.
    "Bayesian-based iterative methods for image restoration.",
     J. Optical Soc. Amer., 62:55-59, 1972.
 [3]  C. R. Vogel.
    "Computational Methods for Inverse Problems",
    SIAM, Philadelphia, PA, 2002


'''

import numpy as np
import matplotlib.pyplot as plt
# import microscPSF.microscPSF as msPSF
import PIL
import scipy

from scipy import matrix
from scipy.sparse import coo_matrix
import time
from scipy import linalg
from skimage import color, data, restoration
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.signal import convolve2d as conv2
import matlab.engine

import os

# os.getcwd()

# eng = matlab.engine.start_matlab()
# eng.addpath(os.path.join(os.getcwd(),'IRrl.m'))




# Load and print the default microscope parameters.
# mp = msPSF.m_params
# for key in sorted(msPSF.m_params):
#     print(key, msPSF.m_params[key])
# print()
#
#
# pixel_size = 0.02
# zv = np.arange(-1.5, 1.51, pixel_size)
# scale = 8
# psf_xyz = msPSF.gLXYZFocalScan(mp, pixel_size*scale, 31, zv, pz = 0.1)
# psf_xy = psf_xyz[(np.abs(zv - 0)).argmin(),:,:]
# # im = PIL.Image.fromarray(psf_xy,mode='F')
# plt.imshow(np.sqrt(psf_xy))
#

#
# astro = color.rgb2gray(data.astronaut())
# astro = rescale(astro, 1.0 / scale)
# astro_blur = conv2(astro, psf_xy, 'same')
# astro_blur_noisy = astro_blur+(np.random.poisson(lam=25, size=astro_blur.shape) - 10) / 255.
# plt.imshow(astro_blur)
# plt.imshow(astro)
# # astro
#
# # astro_blur.shape
# # deconvolved_RL = restoration.richardson_lucy(astro_blur_noisy, psf_xy, iterations=100)
# deconvolved_RL
# # plt.imshow(deconvolved_RL)
#
# # H = [h_i,j] each row is a line of PSF centered on x,y filled with zeros
# N_v = np.ma.size(astro);N_v
# N_p = np.ma.size(astro_blur);N_p
# H = np.matrix(np.zeros((N_p,N_v)))
#
# zero_image = np.zeros_like(astro)
# astro_shape = astro.shape
# # astro_shape
# i=0



    # delta_PSF = psf_xy
    # plt.imshow(delta_image)
    # plt.show()


#     #
#     # print(i)
#
# # plt.imshow(np.log(H))
# num_iterations = 100
# # f = np.matrix(astro_blur.flatten())
# g = [None]*num_iterations

# # g[0] = f
# g[0] = H.transpose()*f
#
# from scipy.sparse import csc_matrix,coo_matrix
# from scipy.sparse.linalg import inv
# k=0
# g_real = astro
# num_iterations = 100
#
# import richardson_lucy
#
# H_sparse
# H_sparse = coo_matrix(H)
# f_sparse = coo_matrix(f)
# a = richardson_lucy.richardson_lucy_reconstruction(H_sparse, f_sparse)
# H_sparse
# f_sparse

# for k in np.arange(num_iterations):
#     # print(k)
#     # H_sparse = csc_matrix(H)
#     # H_trans = H_sparse.transpose()
#     # I = np.eye(*H.shape)
#     chunk_1 = np.linalg.inv(np.diag(np.diag(np.transpose(H)*np.eye(*H.shape))));chunk_1
#     chunk_2 =  np.diag(np.diag(H.transpose()*np.linalg.inv(np.diagflat(np.matrix(H*g[0])))))
#     g[k+1] = chunk_1 * chunk_2 * g[k]
#
#     image_start = np.reshape(np.array(g[k]),astro_blur.shape)
#     image_now = np.reshape(np.array(g[k+1]),astro_blur.shape)
#     image_real = g_real
#     fig, axarr = plt.subplots(1,3)
#     axarr[0].imshow(image_real)
#     axarr[1].imshow(image_start)
#     axarr[2].imshow(image_now)
#     plt.show()
#
# xfrom scipy.linalg import toeplitz
# aa  = toeplitz([1,2,3], [1,4,5,6])
# aa.T




scale = 4.0
# scale = 1.0
# int(12/scale)
psf = np.ones((int(12/scale),int(12/scale)))/int(12/scale)**2
astro = rescale(color.rgb2gray(data.astronaut()),1.0/scale);
astro_blur = conv2(astro, psf, 'same')# astro_blur = rescale(astro_blur, 1.0 / 4)

# Add Noise to Image
astro_noisy = astro_blur.copy()
astro_noisy += (np.random.poisson(lam=25, size=astro_blur.shape) - 10) / 255.
# astro_blur
# deconvolved_RL = restoration.richardson_lucy(astro_blur, psf, iterations=100)
astro_blur = astro_noisy
plt.imshow(astro_noisy)
plt.imshow(psf)

# plt.imshow(deconvolved_RL)
#
# H = [h_i,j] each row is a line of PSF centered on x,y filled with zeros
N_v = np.ma.size(astro);N_v
N_p = np.ma.size(astro_blur);N_p
H = matrix(np.zeros((N_p,N_v)))

zero_image = np.zeros_like(astro)
astro_shape = astro.shape

# H.shape
for i in np.arange(N_v):
    delta_image = np.zeros_like(astro)
    delta_image[np.unravel_index(i,astro_shape)] = 1
    delta_PSF = scipy.ndimage.convolve(delta_image,psf)
    H[i,:] = delta_PSF.flatten()
    # delta_PSF = psf_xy
    # plt.imshow(delta_image)
    # plt.show()
f = np.matrix(astro_noisy.flatten()).transpose();f
plt.imshow(H)
# plt.show()
# plt.imshow(psf)
plt.imshow(delta_image)

x0 = None
Rtol = 1e-6
NE_Rtol = 1e-6
max_iter = 100
sigmaSq = 0.0
beta = 0.0


A = scipy.sparse.linalg.aslinearoperator(H);A
# A = scipy.sparse.linalg.aslinearoperator(scipy.sparse.identity(f.shape[0]))
# A.todense()
b = f
#%%
# H
import richardson_lucy
import reload

# import importlib
# importlib.reload(richardson_lucy)

import pandas as pd
H_df = pd.DataFrame(H)

#Remove all random sampled rows
beads = 100

rows_to_nuke = np.random.choice(np.arange(H.shape[0]),H.shape[0]-beads)
# rows_to_nuke = (np.rint(np.random.choice(H.shape[0]-beads))*H.shape[0]).astype(int)
H_nuked =H.copy()

for i in rows_to_nuke:
    delta_image = np.zeros_like(astro)
    delta_image[np.unravel_index(i,astro_shape)] = 1
    psf_nan = np.zeros_like(psf)*np.NaN;psf_nan
    delta_PSF = scipy.ndimage.convolve(delta_image,psf_nan)
    # plt.imshow(delta_PSF)
    H_nuked[i,:] = delta_PSF.flatten()
    # delta_PSF = psf_xy
    # plt.imshow(delta_image)
    # plt.show()
plt.imshow(H_nuked)
plt.savefig("output/H_nuked.png")

from sklearn.preprocessing import Imputer
# from sklearn.experimental import enable_iterative_imputer
#
# from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean',verbose=1)
imp.fit(H_nuked)
H_fixed = imp.transform(H_nuked)
# print()

error = H_nuked - H_fixed
sum_error = np.sum(np.sum(error))
sum_error
astro_rl = astro
# plt.show()




rows_to_nuke=np.random.choice(np.arange(0,H.shape[0]), H.shape[0]-beads, replace=False);rows_to_nuke
astro_rl_flat = richardson_lucy.matrix_reconstruction(scipy.sparse.linalg.aslinearoperator(H),f,max_iter=30)
astro_rl_flat = astro_rl
astro_rl = np.reshape(np.array(astro_rl_flat),astro_blur.shape)
plt.imshow(astro_rl)
