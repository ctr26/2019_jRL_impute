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
#
# '''
# Richardson-Lucy algorithm
#
# Ported from the RestoreTools MATLAB package available at:
# http://www.mathcs.emory.edu/~nagy/RestoreTools/
#
# Input: A  -  object defining the coefficient matrix.
#        b  -  Right hand side vector.
#
#  Optional Intputs:
#
#        x0      - initial guess (must be strictly positive); default is x0 = A.T*b
#        sigmaSq - the square of the standard deviation for the
#                  white Gaussian read noise (variance)
#        beta    - Poisson parameter for background light level
#        max_iter - integer specifying maximum number of iterations;
#                   default is 100
#        Rtol    - stopping tolerance for the relative residual,
#                  norm(b - A*x)/norm(b)
#                  default is 1e-6
#        NE_Rtol - stopping tolerance for the relative residual,
#                  norm(A.T*b - A.T*A*x)/norm(A.T*b)
#                  default is 1e-6
#
# Output:
#       x  -  solution
#
# Original MATLAB code by J. Nagy, August, 2011
#
# References:
# [1]  B. Lucy.
#     "An iterative method for the rectication of observed distributions."
#      Astronomical Journal, 79:745-754, 1974.
# [2]  W. H. Richardson.
#     "Bayesian-based iterative methods for image restoration.",
#      J. Optical Soc. Amer., 62:55-59, 1972.
#  [3]  C. R. Vogel.
#     "Computational Methods for Inverse Problems",
#     SIAM, Philadelphia, PA, 2002
#
#
# '''

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
# import matlab.engine
import pandas as pd


from sklearn.preprocessing import Imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# from sklearn.experimental import enable_iterative_imputer
from scipy import signal
from sklearn.impute import SimpleImputer

import os

#%% Creating the test card image

scale = 4.0
# scale = 1.0
# int(12/scale)
static_psf = np.ones((int(12/scale),int(12/scale)))/int(12/scale)**2 #Boxcar
def psf_guass(w=10,h=10,sigma=3):
    # blank_psf = np.zeros((w,h))
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    xx,yy = np.meshgrid(np.linspace(-1,1,w),np.linspace(-1,1,h))
    return gaussian(xx,0,sigma)*gaussian(yy,0,sigma)

static_psf =  psf_guass(w=10, h=10, sigma=1/5)
plt.imshow(static_psf)

astro = rescale(color.rgb2gray(data.astronaut()),1.0/scale);
astro_blur = conv2(astro, static_psf, 'same')# astro_blur = rescale(astro_blur, 1.0 / 4)

# Add Noise to Image
astro_noisy = astro_blur.copy()
astro_noisy += (np.random.poisson(lam=25, size=astro_blur.shape) - 10) / 255.
# astro_blur
# deconvolved_RL = restoration.richardson_lucy(astro_blur, psf, iterations=100)
astro_blur = astro_noisy
plt.imshow(astro_noisy)
plt.imshow(static_psf)

# plt.imshow(deconvolved_RL)

#%% Build measurement matrix.
print("Build measurement matrix.")
##s H = [h_i,j] each row is a line of PSF centered on x,y filled with zeros

N_v = np.ma.size(astro);N_v
N_p = np.ma.size(astro_blur);N_p
measurement_matrix = matrix(np.zeros((N_p,N_v)))

zero_image = np.zeros_like(astro)
astro_shape = astro.shape

x_astro,y_astro = astro_blur.shape
xx_astro,yy_astro = np.meshgrid(np.linspace(-1,1,x_astro),np.linspace(-1,1,y_astro))
psf_window_w,psf_window_h = (10,10)
psf_window_volume = np.full((psf_window_w,psf_window_h,N_v),np.NaN)


for i in np.arange(N_v):
    coords = np.unravel_index(i,astro.shape)
    r_dist = np.sqrt(xx_astro[coords]**2+yy_astro[coords]**2)
    psf_current = psf_guass(w=psf_window_w, h=psf_window_h, sigma=1/(5+5*r_dist))
    psf_window_volume[:,:,i] = psf_current
    delta_image = np.zeros_like(astro)
    delta_image[np.unravel_index(i,astro_shape)] = 1
    delta_PSF = scipy.ndimage.convolve(delta_image,psf_current)
    measurement_matrix[i,:] = delta_PSF.flatten()
    # plt.imshow(delta_image)
    # plt.show()
astro_noisy_vector = np.matrix(astro_noisy.flatten()).transpose();astro_noisy_vector
plt.imshow(measurement_matrix)
# plt.show()
plt.imshow(static_psf)

#%% Begin RL matrix deconvolvution
print("Regress full PSF model")

beads = 100

rows_to_nuke = np.random.choice(np.arange(measurement_matrix.shape[0]),measurement_matrix.shape[0]-beads)
# rows_to_nuke
psf_window_volume_nuked = psf_window_volume.copy()
psf_window_volume_nuked[:,:,rows_to_nuke] = np.NaN

X_indices = np.array(np.unravel_index(np.arange(0,psf_window_volume.size), psf_window_volume.shape)).T
y_values = np.array(psf_window_volume_nuked.flatten())

# from scipy import interpolate
# rbfi = interpolate.Rbf(X_indices[:,0],X_indices[:,1],X_indices[:,2],y_values)

X_indices_clean = X_indices[np.isfinite(y_values)]
y_values_clean = y_values[np.isfinite(y_values)]

y_values_clean_2d = np.vstack(y_values_clean)

from sklearn import linear_model
from sklearn import svm
from scipy.stats import pearsonr
from sklearn import neural_network
from sklearn import metrics
from sklearn import gaussian_process
from sklearn import preprocessing
from sklearn import svm

classifiers = [
    # svm.SVR(),
    neural_network.MLPRegressor(hidden_layer_sizes=(64,64),
                                verbose=True),
    # svm.SVR(),
    # gaussian_process.GaussianProcessRegressor(),
    # linear_model.SGDRegressor(),
    # linear_model.BayesianRidge(),
    # linear_model.LassoLars(),
    # linear_model.ARDRegression(),
    # linear_model.PassiveAggressiveRegressor(),
    # linear_model.TheilSenRegressor(),
    # linear_model.LinearRegression()
    ]

#Maybe standard scalar first then reverse.

y_ground_truth = psf_window_volume.flatten()

X_indices_clean_scaled = preprocessing.scale(X_indices_clean)
X_indices_scaled = preprocessing.scale(X_indices)

y_values_clean_2d_scaled = preprocessing.scale(y_values_clean_2d)
y_ground_truth_scaled = preprocessing.scale(y_ground_truth)

for classifier in classifiers:
    # print(classifier)
    name = classifier.__module__;name
    print(f'{name}')
    classifier.fit(X_indices_clean_scaled, y_values_clean_2d_scaled)
    y_values_predict_scaled = classifier.predict(X_indices_scaled)
    score = classifier.score(X_indices_scaled,y_ground_truth_scaled)
    mse = metrics.mean_squared_error(y_ground_truth_scaled,y_values_predict_scaled)
    r2 = metrics.r2_score(y_ground_truth_scaled,y_values_predict_scaled)
    # classifier.score()
    correlation,p_value = pearsonr(y_ground_truth_scaled,y_values_predict_scaled)
    print(f'Correlation: {correlation:.5f} | MSE:{mse:.5f} |  R2:{r2:.5f}  | Score:{score:.5f}')
plt.scatter(y_ground_truth_scaled,y_values_predict_scaled)
score
correlation
# from sklearn.neural_network import MLPClassifier


#%% Begin RL matrix deconvolvution
print("Build measurement matrix.")
#
# x0 = None
# Rtol = 1e-6
# NE_Rtol = 1e-6
# max_iter = 100
# sigmaSq = 0.0
# beta = 0.0

measurement_matrix_LO = scipy.sparse.linalg.aslinearoperator(measurement_matrix)
input_vector = astro_noisy_vector

# Raw RL, no imputation
FLAG_RAW = 0
if(FLAG_RAW):
    astro_rl_flat = richardson_lucy.matrix_reconstruction(
                    scipy.sparse.linalg.aslinearoperator(measurement_matrix),
                    input_vector,max_iter=30)
    # astro_rl = astro
    astro_rl = np.reshape(np.array(astro_rl_flat),astro_blur.shape)

    fig, ax = plt.subplots(1,3,figsize=[6.4*2, 4.8*2])

    ax[0].imshow(astro)
    ax[0].title.set_text("Raw")

    ax[1].imshow(astro_noisy)
    ax[1].title.set_text("Corrupted")

    ax[2].imshow(astro_rl)
    ax[2].title.set_text("Recovered")


#%% Matrix nuking

# H_df = pd.DataFrame(measurement_matrix)

#Remove all random sampled rows
beads = 100

rows_to_nuke = np.random.choice(np.arange(measurement_matrix.shape[0]),measurement_matrix.shape[0]-beads)
# rows_to_nuke=np.random.choice(np.arange(0,measurement_matrix.shape[0]), measurement_matrix.shape[0]-beads, replace=False);rows_to_nuke
# rows_to_nuke = (np.rint(np.random.choice(H.shape[0]-beads))*H.shape[0]).astype(int)
H_nuked = measurement_matrix.copy()

for i in rows_to_nuke:
    ones_image = np.ones_like(astro)
    delta_image = np.zeros_like(astro)
    delta_image[np.unravel_index(i,astro_shape)] = 1
    psf_ones = np.ones_like(static_psf);
    # psf_complex[psf_complex==1j] = np.NaN
    delta_PSF = scipy.ndimage.convolve(delta_image,psf_ones)
    delta_PSF[delta_PSF!=0] = np.NaN
    # plt.imshow(delta_PSF)
    H_nuked[i,:] = delta_PSF.flatten()
    # delta_PSF = psf_xy
    # plt.imshow(delta_image)
    # plt.show()

H_nuked_diag = np.diag(H_nuked)
H_nuked_diag.shape
nans_in_H = np.sum(np.isnan(H_nuked_diag))
nans_not_in_H = np.sum(~np.isnan(H_nuked_diag))

pd_nan = (nans_not_in_H-nans_in_H)/2*(nans_in_H+nans_not_in_H);
ratio = nans_in_H/H_nuked_diag.shape
print(f'Nans in H: {nans_in_H} | Ratio: {ratio}')

plt.imshow(H_nuked)
plt.imsave('./output/H_nuked.png', H_nuked)
# imp = SimpleImputer(missing_values=np.NaN, strategy='mean',verbose=1)
from sklearn.linear_model import BayesianRidge
#%% Matrix impute

imp = IterativeImputer(missing_values=np.NaN,verbose=2,estimator=BayesianRidge())
imp.fit(H_nuked)
H_fixed = imp.transform(H_nuked)

error = measurement_matrix - H_fixed
sum_error = np.sum(np.sum(error))
sum_error
# plt.show()
plt.savefig("output/H_fixed.png")
plt.imshow(H_fixed)
