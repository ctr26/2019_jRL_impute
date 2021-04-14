# %%
# !conda create --name py37 pims numpy scipy matplotlib \
#   python=3.7 jpype1 scikit-image pillow \
#   zarr hdf5
#   ipykernel --yes
# conda install -c conda-forge zarr
# conda install -c anaconda hdf5

# # %%
# from dask.distributed import Client

# client = Client(processes=False, silence_logs=False)

import numpy as np
import PIL
import pims
import matplotlib.pyplot as plt
from os.path import expanduser
import scipy.ndimage as ndimage
from skimage.exposure import equalize_adapthist


# %%

print("Bead folder")

image_file = "~/+projects/2019_jrl/2019_jRl_impute/data/CraigDeconvolutionData/200904_14.02.08_Step_Size_-0.4_Wavelength_DAPI 452-45_FL-15_CUBIC-1_pipetteTipMount/MMStack.ome.tif"

beads_file = "~/+projects/2019_jrl/2019_jRL_impute/data/CraigDeconvolutionData/2020-09-04 - calibration/beads/200904_16.50.57_Step_Size_-0.4_Wavelength_DAPI 452-45_500nm_TetBeads/MMStack.ome.tif"
# beads_file = "~/gdrive/+projects/2019_jrl/2019_jRl_impute/data/CraigDeconvolutionData/200904_14.02.08_Step_Size_-0.4_Wavelength_DAPI 452-45_FL-15_CUBIC-1_pipetteTipMount/MMStack.ome.tif"

print("Expanding dir")
beads_file = expanduser(beads_file)


psf_window = (140, 40, 40)  # (z,y,x)
psf_window = [60, 20, 20]

# %%


tif_stack = pims.open(beads_file)

# psf_image = np.array(tif_stack)

dtype = np.uint8


def scaleImage(image, dtype=np.uint8):
    image = np.array(image)
    scaled = image / image.max()
    scaled_255 = scaled * (np.iinfo(dtype).max)

    scaled_255_8bit = scaled_255.astype(dtype)
    # output = scaled_255_8bit
    return scaled_255_8bit


psf_image_scaled = scaleImage(tif_stack, dtype)
# %%

# eq_scaled_255_float = equalize_adapthist(psf_image_scaled)
# eq_scaled_255_8bit = eq_scaled_255_float*((np.iinfo(dtype)).max()).astype(dtype)
# %%


def getPSFcoords(image, window, sigma=6):
    # scaled = image / image.max()

    # plt.imshow(eq_scaled_255_8bit.max(axis=0))

    img = image

    # Get local maximum values of desired neighborhood
    # I'll be looking in a 5x5x5 area
    print("Max filtering")
    img_max = ndimage.maximum_filter(img, size=np.divide(psf_window, 2))

    # Threshold the image to find locations of interest
    # I'm assuming 6 standard deviations above the mean for the threshold
    print("Thresholding")
    img_thresh = img_max.mean() + img_max.std() * sigma

    # Since we're looking for maxima find areas greater than img_thresh
    print("Labelling")
    labels, num_labels = ndimage.label(img_max > img_thresh)

    # Get the positions of the maxima
    print("Getting coords from centre of mass")
    coords = ndimage.measurements.center_of_mass(
        img, labels=labels, index=np.arange(1, num_labels + 1)
    )

    # Get the maximum value in the labels
    values = ndimage.measurements.maximum(
        img, labels=labels, index=np.arange(1, num_labels + 1)
    )
    return coords


print("Getting coords")

coords = getPSFcoords(psf_image_scaled, psf_window)
# %%
from skimage import util

# img = psf_image
img = psf_image_scaled


def cropND(img, centre, window):

    centre = np.array(centre)
    centre_dist = np.divide(window, 2)
    shape = img.shape
    crops = []
    # for i, dim in enumerate(shape):
    #     l = (centre[-(i + 1)] - np.floor(centre_dist[-(i + 1)])).astype(int)
    #     r = (centre[-(i + 1)] + np.ceil(centre_dist[-(i + 1)])).astype(int)

    x_l = (centre[-1] - np.floor(centre_dist[-1])).astype(int)
    x_r = (centre[-1] + np.ceil(centre_dist[-1])).astype(int)

    y_l = (centre[-2] - np.floor(centre_dist[-2])).astype(int)
    y_r = (centre[-2] + np.ceil(centre_dist[-2])).astype(int)

    z_l = (centre[-3] - np.floor(centre_dist[-3])).astype(int)
    z_r = (centre[-3] + np.ceil(centre_dist[-3])).astype(int)
    # try:
    #     return util.crop(img,((z_l,z_r),(y_l,y_r),(x_l,x_r)))
    # except :
    #     return

    return img[z_l:z_r, y_l:y_r, x_l:x_r]


def cropNDv(img, centres, window=[20, 40, 40]):
    cropped_list = np.full((len(centres), *window), np.nan)
    i = 0
    centres_list = []
    for centre in centres:
        try:
            cropped_list[i, :, :, :] = cropND(img, centre, window=window)
            centres_list.append(centres[i])
            i += 1
        except:
            None
    return cropped_list, centres_list


cropped, coord_list = cropNDv(img, centres=coords, window=psf_window)
# [~np.isnan(cropped_list).any(axis=0)]
# %%

plt.imshow(cropped[700, 10, :, :])
plt.imshow(np.max(cropped[70, :, :, :], axis=1))
# %%
sum_cropped = np.sum(cropped, axis=(1, 2, 3))
plt.hist(sum_cropped)
# sum_thresh = flat.sum(axis=1).mean(axis=0) + flat.sum(axis=1).std(axis=0) * 6
# %%
from sklearn.decomposition import PCA
import pandas as pd

index = pd.MultiIndex.from_arrays(
    np.array(coord_list).transpose(), names=("z", "y", "x")
)
flat = pd.DataFrame(cropped.reshape(cropped.shape[0], -1)).dropna(0).set_index(index)
# %%
flat.sum(axis=1).hist()

low_thres = (flat.sum(axis=1).mean()) - 2 * (flat.sum(axis=1).std())
up_thres = (flat.sum(axis=1).mean()) + 2 * (flat.sum(axis=1).std())

flat_clipped = flat[(flat.sum(axis=1) > low_thres) & (flat.sum(axis=1) < up_thres)]
# sum_cropped = np.sum(cropped,axis=(1,2,3))
# plt.hist(sum_cropped)
flat_clipped.sum(axis=1).hist()

df = flat
# %% Apply Min Max scaler
# Probably shouldn't do this, especially for
# systems with varying background intensity.
df = flat
from sklearn.preprocessing import minmax_scale

# Assume noise level low
# df = flat_clipped

scaled_df = df.apply(minmax_scale, axis=1, result_type="broadcast")

df = scaled_df
# %% Apply Min Max scaler
df = scaled_df
from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.01, random_state=42).fit(df)
outliers = clf.predict(df)
df = df[outliers == 1]
print(f"Dropped {np.sum(outliers==1)} outliers")

# %%
pca = PCA(n_components=0.99, whiten=True).fit(df)
pca_df = pd.DataFrame(pca.transform(df), index=df.index)
plt.scatter(pca_df[0], pca_df[1])
plt.show()
# pca.fit_transform(cropped)
n_components = 20
# include average
eigen_psfs = pca.components_.reshape((-1, *psf_window))
cum_sum_exp_var = np.cumsum(pca.explained_variance_ratio_)
accuracy = cum_sum_exp_var[n_components]

i_len = 5
j_len = 5

j_steps = np.floor_divide(psf_window[0], j_len)
plt.imshow(np.max(eigen_psfs[0, :, :, :], axis=1))
plt.show()

fig, ax = plt.subplots(i_len, j_len)
for i in range(i_len):
    for j in range(j_len):
        ax[i, j].imshow(eigen_psfs[i, j * j_steps, :, :])
        # ax[i,0].imshow(eigen_psfs[1,0,:,:])
        # ax[0,1].imshow(eigen_psfs[0,10,:,:])
        # ax[1,1].imshow(eigen_psfs[1,10,:,:])
plt.show()
plt.plot(cum_sum_exp_var[0:n_components])
plt.title(f"{str(accuracy)})")
plt.show()
fig, ax = plt.subplots(1, i_len)
for i in range(i_len):
    ax[i].imshow(np.max(eigen_psfs[i, :, :, :], axis=1))
    # ax[i,0].imshow(eigen_psfs[1,0,:,:])
    # ax[0,1].imshow(eigen_psfs[0,10,:,:])
    # ax[1,1].imshow(eigen_psfs[1,10,:,:])
# plt.show()
# plt.imshow(np.sum(eigen_psfs[0, :, :, :], axis=1))  # %%
plt.show()
plt.imshow(np.max(np.array(df.iloc[0, :]).reshape(psf_window), axis=1))



# %%
import seaborn as sns

# xy_coords = np.array(coord_list)[:, [0, 2]]
# x_list = xy_coords[:, 0]
# y_list = xy_coords[:, 1]

x_list = pca_df.index.get_level_values("x")
y_list = pca_df.index.get_level_values("y")
z_list = pca_df.index.get_level_values("z")

pc_weights = np.sqrt((pca_df.iloc[:, 0:] ** 2).sum(axis=1))
pc_weights = pca_df[0]
pc_weights = pca_df

# pc_weights[pc_weights>500] = np.nan

sns.scatterplot(x=x_list, y=y_list, hue=pc_weights[0])
# plt.plot(xy_coords[0,:],xy_coords[1,:],c=pc_weights)
# xy_coords[:,0], xy_coords[:,1], pc_weights
# %%

from scipy.interpolate import interp2d

f = interp2d(x_list, y_list, pc_weights[0])
plt.hist(pc_weights[0])
plt.show()
# x_coords = np.arange(min(x_list), max(x_list) + 1)
# y_coords = np.arange(min(y_list), max(y_list) + 1)
# z_coords = np.arange(min(z_list), max(z_list) + 1)

z_coords = np.linspace(min(z_list), max(z_list), num=psf_image_scaled.shape[0])
y_coords = np.linspace(min(y_list), max(y_list), num=psf_image_scaled.shape[1])
x_coords = np.linspace(min(x_list), max(x_list), num=psf_image_scaled.shape[2])


c_i = np.log(f(x_coords, y_coords))

plt.imshow(c_i)
plt.show()
# %%
# from distributed import Client
# client = Client(processes=False);display(client)
# %% rbf
from scipy.interpolate import Rbf
import dask.array as da

# rbfi = Rbf(z_list, y_list, x_list, pc_weights, function="cubic")
# # interped_rbf = rbfi(x_coords,y_coords,z_coords)
# # plt.imshow(interped_rbf[:,:,0])
# # plt.show()
# # interped_rbf_grid = rbfi(grid_x[0], grid_y[0])
# grid_z, grid_y, grid_x = np.mgrid[
#     min(z_list) : max(z_list) : 100j,
#     min(y_list) : max(y_list) : 200j,
#     min(x_list) : max(x_list) : 200j,
# ]
# dask_xyz = da.from_array((grid_z, grid_y, grid_x), chunks=20)
# dask_xyz
# # def rbfi_da(dask_xyz,rbfi):
# # dask_xx = dask_xyz[0,:,:,:]
# # dask_yy = dask_xyz[1,:,:,:]
# # dask_zz = dask_xyz[2,:,:,:]
# # return rbfi(dask_xx,dask_yy,dask_zz)
# # out = dask.delayed(rbfi)(dask_xx,dask_yy,dask_zz)
# # %%
# from dask.diagnostics import ProgressBar

# with ProgressBar():
#     f = da.map_blocks(
#         rbfi, dask_xyz[0, :, :, :], dask_xyz[1, :, :, :], dask_xyz[2, :, :, :]
#     )
#     # g = client.persist(f)
#     g = f.compute()
# # %%
# # plt.imshow(f(x_coords, y_coords))
# # plt.show()

# plt.imshow(g[50, :, :])
# plt.show()
# plt.imshow(g[0, :, :])
# plt.show()
# plt.imshow(g[-1, :, :])
# plt.show()

# plt.imshow(np.max(g, axis=0))
# plt.show()

# %%
# from scipy.interpolate import griddata

# grid_x, grid_y, grid_z = np.mgrid[
#     min(x_list) : max(x_list) : 200j,
#     min(y_list) : max(y_list) : 200j,
#     min(z_list) : max(z_list) : 200j,
# ]
# grid_0 = griddata(
#     (x_list, y_list, z_list), pc_weights, (grid_x, grid_y, grid_z), method="nearest"
# )
# plt.imshow(grid_0[100, :, :])
# %% rbf


# %%
# plt.imshow(np.log(grid_0[:, :, 100]))
# plt.show()
# plt.imshow((grid_0[:, :, 100]))
# plt.show()


# %%
# from scipy.interpolate import interpn


# from scipy.ndimage import gaussian_filter

# grid_0_smooth = gaussian_filter(grid_0, sigma=(np.array(psf_window) / 2).mean())

# # plt.imshow(np.log(grid_0[:,:,100]))
# # plt.show()
# plt.imshow((grid_0_smooth[:, :, 30]))
# plt.show()
# %%
melt_df = pd.melt(
    pca_df.iloc[:, 0:n_components],
    var_name="PC",
    value_name="Weight",
    ignore_index=False,
).reset_index()

from scipy.interpolate import Rbf
import dask.array as da
from dask.diagnostics import ProgressBar

# pc_idx,x_list,y_list,z_list = X["PC"],X["x"],X["y"],X["z"]
rbfi = Rbf(
    melt_df["PC"],
    melt_df["z"],
    melt_df["y"],
    melt_df["x"],
    melt_df["Weight"],
    function="cubic",
)

grid_pc, grid_z, grid_y, grid_x  = np.mgrid[
    0:n_components,
    min(z_list) : max(z_list) : 50j,
    min(y_list) : max(y_list) : 100j,
    min(x_list) : max(x_list) : 100j,
]

dask_xyzp = da.from_array((grid_pc, grid_z, grid_y, grid_x), chunks=5)

# Slow
with ProgressBar():
    f = da.map_blocks(
        rbfi,
        dask_xyzp[0, :, :, :],
        dask_xyzp[1, :, :, :],
        dask_xyzp[2, :, :, :],
        dask_xyzp[3, :, :, :],
    )
    # g = client.persist(f)
    g = f.compute()
# %%
plt.imshow(g[0, 25, :, :])
plt.show()
plt.imshow(g[0, 0, :, :])
plt.show()
plt.imshow(g[0, -1, :, :])
plt.show()

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge

# y = melt_df["Weight"].to_numpy()
# X = melt_df["PC"].reset_index()

y = melt_df["Weight"]
X = melt_df.drop("Weight", 1)

X_train, X_test, y_train, y_test = train_test_split(X, y)

est = RandomForestRegressor()
# est = MLPRegressor(max_iter=2000)
est = GaussianProcessRegressor()
# est = KernelRidge()
est.fit(X_train, y_train)

print(
    f"Training score: {est.score(X_train,y_train)} and Pred score: {est.score(X_test,y_test)}"
)
# %%
import scipy.stats as stats
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# param_dist = {"alpha": loguniform(1e-4, 1e0)}

clf = est
clf = RandomForestRegressor()
param_dist = {
    "n_estimators": [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
    "max_features": ["auto", "sqrt"],
    "max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

# run randomized search
# n_iter_search = 20
random_search = RandomizedSearchCV(
    clf,
    param_distributions=param_dist,
    n_iter=100,
    cv=3,
    verbose=100,
    random_state=42,
    n_jobs=-1,
)

random_search.fit(X, y)
random_search.best_score_

# %%

from scipy.stats import randint, uniform

clf = MLPRegressor(max_iter=2000)

param_dist = {
    "hidden_layer_sizes": [
        (
            randint.rvs(100, 600, 1),
            randint.rvs(100, 600, 1),
        ),
        (randint.rvs(100, 600, 1),),
    ],
    "activation": ["tanh", "relu", "logistic"],
    "solver": ["sgd", "adam", "lbfgs"],
    "alpha": uniform(0.0001, 0.9),
    "learning_rate": ["constant", "adaptive"],
}

# run randomized search
# n_iter_search = 20
random_search = RandomizedSearchCV(
    clf,
    param_distributions=param_dist,
    n_iter=100,
    cv=3,
    verbose=100,
    random_state=42,
    n_jobs=-1,
)

random_search.fit(X, y)
random_search.best_score_


# %%
# print(cropped.shape)
# # %%
# import dask_image.imread
# import dask_image.ndmeasure
# import dask_image.ndfilters
# import dask as da

# # %%

# # from dask.distributed import Client
# # client = Client(processes=False, silence_logs=False)

# image_da = dask_image.imread.imread(beads_file)


# def scaleImageDask(image, dtype=np.uint8):
#     scaled = image / image.max()
#     # scaled_255 = scaled * (np.info(dtype).max)
#     scaled_255 = scaled * 255

#     scaled_255_8bit = scaled_255.astype(dtype)
#     # output = scaled_255_8bit
#     return scaled_255_8bit


# def getPSFcoordsDask(image, window=psf_window, sigma=6):
#     # scaled = image / image.max()

#     # plt.imshow(eq_scaled_255_8bit.max(axis=0))

#     img = image

#     # Get local maximum values of desired neighborhood
#     # I'll be looking in a 5x5x5 area
#     img_max = dask_image.ndfilters.maximum_filter(
#         img, size=np.divide(window, 2).astype(int)
#     )

#     # Threshold the image to find locations of interest
#     # I'm assuming 6 standard deviations above the mean for the threshold
#     img_thresh = img_max.mean() + img_max.std() * sigma

#     # Since we're looking for maxima find areas greater than img_thresh

#     labels, num_labels = dask_image.ndmeasure.label(img_max > img_thresh)

#     # Get the positions of the maxima
#     coords = dask_image.ndmeasure.center_of_mass(img, label_image=labels)

#     #  index=da.array.arange(1, num_labels + 1)

#     # Get the maximum value in the labels
#     values = dask_image.ndmeasure.maximum(img, label_image=labels)
#     return coords

# psf_image_scaled = scaleImageDask(image_da, np.uint8)
# # eq_scaled_255_8bit = equalize_adapthist(psf_image_scaled)
# # eq_scaled_255_8bit = psf_image_scaled.map_blocks(equalize_adapthist, dtype=np.uint8)
# coords = getPSFcoordsDask(psf_image_scaled, psf_window)
# # result = coords.compute()
# # result = client.compute(coords)
# result = client.persist(coords)
# # result.result()

# result = getCoords(image_da)
# eq_scaled_255_8bit = apply_parallel(equalize_adapthist, psf_image_scaled)
# eq_scaled_255_8bit = equalize_adapthist(psf_image_scaled)

# %%

# plt.imshow(equalize_adapthist(psf_image.max(axis=0)))

# put a blue dot at (10, 20)
# plt.scatter(np.array(coords)[:, -1], np.array(coords)[:, -2])
# plt.show()

# %%
# cropped = cropND(img, reversed(coords[0]))

# %%
# import dask_image.imread

# image = dask_image.imread.imread(beads_file)

# np_tif_stack = np.array(tif_stack)
# %%
# print("Loading beads")
# # tif_stack = pims.Bioformats(beads_file, java_memory="1024m")
# # image = tif_stack

# print("Making numpy array")
# # np_image = np.array(image)
# np_image = image
# # np_image = np.vectorise(image[0:2])
# print("Scaling to max 1")
# scaled = np_image / np_image.max()
# print("Scaling to  255")
# scaled_255 = scaled * 255
# print("8bit")

# scaled_255_8bit = scaled_255.astype(np.uint8)
# output = scaled_255_8bit

# with ProgressBar():
#     output.compute(memory_limit="8GB")
# %%

# img = psf_image_scaled


# import scipy.ndimage as ndimage
# import dask

# # Get local maximum values of desired neighborhood
# # I'll be looking in a 5x5x5 area
# img_max = dask.delayed(ndimage.maximum_filter)(img, psf_window)

# img_thresh = img_max.mean() + img_max.std() * 6

# label_out = dask.delayed(ndimage.label)(img_max > img_thresh)

# labels = label_out[0]
# num_labels = label_out[1]


# coords = dask.delayed(ndimage.measurements.center_of_mass)(
#     img, labels=labels, index=dask.delayed(np.arange)(1, num_labels + 1)
# )

# values = dask.delayed(ndimage.measurements.maximum)(
#     img, labels=labels, index=dask.delayed(np.arange)(1, num_labels + 1)
# )


# with ProgressBar():
#     raw_coords = coords.compute(memory_limit="8GB")


# # Get the positions of the maxima
# coords = ndimage.measurements.center_of_mass

# values = ndimage.measurements.maximum(img, labels=labels, index=np.arange(1, num_labels + 1))

# values.compute(memory_limit="32GB")
# %%
# labels, num_labels = ndimage.label(img_max > img_thresh)

# Get the positions of the maxima
# coords = ndimage.measurements.center_of_mass(
#     img, labels=labels, index=np.arange(1, num_labels + 1)
# )

# Get the maximum value in the labels
# values = ndimage.measurements.maximum(
#     img, labels=labels, index=np.arange(1, num_labels + 1)
# )

# np.save("output/np_image_scaled01", scaled_255_8bit)
# %%
# print("Saving")
# from dask.diagnostics import ProgressBar

# with ProgressBar():
#     output.to_zarr(
#         "output/dask_image_scaled01.zarr", overwrite=True, memory_limit="32GB"
#     )
#     # output.to_hdf5("output/dask_image_scaled01.zarr", memory_limit="32GB")

# # output.compute(memory_limit="64GB")


# %%
##################print("Saving")
# from dask.diagnostics import ProgressBar

# with ProgressBar():
#     output.to_zarr(
#         "output/dask_image_scaled01.zarr", overwrite=True, memory_limit="32GB"
#     )
#     # output.to_hdf5("output/dask_image_scaled01.zarr", memory_limit="32GB")

# # output.compute(memory_limit="64GB")

# # from scipy import ndimage, misc

# # result = ndimage.maximum_filter(image, size=np.divide(psf_window, 4).astype(int))

# # %%
# # max_z_filtered = result.max(0)
# # max_z = np_tif_stack.max(0)
# # %%
# # %%

# plt.imshow(max_z)
# # %%
# plt.imshow(max_z_filtered)
# # %%

# # %%
# from skimage import data, feature

# localisations = feature.blob_log(np_tif_stack, threshold=0.5, max_sigma=40)

# # %%
# fig, axes = plt.plot()

# axes.imshow(max_z_filtered)

# for blob in blobs:
#     z, ay, x, r = blob
#     c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
#     axes.add_patch(c)
# axes.set_axis_off()

# plt.tight_layout()
# plt.show()
# %%