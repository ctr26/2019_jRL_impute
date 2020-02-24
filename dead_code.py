
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
