import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
from os import listdir
from os.path import isfile, join

num_octaves = 4
octave_size = 5
k = 2**(1/2)
sigma = 1.6
thres = 0.03

def display_octaves(octaves, w, h):

	fig=plt.figure(figsize=(8, 8))
	columns = w
	rows = h
	cnt = 1
	for i in range(columns):
		for j in range(rows):
		    img = octaves[i][j,:,:]
		    fig.add_subplot(columns, rows, cnt)
		    cnt += 1
		    plt.imshow(img)
	
	plt.show()


def DoG(img):
	X, Y = img.shape

	# Generate all octaves
	octaves = list()
	for i in range(num_octaves):
		sig = (k**(2*i))*sigma
		octave = list()
		img = cv2.resize(img, (int(X/(2**i)) , int(Y/(2**i))), interpolation = cv2.INTER_AREA)
		for j in range(octave_size):
			filter_size = 0
			octave.append(cv2.GaussianBlur(img,(0, 0),(k**j)*sig))
		vol = np.stack(octave, axis = 0)
		octaves.append(vol)

	# hehe. generating DoG of Gaussians above.
	dogtaves = list()
	for i in octaves:
		dogtaves_int = list()
		for j in range(1, i.shape[0]):
			dogtaves_int.append(np.subtract(i[j, :, :], i[j-1, :, :]))
		vol = np.stack(dogtaves_int, axis = 0)
		dogtaves.append(vol)

	# display_octaves(dogtaves, 4, 4)
	# print("DoG created!")
	return dogtaves

def is_maxima(cube, thresh):
	# cube = cube / 255.0
	result = np.amax(cube)
	if result >= thresh:
		z, x, y = np.unravel_index(cube.argmax(), cube.shape)
		# print((x, y, z))
		if (x == 1 and y == 1 and z == 1):
			# yes this is the centre of the cube.
			return True

	return False

def hessian(x):
	# https://stackoverflow.com/questions/31206443/numpy-second-derivative-of-a-ndimensional-array
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian

def is_less_strong_edge(hessian, r):
	a = np.square(np.trace(hessian)) / np.linalg.det(hessian)
	if (a < float(((r+1)**2)/r)):
		return True
	else:
		return False


def scale_space_extrema(dogtaves):
	# this also includes keypoint localisation.

	# neigbour comparison
	coord_octaves = list()
	for sp in dogtaves:
		z, x, y = sp.shape
		coord_octs = list()
		for k in range(1, z - 1):
			hess = hessian(sp[k,:,:])
			for i in range(1, x - 1):
				for j in range(1, y - 1):
					cube = sp[k-1:k+2, i-1:i+2, j-1:j+2]
					# check for maxima and thresholding and edge removal
					hes = hess[:,:,i,j]
					if (is_maxima(cube, thres) and is_less_strong_edge(hes, 10)):
						coord_octs.append(np.array([k, i, j]))

		# print(len(coord_octs))
		if (len(coord_octs) != 0):
			# vol = np.stack(coord_octs, axis = 0)
			coord_octaves.append(coord_octs)

	# print(coord_octaves)
	# print("Keypoints localized")
	return coord_octaves

def hogger(dog_image, sigma):
	num_bins = 36
	gx = cv2.Sobel(dog_image, cv2.CV_64F, 1, 0, ksize=1)
	gy = cv2.Sobel(dog_image, cv2.CV_64F, 0, 1, ksize=1)
	mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees = True)
	hist = np.zeros(num_bins)

	N = 2*(int(1.5*sigma)) + 1
	X, Y = dog_image.shape
	blurred = cv2.GaussianBlur(mag, (N, N), 1.5*sigma)
	for i in range(X):
		for j in range(Y):
			ang = angle[i, j]
			mg = blurred[i, j]
			hist[int(np.round((ang * num_bins)/360. )) % num_bins] += mg

	mx = np.max(hist)

	orientations = []
	for idx, peaks in enumerate(hist):
		if float(peaks / mx) > 0.85:
			orientations.append((idx*360)/num_bins + 360/(2 * num_bins))

	return orientations



def orientation_assignment(coord_octaves, dogtaves):
	octave_idx = 0
	orientations = list()
	for octave in coord_octaves:
		oct_orientations = list()
		for points in octave:
			scale, x, y = points
			inc = int(np.round(2.5*scale))
			mn_x = x - inc
			mx_x = x + inc
			mn_y = y - inc
			mx_y = y + inc

			if (mn_x < 0):
				mn_x = 0
			if (mx_x > dogtaves[octave_idx].shape[1]):
				mx_x = dogtaves[octave_idx].shape[1]
			if (mn_y < 0):
				mn_y = 0
			if (mx_y > dogtaves[octave_idx].shape[2]):
				mx_y = dogtaves[octave_idx].shape[2]
			

			patch = dogtaves[octave_idx][scale, mn_x : mx_x, mn_y : mx_y]
			# the angles have been calculated
			ors = hogger(patch, (k**scale)*sigma)
			oct_orientations.append([scale, x, y, ors])

		orientations.append(oct_orientations)
		octave_idx += 1

	# print("orientations assigned.")
	return orientations

def getFeatureVectors(region):
	# divide in 4 regions
	gx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=1)
	gy = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=1)
	dx = np.zeros((4, 4))
	dy = np.zeros((4, 4))
	dx_a = np.zeros((4, 4))
	dy_a = np.zeros((4, 4))

	for i in range(4):
		for j in range(4):
			start_x = i*4 
			end_x = start_x + 4
			start_y = j*4
			end_y = start_y + 4

			patch_x = gx[start_x:end_x, start_y:end_y]
			patch_y = gy[start_x:end_x, start_y:end_y]

			dx += patch_x
			dy += patch_y
			dx_a += np.fabs(patch_x)
			dy_a += np.fabs(patch_y)

	vec = np.concatenate((dx.flatten(), dy.flatten(), dx_a.flatten(), dy_a.flatten()))
	vec2 = np.square(vec)
	norm = (np.sum(vec2))**(0.5)

	# normalization required.
	return (vec / norm)


def descriptor(orientations, dogtaves):
	# getting a 16x16 window
	features = list()
	sz = 16
	for idx, oct_orients in enumerate(orientations):
		# idx is the octave index.
		ig = dogtaves[idx]
		z, x, y = ig.shape
		for points in oct_orients:
			s = points[0]
			p_x = points[1]
			p_y = points[2]
			angles = points[3]
			img = ig[s,:,:]
			for a in angles:
				# go along sin direction and cosine direction for getting tilted rectangle
				region = np.zeros((sz, sz))

				a_rad = a*(3.14 / 180.)
				sn = math.sin(a_rad)
				cs = math.cos(a_rad)

				offset = sz / 2
				# rotation transform for the centre pixels
				start_x = p_x + offset*cs + offset*sn
				start_y = p_y - offset*sn + offset*cs

				for i in range(sz):
					pix_x = start_x
					pix_y = start_y

					for j in range(sz):
						xx = min(max(int(round(pix_x)), 0), y-1)
						yy = min(max(int(round(pix_y)), 0), x-1)
						region[i, j] = img[yy, xx]

						# updates
						pix_x += cs
						pix_y -= sn

					# updates
					start_x += sn
					start_y += cs

				# the region is now made
				ft = getFeatureVectors(region)
				features.append(ft)

	# print("Feature Vector Made!!!")
	return features

def surf(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = img/255.

	dogtaves = DoG(img)
	init_keypoints = scale_space_extrema(dogtaves)
	orient = orientation_assignment(init_keypoints, dogtaves)
	descriptors = descriptor(orient, dogtaves)

	return descriptors

def store(pth = "./surf/"):
	mypath = './images/'
	names = [f for f in listdir(mypath) if isfile(join(mypath, f))]

	idx = 1
	for i in names:
		kp, res = surf(cv2.imread(mypath + i),None)
		pickle.dump(res, open(pth + i[:-4] + '.p', 'wb'))
		print(idx)
		idx += 1

def main():
	img = cv2.imread('images/all_souls_000001.jpg')
	features = surf(img)
	print(features)

if __name__ == '__main__':
	main()
	# store()