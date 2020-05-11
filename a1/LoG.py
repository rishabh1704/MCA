import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
from os import listdir
from os.path import isfile, join

def LoG(img, sigma):
	# apply gaussian filter then laplacian
	blur = cv2.GaussianBlur(img,(0,0),sigma)
	laplacian = cv2.Laplacian(blur,cv2.CV_64F)
	# laplacian = np.pad(laplacian,((1,1),(1,1)),'constant')
	return np.square(laplacian)

def generate_volume(img, start_sigma, k):
	levels = 10
	ll = list()

	for i in range(levels):
		sig = (k**i) * start_sigma
		log = LoG(img, sig)
		# cv2.imshow('t',log)
		# cv2.waitKey(0)
		ll.append(log)
	
	# vol = np.array([i for i in ll])
	vol = np.stack(ll, axis = 0)
	# cv2.destroyAllWindows()
	return vol

def non_max_suppression(img, sigma):
	# normalization
	img = img/255.0
	k = 2**(1/4)
	thresh = 0.03
	X, Y = img.shape
	coords = list()
	vol = generate_volume(img, sigma, k)
	for i in range(1, X):
		for j in range(1, Y):
			# for z in range(1, vol.shape[0] - 1):
			# 	sl = vol[z-1: z+2, i - 1 : i + 2, j-1 : j+2]
			# 	result = np.amax(sl)
			# 	if result >= thresh:
			# 		z1, x, y = np.unravel_index(sl.argmax(), sl.shape)
			# 		if z1 == 1 and x == 1 and y == 1:
			# 			coords.append((i + x - 1, j + y - 1, (k**(z + z1 - 1))*sigma))
			sliced = vol[:, i - 1 : i + 2, j-1 : j+2]
			result = np.amax(sliced)
			if result >= thresh:
				z, x, y = np.unravel_index(sliced.argmax(), sliced.shape)
				coords.append((i + x - 1, j + y - 1, (k**z)*sigma))

	x = list(set(coords))
	return overlap_removal(x, 0.3)
	# return x


def intersect(r, R, d):
	if R < r:
		r, R = R, r

	# print(r, R, d, sep='::')
	part1 = r*r*math.acos((d*d + r*r - R*R)/(2*d*r));
	part2 = R*R*math.acos((d*d + R*R - r*r)/(2*d*R));
	part3 = 0.5*math.sqrt((-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R));

	return part1 + part2 - part3;

def overlap_removal(points, thresh):
	removal_indices = list()
	for i in range(len(points)):
		for j in range(i + 1, len(points)):
			p1, p2 = points[i], points[j]
			dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**(1/2)
			if (1.414*(p1[2] + p2[2]) <= dist):
				pass
			elif dist <= 1.414*abs(p1[-1] - p2[-1]):
				# complete overlap
				if (p1[-1] > p2[-1]):
					removal_indices.append(j)
				else:
					removal_indices.append(i)
			else :
				# partial intersection
				area = intersect(p1[-1]*1.414, p2[-1]*1.414, dist)
				if (area > thresh):
					if (p1[-1] > p2[-1]):
						removal_indices.append(j)
					else:
						removal_indices.append(i)

	ll = [i for j, i in enumerate(points) if j not in removal_indices]

	return list(set(ll))

def blob(img):
	res = non_max_suppression(img, 0.1)
	return res

# def similarity(log1, log2):
def store(pth = "./LoG/"):
	detector = cv2.SimpleBlobDetector_create()
	mypath = './images/'
	names = [f for f in listdir(mypath) if isfile(join(mypath, f))]

	idx = 1
	for i in names:
		# res = blob(cv2.imread(mypath + i))
		res = detector.detect(cv2.imread(mypath + i, cv2.IMREAD_GRAYSCALE))
		lst = []
		for point in res:
			temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)	
			lst.append(temp)
		
		pickle.dump(lst, open(pth + i[:-4] + '.p', 'wb'))
		print(idx)
		idx += 1

def _pickle_keypoints(point):
	return cv2.KeyPoint, (*point.pt, point.size, point.angle, point.response, point.octave, point.class_id)

if __name__ == '__main__':
	img = cv2.imread('images/all_souls_000000.jpg')
	# img = cv2.resize(img, (400, 400), interpolation = cv2.INTER_AREA)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# store()

	res = blob(img)
	
	# plotting the blobs
	fig, ax = plt.subplots()

	nh, nw = img.shape
	ax.imshow(img)
	for blob in res:
	    y,x,r = blob
	    c = plt.Circle((x, y), r*1.414, color='red', linewidth=1.5, fill=False)
	    ax.add_patch(c)
	ax.plot()  
	plt.show()	