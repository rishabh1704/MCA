import cv2
import numpy as np
from PIL import Image  
import PIL  
import pickle
from os import listdir
from os.path import isfile, join
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support
import os
from time import time
import pandas as pd

def dist(cac1, cac2):
	if (cac1.shape != cac2.shape):
		print("Dimensions doen't match in the correlograms.")
		return
	size, dim = cac1.shape

	score = 0.0
	for i in range(size):
		for j in range(dim):
			score += abs(cac1[i, j] - cac2[i, j])/(1 + cac1[i, j] + cac2[i, j])

	return score


def valid(X, Y, point):
    if point[0] < 0 or point[0] >= X or point[1] < 0 or point[1] >= Y:
    	return False
    
    return True

def get_nebrs(image, x, y, dist):
	X, Y = image.shape
	points = [[x+dist, y+dist], [x+dist, y], [x+dist, y-dist], [x, y-dist], [x-dist, y-dist], [x-dist, y], [x-dist, y+dist],[x, y+dist]]

	valid_list = list()
	for pt in points:
		if (valid(X, Y, pt)):
			valid_list.append(pt)

	return valid_list

def create_map(img, q):
	quantized_dict = dict()

	for quant in range(q):
		ll = list()
		res = np.where(img == quant)
		res_x = res[0]
		res_y = res[1]
		a = len(res_x)
		for i in range(a):
			ll.append((res_x[i],res_y[i]))

		quantized_dict[quant] = ll

	return quantized_dict

def autocc(img):
	distances = [1, 3]
	quantization_level = 60
	# img = cv2.resize(img, (256 , 256), interpolation = cv2.INTER_CUBIC)
	# Quantize images into m colors.
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	im_pil = Image.fromarray(img)
	q_img = im_pil.quantize(quantization_level)
	im_np = np.asarray(q_img)

	cac = np.zeros((quantization_level, len(distances)))
	cmap = create_map(im_np, quantization_level)

	for idx, val in enumerate(distances):
		for q in range(quantization_level):
			legit = 0
			total = 0
			for j in cmap[q]:
				pts = get_nebrs(im_np, j[0], j[1], val)
				for p in pts:
					if (q == im_np[p[0], p[1]]):
						legit += 1
					total += 1

			cac[q, idx] = float(legit)/float(total)


	return cac

def store(pth = "./CC/"):
	mypath = './images/'
	names = [f for f in listdir(mypath) if isfile(join(mypath, f))]

	idx = 1
	for i in names:
		img = cv2.imread(mypath + i)
		img = cv2.resize(img, (250 , 250), interpolation = cv2.INTER_CUBIC)
		res = autocc(cv2.imread(mypath + i))
		pickle.dump(res, open(pth + i[:-4] + '.p', 'wb'))
		print(idx)
		idx += 1

def load():
	res = pickle.load(open('CC/all_souls_000047.p', 'rb'))
	print(res)

def prep_x(x, f):
    x = f.readline()
    # print(x)
    x  = x.split()[0]
    x = x[5:]
    return x

def get_name(j):
	return j.split('.')[0]

def prep_res(results):
    results = results.sort_values('score')
    return results

def printer(truth, i, results, good, ok, junk):
    for a in truth:
        comp = list()
        jk = i.split('query')
        path = jk[0]
        ss = path + a
        file_f = open('./train/ground_truth/'+ ss +'.txt')
        for q in file_f :
            comp.append(q.split('\n')[0])
        num = np.shape(comp)
        check = list(results['name'][0:num[0]])
        for qq in comp :
            if qq in check and a == 'good':
                good = good + 1
            elif qq in check and a == 'ok':
                ok = ok + 1
            elif qq in check and a == 'junk':
                junk = junk + 1

        print(a)
        print('ground_truth')
        print(comp)
        print('Retrieved')
        print(list(results['name'][:np.shape(comp)[0]]))
        print('Scores of retrieved')
        print(results[:np.shape(comp)[0]])

        p, r, f1, s = precision_recall_fscore_support(comp, list(results['name'][0:np.shape(comp)[0]]),average='macro')
        print('precision '+ str(p))
        print('recall '+ str(r))
        print('f1 '+ str(f1))
        return (good, ok, junk)

def matcher():
	query_path = "./train/query"
	no_q, good, ok, junk = (0, 0, 0, 0)

	for i in os.listdir(query_path):
		t1 = time()
		no_q += 1
		names = list()
		score = list()
		comp = list()
		f = open(query_path+'/'+i)
		x = ''
		x = prep_x(x, f)
		y = pickle.load(open('./cc/'+x +'.p', 'rb'))

		for j in os.listdir('./cc'):
		    if (get_name(j) != x) :
		        z = pickle.load(open('./cc/'+j , 'rb'))
		        names.append(get_name(j))
		        score.append(dist(y,z))
		results = pd.DataFrame({'name': names ,'score':score})
		results = prep_res(results)
		print('time taken = ')
		print(time()-t1)
		truth = ['good','ok', 'junk']
		good, ok, junk = printer(truth, i, results, good, ok, junk)

	print('*** Average stats *** ')
	print('good retrieved = '+ str(good/no_q))
	print('ok retrieved = '+ str(ok/no_q))
	print('junk retrieved = '+ str(junk/no_q))


if __name__ == '__main__':
	# store()
	matcher()
	# # load()