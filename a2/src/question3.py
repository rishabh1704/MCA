# Question 3

from os import listdir
import os.path
import numpy as np
from scipy.io import wavfile
import librosa
import random
from tqdm import tqdm
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.svm import SVC 
from sklearn.svm import LinearSVC
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.metrics import classification_report

train_path = './training/'
validation_path = './validation/'
noise_path = './_background_noise_/'
folders = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# getting the noise.
noise_vector = list()
for f in listdir(noise_path):
	fs, file_noise = wavfile.read(noise_path + f)
	noise_vector.append(file_noise)

def spectrogram(samples, window_length, duration, overlap, sampling_rate = None):

	if sampling_rate is not None:
		hopper = int(round(duration * sampling_rate))
		step_size = int(len(samples) / hopper)
		samples = samples[::step_size]

	stride = int(window_length * overlap)
	wd = np.hanning(window_length)
	out = np.empty(( (int(len(samples) - window_length) // stride + 1), int(window_length)), dtype=np.float32)  

	for i in range( (len(samples) - window_length) // stride + 1) :
		a = i*stride
		# winn = np.fft.rfft(samples[a:a+window_length] * wd) / window_length
		winn = dft(samples[a:a+window_length] * wd) / window_length
		# print(winn.shape)
		out[i,:] = np.abs(winn)**2

	return out

def hertz2mel(freq):
	return (2595*math.log10(1 + freq / 700.0))

def mel2hertz(mel):
	return 700*(10**(mel/2595.0) - 1)

def plot_spectrogram(data):
	data = todB(data)
	plt.pcolormesh(data.T);
	plt.show()

def todB(f):
	return 20*np.log10(f + 1e-10)

def mfcc(sample, window_length, step_size, sr, n_coeff):
	n = len(sample)
	num_windows = (int(len(samples) - window_length) // step_size + 1)
	frames = list()
	for i in range(num_windows):
		frames.append(sample[i*step_size:i*step_size + window_length])

	frames = np.array(frames)
	wd = np.hamming(window_length)
	frames = frames * wd

	NFFT = 512
	filts = 40
	# power spectrum
	periodogram = (float(NFFT)**(-1)*(np.abs(np.fft.rfft(frames, NFFT))**2))

	high = int(hertz2mel(sr))
	lst = list()
	for i in range(filts + 2):
		lst.append(high*(i/(filts + 1.0)))

	hzpts = list()
	for p in lst:
		hzpts.append((NFFT + 1)* mel2hertz(p)/sr)

	pts = np.array(hzpts)
	filter_bank = np.zeros((filts, int(math.floor(NFFT/2 + 1))))
	n_col = filter_bank.shape[1]

	for kk in range(filts):
		m = kk + 1
		f_mminus_one = (pts[m - 1])
		f_m = (pts[m])
		f_mplus_one = (pts[m + 1])

		# for k > f_m filter bank is 0
		# for k < f_mminus_one filter_bank is 0
		for k in range(n_col):
			if f_mminus_one <= k < f_m:
				filter_bank[kk, k] = (k - f_mminus_one)/(f_m - f_mminus_one)
			elif f_m < k <= f_mplus_one:
				filter_bank[kk, k] = (f_mplus_one - k)/(f_mplus_one - f_m)
			elif k == f_m:
				filter_bank[kk, k] = 1.0

	
	fltr_bks = np.dot(periodogram, filter_bank.T)
	fltr_bks = todB(fltr_bks)
	mfcc = dct(fltr_bks)[:,1:n_coeff]

	# the following step was inspired from https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
	mfcc = mfcc - (np.mean(mfcc, axis=0) + 1e-16)

	return mfcc

def add_noise(data, coeff = 1):
	it = random.randrange(0, len(noise_vector))
	vec = noise_vector[it]
	diff = len(vec) - len(data)

	if diff > 0:
		# truncate
		return (data + coeff*vec[:len(data)])
	elif diff < 0:
		# pad
		return (data + coeff*np.pad(vec, (0, diff), 'constant', constant_values = 0))
	else:
		return (data + coeff*vec)

# feature functions
def spectrogram_features(file, noise = False):
	n_fft = 2048
	hop_length = int(n_fft / 1.5)
	x, sr = librosa.load(file, sr=None)
	if (noise):
		x = add_noise(x, 0.000001)
	# X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
	# Xdb = librosa.amplitude_to_db(abs(X))
	Xdb = abs(spectrogram(x, n_fft, (len(x)/sr), 0.5))
	return Xdb.flatten()

def mfcc_features(file, noise = False):
	x, sr = librosa.load(file, sr=None)
	if (noise):
		x = add_noise(x, 0.000001)
	# mfccs = librosa.feature.mfcc(x, sr=sr)
	mfccs = mfcc(x, 512, 256, sr, 13)
	return mfccs.flatten()

# creating the dataset.
train_features_s = list()
train_features_m = list()
train_labels = list()

validation_features_s = list()
validation_features_m = list()
validation_labels = list()

storage = dict()

pickle_storage = './raw_data6.p'
if not os.path.exists(pickle_storage):

	print("feature calc started...")

	for idx , fol in tqdm(enumerate(folders)):
		path_train = train_path + fol + '/'
		path_validate = validation_path + fol + '/'

		pather = [path_train, path_validate]
		
		for i, pp in tqdm(enumerate(pather)):
			for f in listdir(pp):
				filename = pp + f
				if (i == 0):
					train_features_s.append(spectrogram_features(filename, True))
					train_features_m.append(mfcc_features(filename, True))
					train_labels.append(idx)
				else:
					validation_features_s.append(spectrogram_features(filename))
					validation_features_m.append(mfcc_features(filename))
					validation_labels.append(idx)				

	print("dataset calculated...")
	store = {'train_features_s':train_features_s,'train_features_m' : train_features_m,'train_labels' : train_labels,'validation_features_s':validation_features_s,'validation_features_m':validation_features_m,'validation_labels':validation_labels}

	storage = store
	pickle.dump(store, open(pickle_storage, 'wb'))
	print("pickled")
else:
	storage = pickle.load(open(pickle_storage, 'rb'))
	print('unpickled.')


ll = [storage['train_features_s'], storage['validation_features_s']]
ll2 = [storage['train_features_m'], storage['validation_features_m']]

for kk in ll:
	ix = list()
	for ss in kk:
		ix.append(ss.shape[0])

	len1 = max(ix)
	for idx, _ in enumerate(kk):
		array = kk[idx]
		if array.shape[0] < len1:
			kk[idx] = np.pad(array, (0, (len1 - array.shape[0])), 'constant', constant_values = 0)

for kk in ll2:
	ix = list()
	for ss in kk:
		ix.append(ss.shape[0])

	len2 = max(ix)
	for idx, _ in enumerate(kk):
		array = kk[idx]
		if array.shape[0] < len2:
			kk[idx] = np.pad(array, (0, (len2 - array.shape[0])), 'constant', constant_values = 0)

print("Length equalisation done.")

y_train = np.array(storage['train_labels'])
y_test = np.array(storage['validation_labels'])


# data normalization
scaler = preprocessing.MinMaxScaler()
# scaler.fit(X_train_m).transform(X_train_m)


# spectrogram features
X_train_s = normalize(np.array(storage['train_features_s']))
X_test_s = normalize(np.array(storage['validation_features_s']))

# #######################################################

X_train_m = (np.array(storage['train_features_m']))
X_test_m = (np.array(storage['validation_features_m']))

scaler.fit(X_train_m)
X_train_m = scaler.transform(X_train_m)

scaler.fit(X_test_m)
X_test_m = scaler.transform(X_test_m)

# md = pickle.load(open('./model2.p', 'rb'))
# print(classification_report(y_test, md.predict(X_test_m)))

print('Model training started...')


model1 = LinearSVC(random_state=1110, tol=1e-5, max_iter = 1000, verbose = 1, C = 0.5)
model1.fit(X_train_s, y_train)


# model2 = LinearSVC(random_state=10, tol=1e-5, max_iter = 1000, C =0.0001, verbose = 1)
# model2.fit(X_train_m, y_train)

print('Trained...')
	

y_pred = model1.predict(X_test_s)
print(classification_report(y_test, y_pred))
# pickle.dump(model1, open('model1_n.p', 'wb'))


# y_pred = model2.predict(X_test_m)
# print(classification_report(y_test, y_pred))
# pickle.dump(model2, open('model2_n.p', 'wb'))
