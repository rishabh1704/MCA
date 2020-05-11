# Question 2

import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import math
import librosa


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
	


fileName = './training/' + 'eight/004ae714_nohash_0.wav'

sampling_frequency, samples = wavfile.read(fileName)
samples = samples / 32768.

a = mfcc(samples, 512, 256, sampling_frequency, 13)
plot_spectrogram(a)