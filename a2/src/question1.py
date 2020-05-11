# Question 1

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import math

train_path = './training/'
validation_path = './validation'
noise_path = './_background_noise'

def dft(inp):
	n = len(inp)
	end = list()

	for i in range(n):
		semi = 0.0
		for k in range(n):
			semi += inp[k]*(math.cos(-2 * math.pi * i * k / n)) 
		end.append(semi)
	return np.array(end)

def plot_spectrogram(data):
	data = 20*np.log10(data + 0.000000001);
	plt.pcolormesh(data.T);
	plt.show()

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

fileName = train_path + 'zero/004ae714_nohash_0.wav'

sampling_frequency, samples = wavfile.read(fileName)
samples = samples / 32768.
duration = (len(samples) / sampling_frequency)
print(duration)

plot_spectrogram(spectrogram(samples, 100, duration, 0.5))