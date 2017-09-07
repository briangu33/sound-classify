import parse_samples as ps
import spectra

import random
from random import shuffle, randint

import keras
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D
from keras.models import Model
from keras import optimizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize

def convert_sample_to_conv_input(sample):
	mat = spectra.get_frequency_matrix(sample)
	input_mat = []
	for i in range(mat.shape[1]):
		tup = spectra.calculate_spectra_on_frame(i)
		tmp = []
		for j in range(12):
			tmp.append(tup[3][j] if j < len(tup[3]) else 0.0)
		input_mat.append(tmp)
	return input_mat


def read_freq_csv(sample):
	mat = pd.read_csv('./res/' + sample.instr + '_spectra/' + sample.instr + '_' + sample.note + str(sample.octave) + '_' + sample.duration + '_' + sample.vol + '_' + sample.condition + '.csv')
	mat.drop('Unnamed: 0', inplace=True, axis=1)
	mat = mat.as_matrix()
	mat = np.vectorize(lambda input: complex(input) if isinstance(input, str) else input)(mat)
	return mat

def test_normallized_input(sample, has_csvs=False):
	mat = read_freq_csv(sample) if has_csvs else spectra.get_frequency_matrix(sample)
	mat = np.abs(mat)
	mat = normalize(mat, norm='l1', axis=0)
	return mat

def process_input(sample_list):
	tmp_list = []
	for i in range(len(sample_list)):
		if i % 100 == 99:
			print(i + 1, '/', len(sample_list))
		x = test_normallized_input(sample_list[i]).T
		tmp_list.append(x.reshape(x.shape + (1,)))
	return np.array(tmp_list)

def run_model():
	sample_list = []
	sample_list += ps.get_violin_samples()
	sample_list += ps.get_flute_samples()

	random.shuffle(sample_list)
	ys = np.array([[1, 0] if sample.instr == 'flute' else [0, 1] for sample in sample_list])
	print(ys)

	print(len(sample_list))
	print("Processing input:")
	sample_list = process_input(sample_list)
	print("Done")
	cut = len(sample_list) * 4 // 5
	train_samples = sample_list[:cut]
	train_ys = ys[:cut]

	test_samples = sample_list[cut:]
	test_ys = ys[cut:]

	inputs = Input((None, 1025, 1))
	x = Conv2D(64, (3, 1025), activation='tanh')(inputs)
	x = GlobalAveragePooling2D()(x)
	outputs = Dense(2, activation='softmax')(x)
	model = Model(inputs=inputs, outputs=outputs)
	sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0)
	model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
	# model.fit(train_samples, train_ys)
	accuracies = []
	accuracies_2 = []
	num_epochs = 40
	for j in range(num_epochs):
		count_train = 0
		for i in range(len(train_samples)):
			rand_index = randint(0, len(train_samples) - 1)
			if i % 500 == 499:
				print(i + 1)
			ans = model.predict_on_batch(np.array([train_samples[i]]))
			ans = (0, ans[0][0]) if ans[0][0] > ans[0][1] else (1, ans[0][1])
			# print(ans, test_ys[i])
			if ans[0] == train_ys[i][1]:
				count_train += 1.0
			(model.train_on_batch(np.array([train_samples[rand_index]]), np.array([train_ys[rand_index]])))
		print("Training accuracy: %0.0f / %d -- %f"  % (count_train, len(train_samples), count_train / len(train_samples)))
		accuracies.append(count_train / len(train_samples))

		count = 0
		for i in range(len(test_samples)):
			ans = model.predict_on_batch(np.array([test_samples[i]]))
			ans = (0, ans[0][0]) if ans[0][0] > ans[0][1] else (1, ans[0][1])
			# print(ans, test_ys[i])
			if ans[0] == test_ys[i][1]:
				count += 1.0
		print(count, len(test_samples))
		print("Validation Accuracy:",  count / len(test_samples))
		accuracies_2.append(count / len(test_samples))
	# print(sample_list)
	# for layer in model.layers:
	# 	print(layer.get_weights())
	print(accuracies)
	plt.close()
	plt.ylim([0, 1])
	plt.plot(range(num_epochs), accuracies, 'r')
	plt.plot(range(num_epochs), accuracies_2, 'b')
	plt.show()

run_model()
# tmp = process_input(ps.get_violin_samples())
# print(tmp[0].shape)