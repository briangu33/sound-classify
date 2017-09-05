import spectra
import keras

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




