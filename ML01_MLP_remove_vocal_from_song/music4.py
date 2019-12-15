#%%
import librosa
import numpy as np
import soundfile as sf
from glob import glob
from sklearn.neural_network import MLPRegressor
import imageio
import matplotlib.pyplot as plt


#%%
files = glob("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/train/*.*")
mlp = MLPRegressor(hidden_layer_sizes=(20,20,), activation='logistic', 
	learning_rate='constant', learning_rate_init=0.001, 
	max_iter=10, batch_size=int(len(files)/2))


#%%
def fitSongs(novoice, voice):
	print("loading song "+voice+"...")
	audio0, sample_rate0 = librosa.load(novoice) 
	spectrum0 = librosa.stft(audio0)
	print("Spectrum0 shape: " + str(spectrum0.shape))

	# librosa.display.specshow(librosa.amplitude_to_db(spectrum0, ref = np.max), y_axis = 'log', x_axis = 'time')
	# plt.title("Spectrum0")
	# plt.colorbar(format='%+2.0f dB')
	# plt.tight_layout()
	# plt.show()

	audio1, sample_rate1 = librosa.load(voice) 
	spectrum1 = librosa.stft(audio1)
	# librosa.display.specshow(librosa.amplitude_to_db(spectrum1, ref = np.max), y_axis = 'log', x_axis = 'time')
	# plt.title("Spectrum1")
	# plt.colorbar(format='%+2.0f dB')
	# plt.tight_layout()
	# plt.show()

	# use 'a.T.view(...).T' instead
	print("fitting model...")
	spec0f = spectrum0.T.view(np.float32).T
	spec1f = spectrum1.T.view(np.float32).T

	spec1f = spec1f.T
	spec0f = spec0f.T
	mlp.partial_fit(spec1f, spec0f)

#%%
# for something
files = glob("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/train/*.*")
print(str(len(files)/2)+" musics")
for i in range(len(files)):
	if (i % 2) != 0:
		print("Fitting..." + str(i) + " pairs")
		fitSongs("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/train/"+str(int((i+1)/2))+"_0.wav", 
		"D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/train/"+str(int((i+1)/2))+"_1.wav")

print("loading song to be predicted...")
a3, s3 = librosa.load("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/test/prototype1.wav") 
spectrum3 = librosa.stft(a3)
spec3f = spectrum3.T.view(np.float32).T

print("predicting song...")
# size 747 is different from 634
spec3f = spec3f.T
spec_resf = mlp.predict(spec3f)
spec_resf = spec_resf.T

print(spec_resf.shape)

#spec_res = spec_resf.T.view(np.complex64).T
print("preparing output song...")
spec_resf = spec_resf.reshape(int(len(spec_resf)/2), int(len(spec_resf[0])), 2)
print(spec_resf.shape)
spec_resf = np.apply_along_axis(lambda args: [complex(*args)], 2, spec_resf)

print(spec_resf.shape)
new_res = []
for i in range(len(spec_resf)):
    tmp = []
    for j in range(len(spec_resf[i])):
        tmp.append(spec_resf[i][j][0])
    new_res.append(tmp)
new_res = np.array(new_res)
print(new_res.shape)


#%%
print("writing output song...")
imageio.imwrite('D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/img/new_res.jpg', new_res)
#reconstructed_audio = librosa.istft(new_res)
#sf.write("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/test/prototype_result.wav", reconstructed_audio, s3)	


# %%
