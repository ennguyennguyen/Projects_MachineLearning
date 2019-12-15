#%% aaaaaaaaaaaaa
import librosa
import numpy as np
import soundfile as sf
from glob import glob
from sklearn.neural_network import MLPRegressor
import librosa.display
import imageio
import matplotlib.pyplot as plt

#%% bbbbbbbbbbbbbbbbbbbb
mlp = MLPRegressor(hidden_layer_sizes=(35,35,), activation='logistic', 
		learning_rate='constant', learning_rate_init=0.001, 
		max_iter=10)

def fitSongs(novoice, voice):
	print("loading song "+voice.split("_")[0].split("/")[1]+"...")
	audio0, sample_rate0 = librosa.load(novoice) 
	spectrum0 = librosa.stft(audio0)
	audio1, sample_rate1 = librosa.load(voice) 
	spectrum1 = librosa.stft(audio1)

	# use 'a.T.view(...).T' instead
	spec0f = spectrum0.T.view(np.float32).T
	spec1f = spectrum1.T.view(np.float32).T

	spec1f = spec1f.T
	spec0f = spec0f.T
	print("training shape: "+str(spec0f.shape))
	mlp.fit(spec1f, spec0f)

files = glob("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/train/*.*")
print(str(len(files)/2)+" musics")
for i in range(len(files)):
	if (i % 2) != 0:
		x = "D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/train/"+str(int((i+1)/2))+"_0.wav"
		y = "D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/train/"+str(int((i+1)/2))+"_1.wav"
		fitSongs(x, y)

a3, s3 = librosa.load("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/test/prototype1.wav") 
spectrum3 = librosa.stft(a3)
spec3f = spectrum3.T.view(np.float32).T # spectrum3.T.view(np.float64).T

# size 747 is different from 634
spec3f = spec3f.T
print("prediction shape: "+str(spec3f.shape))
spec_resf = mlp.predict(spec3f)

new_stuff = []
for j in range(len(spec_resf)):
    for i in range(len(spec_resf[j])):
        if(i%2!=0):
            new_stuff.append(np.complex(spec_resf[j][i-1], spec_resf[j][i]))
            
spec_resfn = np.array(new_stuff).reshape(spectrum3.T.shape)
print(spec_resfn)
print(spec_resfn.shape)

#%% cccccccccccccc
fff = np.complex64(spec_resfn).T
reconstructed_audio = librosa.istft(fff)
sf.write("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/test/prototype_result.wav", reconstructed_audio, s3)

librosa.display.specshow(librosa.amplitude_to_db(fff, ref = np.max), y_axis = 'log', x_axis = 'time')
plt.title("Output")
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# %%
