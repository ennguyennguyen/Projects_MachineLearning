#%% aaaaaaaaaaa
import librosa
import numpy as np
import soundfile as sf
from sklearn.neural_network import MLPRegressor
import librosa.display
import imageio
import matplotlib.pyplot as plt

#%% bbbbbbbbbbbbbb
audio0, sample_rate0 = librosa.load("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/train/1_0.wav") 
spectrum0 = librosa.stft(audio0)

audio1, sample_rate1 = librosa.load("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/train/1_1.wav") 
spectrum1 = librosa.stft(audio1)

mlp = MLPRegressor(hidden_layer_sizes=(20,20,), activation='logistic', 
	learning_rate='constant', learning_rate_init=0.003, 
	max_iter=3000)

print(spectrum0.shape)
print(spectrum0[0][0])

# use 'a.T.view(...).T' instead
spec0f = spectrum0.T.view(np.float32).T
spec1f = spectrum1.T.view(np.float32).T

print(spec0f.shape)
print(spec0f[0][0])

spec1f = spec1f.T
spec0f = spec0f.T
print("training shape: "+str(spec0f.shape))
print(type(spec0f[0][0]))
mlp.fit(spec1f, spec0f)

a3, s3 = librosa.load("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/test/prototype1.wav") 
spectrum3 = librosa.stft(a3)
spec3f = spectrum3.T.view(np.float32).T # spectrum3.T.view(np.float64).T
print(spec3f.shape)

# size 747 is different from 634
spec3f = spec3f.T
print(spec3f.shape)
print("prediction shape: "+str(spec0f.shape))
print(type(spec0f[0][0]))
spec_resf = mlp.predict(spec3f)

print(type(spec_resf[0][0]))
print(spec_resf.shape)
#spec_resf = spec_resf.T
print(spec_resf.shape)
print(spec_resf[0][0])

print(spectrum0.T.shape)
print(spec_resf.shape)
print(type(spec_resf[0][0]))
new_stuff = []
for j in range(len(spec_resf)):
    for i in range(len(spec_resf[j])):
        if(i%2!=0):
            new_stuff.append(np.complex(spec_resf[j][i-1], spec_resf[j][i]))
            
spec_resfn = np.array(new_stuff).reshape(spectrum3.T.shape)
print(spec_resfn.shape)
print(type(spec_resfn[0][0]))

#%% cccccccccccccc
fff = np.complex64(spec_resfn).T
print(type(fff[0][0]))
print(type(spectrum3[0][0]))
reconstructed_audio = librosa.istft(fff)
sf.write("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/test/prototype_result.wav", reconstructed_audio, s3)

librosa.display.specshow(librosa.amplitude_to_db(fff, ref = np.max), y_axis = 'log', x_axis = 'time')
plt.title("Output")
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# %%
