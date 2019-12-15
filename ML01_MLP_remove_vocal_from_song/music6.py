#%% aaaaaaaaaaaaaaaaaaaaa
import numpy as np
import soundfile as sf
from sklearn.neural_network import MLPRegressor
import librosa
import librosa.display
import imageio
import matplotlib.pyplot as plt


#%% bbbbbbbbbbbbbbbbbbbbb
audio0, sample_rate0 = librosa.load("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/train/1_0.wav") 
spectrum0 = librosa.stft(audio0)

audio1, sample_rate1 = librosa.load("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/train/1_1.wav") 
spectrum1 = librosa.stft(audio1)


#%% cccccccccccccccccccccccc
mlp = MLPRegressor(hidden_layer_sizes=(50,50,), activation='logistic', 
	learning_rate='constant', learning_rate_init=0.25, 
	max_iter=500)

print(spectrum0.shape)
print(spectrum0[0][0])

# use 'a.T.view(...).T' instead
spec0f = spectrum0.T.view(np.float64).T
spec1f = spectrum1.T.view(np.float64).T



print(spec0f.shape)
print(spec0f[0][0])

spec1f = spec1f.T
spec0f = spec0f.T
print("training shape: "+str(spec0f.shape))
mlp.fit(spec1f, spec0f)



#%% ddddddddddddddddddddddddd
audio3, sample_rate3 = librosa.load("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/test/prototype1.wav") 
spectrum3 = librosa.stft(audio3)
print(spectrum3)
spec3f = spectrum1.T.view(np.float64).T # spectrum3.T.view(np.float64).T
print(spec3f.shape)

# size 747 is different from 634
spec3f = spec3f.T
print(spec3f.shape)
print("prediction shape: "+str(spec0f.shape))
spec_resf = mlp.predict(spec3f)
print(spec_resf.shape)
spec_resf = spec_resf.T
print(spec_resf.shape)
print(spec_resf[0][0])

# ValueError: cannot reshape array of size 765675 into shape (1025,747,2)
#spec_resf = spec_resf.reshape(int(len(spec_resf)/2), int(len(spec_resf[0])), 2)
#spec_resf = np.apply_along_axis(lambda args: [complex(*args)], 2, spec_resf)


#spec_resf = spec_resf.T.view(np.complex64).T
# print(spec_resf.shape)
# print(spec_resf)



#%% eeeeeeeeeeeeeeeeeeeee
reconstructed_audio = librosa.istft(spec_resf, dtype=spec3f[0][0].dtype)

librosa.display.specshow(librosa.amplitude_to_db(spectrum0, ref = np.max), y_axis = 'log', x_axis = 'time')
plt.title("For Reference")
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

librosa.display.specshow(librosa.amplitude_to_db(spec_resf, ref = np.max), y_axis = 'log', x_axis = 'time')
plt.title("Output")
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

sf.write("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/test/prototype_result.wav", reconstructed_audio, sample_rate3)	


# %%
audioTest, sample_rateTest = librosa.load("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/test/prototype1.wav") 
spectrum3 = librosa.stft(audioTest)
print(spectrum3)
print("\n")

spec3fT = spectrum3.T
print(spec3fT)


#%%
spec3fReT = spec3fT.view(np.float64).T
print(spec3fReT)


# %%
