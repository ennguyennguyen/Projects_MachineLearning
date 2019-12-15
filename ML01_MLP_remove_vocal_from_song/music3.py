#%% import libraries
import librosa
import numpy as np
import soundfile as sf
from sklearn.neural_network import MLPRegressor

#%% Initiate Training Data
audio0, sample_rate0 = librosa.load("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/train/1_0.wav") 
spectrum0 = librosa.stft(audio0)
audio1, sample_rate1 = librosa.load("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/train/1_1.wav") 
spectrum1 = librosa.stft(audio1)

#Convert into Spectrogram array, use 'a.T.view(...).T' instead
spec0f = spectrum0.T.view(np.float32).T
spec1f = spectrum1.T.view(np.float32).T

spec0f = spec0f.T
spec1f = spec1f.T

#%% Initiate MLP Regression
mlp = MLPRegressor(hidden_layer_sizes=(50,50,), 
                    activation='logistic', 
                    learning_rate_init = 0.003,
                    max_iter = 3000,
                    batch_size = 7)

mlp.fit(spec1f, spec0f)

#%% Read Test Data & Run Prediction
a3, s3 = librosa.load("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/test/prototype1.wav") 
spectrumTest = librosa.stft(a3)
specTestf = spectrumTest.T.view(np.float32).T

# size 747 is different from 634
specTestf = specTestf.T
spec_resf = mlp.predict(specTestf)
spec_resf = spec_resf.T

print(spec_resf.shape)

#spec_res = spec_resf.T.view(np.complex64).T
print(int(len(spec_resf[0])))
print(int(len(spec_resf))/2)
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

#%% Export Output
reconstructed_audio = librosa.istft(new_res)
sf.write("D:/00_MASTER OF COMPUTER SCIENCE_MUM/00_Projects/02_Project_MachineLearning/ML01_MLP_remove_vocal_from_song/test/prototype_result.wav", reconstructed_audio, s3)	
