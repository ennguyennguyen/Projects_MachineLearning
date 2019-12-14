#%% import libraries
import librosa
import librosa.display as disp
import numpy as np
import matplotlib.pyplot as plt
import scipy
import imageio
from glob import glob
from sklearn.neural_network import MLPClassifier
import soundfile as sf

#%% convert audio -> image
def audi2img(file):
	# read audio
	wave, rate = librosa.load(file)
	# convert to spectogram
	s = librosa.feature.melspectrogram(y=wave, sr=rate)
	fig = plt.figure(figsize=(4, 4)) # figsize=(2, 4)
	disp.waveplot(wave, sr=rate)
	disp.specshow(librosa.power_to_db(s, ref=np.max))
	img = file.split("/")[-1].split(".")[0]+".jpg"
	plt.savefig("img/"+img) # , dpi=400, bbox_inches='tight',pad_inches=0
	plt.close()    
	fig.clf()
	plt.close(fig)
	plt.close('all')
	del file,fig,wave,rate,s,img

#%% run some sample cases
train = glob("train/*.*")

for f in range(2):
	audi2img("1_0")
	audi2img("1_1")


#%% load both files into arrays (x and y)
imgs = glob("img/*.*")
arrs = []
for img in imgs:
	dim = imageio.imread(img)
	print(dim.shape, img)
	arrs.append(dim)

dt = arrs[1]

print(len(dt[:,1,:]))

print(np.concatenate(dt[:,1,:]).shape)


#%% build  mlp model
mlp = MLPClassifier(hidden_layer_sizes=(50,50,), activation='relu') # learning_rate=0.2, 

# train for each pixel

for col in range(len(arrs[1][1])):
	print(str(len(arrs[1][1]))+" - "+str(col))
	y = np.concatenate(arrs[0][:,col,:])
	x = np.concatenate(arrs[1][:,col,:])
	mlp.fit(x.reshape(-1,1), y.reshape(-1,1)) # x = music with voice, y = music without voice

# predict
audi2img("test/prototype.wav") # random music from train data
xx = imageio.imread("img/prototype.jpg")
new_img = [[[]]]
for x in range(len(xx[1])):
	print(str(len(xx[1]))+" - "+str(x))
	inp = np.concatenate(xx[:,x,:])
	new_data = mlp.predict(inp.reshape(-1,1))
	new_data.reshape(int(len(new_data)/3),3)
	for y in range(len(new_data)):
		new_img[y][x] = new_data[y]

print(np.array(new_img).shape)

#%% convert output to spectogram
imageio.imwrite('img/result.jpg', new_img)

# convert spectogram to a song
#res = librosa.feature.inverse.mel_to_audio(specto)

