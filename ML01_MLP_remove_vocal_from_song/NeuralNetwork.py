#%% import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

#%% convert wav to spectrogram
sample_rate, samples = wavfile.read('1_0.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

#%% Initial input and output

# x = (hours of sleeping, hours of studying)
# y = test score of student
x = np.array(([2,9], [1,5], [3,6]), dtype = float)
y = np.array(([92], [86], [89]), dtype = float)

#scale unit
x = x/np.max(x, axis = 0) #maximum of x array
y = y/100 # max test score is 100

#%% Neural Network class & methods
class NeuralNetwork(object):
    def __init__(self):
        #parameter
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        #weights
        self.w1 = np.random.randn(self.inputSize, self.hiddenSize) #(3x2) weights matrix from input -> hidden layer
        self.w2 = np.random.randn(self.hiddenSize, self.outputSize) #(3x1) weights matrix from hidden -> output layer

    def forwardPass(self, x):
        #forward propagate
        self.z1 = np.dot(x, self.w1) #dot product of x and first set of weight
        self.z2 = self.sigmoid(self.z1) #activation function
        self.z3 = np.dot(self.z2, self.w2) #dot product of hidden layer and second set of weight
        output = self.sigmoid(self.z3)
        return output

    def sigmoid(self, s, deriv = False):
        if(deriv == True):
            return s * (1 - s)
        return 1/(1 + np.exp(-s))
        
    def backwardPass(self, x, y, output):
        #backward propagate
        self.output_error = y - output #error is output
        self.output_delta = self.output_error * self.sigmoid(output, deriv = True)

        self.z2_error = self.output_delta.dot(self.w2.T) #z2 error: how much hidden layer weights contribute to output error
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv = True) #apply derivative of sigmoid to z2 error

        self.w1 += x.T.dot(self.z2_delta) #adjust input -> hidden weight
        self.w2 += self.z2.T.dot(self.output_delta) #adjust hidden -> output weights

    def train(self, x, y):
        output = self.forwardPass(x)
        self.backwardPass(x, y, output)

#%% Call NN class and iterate 1000 times -> print out result
NN = NeuralNetwork()

for i in range(1000): #train NN 1000 times
    NN.train(x, y)

print("Input: ")
print(str(x))
print("\n")
print("Actual Output: ")
print(str(y))
print("\n")
print("Loss: " + str(np.mean(np.square(y - NN.forwardPass(x)))))
print("\n")
print("Predicted output: ")
print(str(NN.forwardPass(x)))

# %%
