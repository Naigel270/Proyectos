import numpy as np

def sigmoid(x):
    #Activation function
    return 1/(1+np.exp(-x))

class Neurona:
    def __init__(self,pesos,bias):
        self.pesos = pesos
        self.bias= bias
    
    def feedforward(self,inputs):
        #Entradas correspondientes a pesos, bias y funcion de activacion.
        fd=np.dot(self.pesos,inputs)+self.bias
        return sigmoid(fd)
    

#Probamos nuestra Neurona.

pesos= np.array([0,1]) #w1 = 0, #w2 =2
bias = 4               # b= 4
n=Neurona(pesos,bias)

x=np.array([2,3])     #x1=2, x2=3"
print(n.feedforward(x))

