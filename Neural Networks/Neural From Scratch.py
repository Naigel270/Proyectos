import numpy as np

def sigmoid(x):
    #Activation function
    return 1/(1+np.exp(-x))

def deriv_sigmoid(x):
    #derivada de la sigmoide
    fx= sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    #y_true and y_pred son las etiquetas reales y las predicciones respectivamente.
    return ((y_true -y_pred) ** 2).mean()

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



class NeuralNetwork:

    
    '''
    Red con:
    -2 entradas
    -Una capa con 2 neuronas (h1,h2)
    -Una capa de salida con una neurona (o1)
   Cada neurona con los mismos pesos y bias:
    -w =[0,1]
    -b = 0
    '''
    def __init__(self):
       
        #pesos

        self.w1=np.random.normal()
        self.w2=np.random.normal()
        self.w3=np.random.normal()
        self.w4=np.random.normal()
        self.w5=np.random.normal()
        self.w6=np.random.normal()

        #Bias Iniciales
        
        self.b1= np.random.normal()
        self.b2= np.random.normal()
        self.b3= np.random.normal()
        print(self.w1)


    def feedforward(self,x):
       
        h1= sigmoid(self.w1*x[0]+self.w2*x[1]+self.b1)
        h2= sigmoid(self.w3*x[0]+self.w4*x[1]+self.b2)

        #Entradas de o1 son las salidas de h1 y h2
        out_o1=sigmoid(self.w5*h1+self.w6*h2+self.b1)
        return out_o1
    
    def train(self, data, all_y_trues):
        '''
        -data es un numpy Array de dimensión(n x 2), n= # de datos en el dataset.
        -all_y_trues es un numpy Array con n elementos
        Los elementos en all_y_trues corresponde a las etiquetas de resultados de data
        '''
        learn_rate=0.1 
        epochs=1000 # numero de iteraciones a realizar en el dataset completo

        for epoch in range(epochs):
            for x, y_true in zip(data,all_y_trues):
             #feedforward inicial

             sum_h1=self.w1*x[0]+self.w2*x[1]+self.b1
             h1= sigmoid(sum_h1)

             sum_h2=self.w3*x[0]+self.w3*x[1]+self.b2
             h2= sigmoid(sum_h2)

             sum_o1=self.w5*h1+self.w6*h2+self.b3
             o1= sigmoid(sum_o1)

             y_pred = o1
             
            # Calculamos las derivadas parciales
            # Nombramos d_L_d_w1 representado la nomenclatura "Derivada de L /Derivada de w1"

            d_L_d_ypred = -2*(y_true -y_pred)

            #Neurona o1

            d_ypred_d_w5= h1*deriv_sigmoid(sum_o1)
            d_ypred_d_w6= h2 *deriv_sigmoid(sum_o1)
            d_ypred_d_b3=deriv_sigmoid(sum_o1)

            d_ypred_d_h1=self.w5*deriv_sigmoid(sum_o1)
            d_ypred_d_h2=self.w6*deriv_sigmoid(sum_o1)


            #neurona h1
            d_h1_d_w1=x[0] * deriv_sigmoid(sum_h1)
            d_h1_d_w2=x[1] * deriv_sigmoid(sum_h1)
            d_h1_d_b1=deriv_sigmoid(sum_h1)

            #neurona h2
            d_h2_d_w3 =x[0] * deriv_sigmoid(sum_h2)
            d_h2_d_w4 =x[1] * deriv_sigmoid(sum_h2)
            d_h2_d_b2=deriv_sigmoid(sum_h2)


            #Actualizamos con los nuevos valores de peso y bias

            #Neurona h1
            self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
            self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
            self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

            # Neuron h2
            self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
            self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
            self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

            # Neuron o1
            self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
            self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
            self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

        # Calculamos la perdida total al final de cada iteracion

        if epoch % 10 == 0 :
           y_preds =np.apply_along_axis(self.feedforward,1,data)
           loss=mse_loss(all_y_trues,y_preds)
           print("Perdida : %.3f" % (epoch,loss))


    #definimos nuestros datos para predicciones

datos = np.array([
     [-2,1],
     [25,6],
     [17,4],
     [-15,-6],   
])
y = np.array([
        1,
        0,
        0,
        1,
])

    #Entrenamos nuestra red
    
red= NeuralNetwork()
red.train(datos,y)


