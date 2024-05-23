# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:26:21 2024

@author: Usuario
"""

import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea
from time import sleep
from IPython.display import clear_output
import numpy as np
import pandas as pd
import random as rd
from PIL import Image, ImageDraw, ImageFont
import networkx as nx
from itertools import product, permutations
from nltk import Tree
from collections import namedtuple
import time

class Vendedor:
    '''
    
    '''
    """ Clase Vendedor representa las posibles acciones que puede realizar el vendedor y
        si el objetivo se cumplió o no
            Atributos: -estado_inicial: Localidad de partida
                       -rutas: Diccionario con el peso entre cada desplazamiento entre localidades.
                       -localidades
                       -G: Objeto de networkx para graficar la representacion inicial
    """
    def __init__(self, localidad):
        self.estado_inicial = [localidad]
        self.rutas =  {'Usaquen':{'Chapinero':7.5, 'Santa Fe':11.6, 'San Cristobal':23, 'Usme':30, 'Tunjuelito':22.2, 'Bosa':24.8, 'Kennedy':21.1, 'Fontibon':21.1, 'Engativa':10.9, 'Suba':7.1, 'Barrios Unidos':8.8},\
                    'Chapinero':{'Usaquen':7.5, 'Santa Fe':6.2, 'San Cristobal':13.9, 'Usme':26.2, 'Tunjuelito':18.1, 'Bosa':20.3, 'Kennedy':14.8, 'Fontibon':15.2, 'Engativa':10.5, 'Suba':9.8, 'Barrios Unidos':4.6},\
                    'Santa Fe':{'Chapinero':6.2, 'Usaquen':11.6, 'San Cristobal':7.7, 'Usme':18.3, 'Tunjuelito':12, 'Bosa':14.6, 'Kennedy':10.3, 'Fontibon':12.1, 'Engativa':18.4, 'Suba':15.6, 'Barrios Unidos':9.3},\
                    'San Cristobal':{'Chapinero':13.9, 'Usaquen':23, 'Santa Fe':7.7, 'Usme':10.2, 'Tunjuelito':9.4, 'Bosa':13.9, 'Kennedy':11.2, 'Fontibon':17.4, 'Engativa':23.6, 'Suba':22.8, 'Barrios Unidos':15.6},\
                    'Usme':{'Chapinero':26.2, 'Usaquen':30, 'Santa Fe':18.3, 'San Cristobal':10.2, 'Tunjuelito':10.3, 'Bosa':16.1, 'Kennedy':15, 'Fontibon':21.5, 'Engativa':28.7, 'Suba':27.8, 'Barrios Unidos':24.4},\
                   'Tunjuelito':{'Chapinero':18.1, 'Usaquen':22.2, 'Santa Fe':12, 'San Cristobal':9.4, 'Usme':10.3, 'Bosa':7.7, 'Kennedy':7.4, 'Fontibon':13.9, 'Engativa':21.1, 'Suba':20.2, 'Barrios Unidos':16},\
                    'Bosa':{'Chapinero':20.3, 'Usaquen':24.8, 'Santa Fe':14.6, 'San Cristobal':13.9, 'Usme':16.1, 'Tunjuelito':7.7, 'Kennedy':7, 'Fontibon':11.1, 'Engativa':20.1, 'Suba':19.1, 'Barrios Unidos':16.8},\
                    'Kennedy':{'Chapinero':14.8, 'Usaquen':21.1, 'Santa Fe':10.3, 'San Cristobal':11.2, 'Usme':15, 'Tunjuelito':7.4, 'Bosa':7, 'Fontibon':6.5, 'Engativa':13.8, 'Suba':12.8, 'Barrios Unidos':11.3},\
                    'Fontibon':{'Chapinero':15.2, 'Usaquen':21.1, 'Santa Fe':12.1, 'San Cristobal':17.4, 'Usme':21.5, 'Tunjuelito':13.9, 'Bosa':11.1, 'Kennedy':6.5, 'Engativa':11, 'Suba':14.5, 'Barrios Unidos':13.1},\
                    'Engativa':{'Chapinero':10.5, 'Usaquen':10.9, 'Santa Fe':18.4, 'San Cristobal':23.6, 'Usme':28.7, 'Tunjuelito':21.1, 'Bosa':20.1, 'Kennedy':13.8, 'Fontibon':11, 'Suba':7, 'Barrios Unidos':6.9},\
                    'Suba':{'Chapinero':9.8, 'Usaquen':7.1, 'Santa Fe':15.6, 'San Cristobal':22.8, 'Usme':27.8, 'Tunjuelito':20.2, 'Bosa':19.1, 'Kennedy':12.8, 'Fontibon':14.5, 'Engativa':7, 'Barrios Unidos':10.4},\
                    'Barrios Unidos':{'Chapinero':4.6, 'Usaquen':8.8, 'Santa Fe':9.3, 'San Cristobal':15.6, 'Usme':24.4, 'Tunjuelito':16, 'Bosa':16.8, 'Kennedy':11.3, 'Fontibon':13.1, 'Engativa':6.9, 'Suba':10.4}
                    }
        self.localidades = list(self.rutas.keys())
        self.G = None

    def pintar_estado(self, estado):
        """ Creacion y plot del grafo como vertices las localidades y pesos el valor de la distancia entre cada para de localidades """
        self.G = nx.Graph()
        n = len(estado)
        for i in range(n):
            x, y = self.coords[estado[i]]
            self.G.add_node(estado[i], pos = (x,y))
        for i in range(n-1):
            self.G.add_edge(estado[i], estado[i+1], weight = self.rutas[estado[i]][estado[i+1]])
        pos = nx.get_node_attributes(self.G, 'pos')
        pesos = nx.get_edge_attributes(self.G,'weight')
        plt.figure(figsize=(12,12))
        nx.draw_networkx(self.G, pos)
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels = pesos)
        plt.show()

    def pintar_camino(self, camino):
        # Input: lista con el camino de los estados
        # Output: plot con la forma de resolver las torres
        for estado in camino:
            clear_output(wait=True)
            self.pintar_estado(estado)
            plt.show()
            sleep(.5)

    def acciones_aplicables(self, estado):
        """ Retorna las posibles acciones dado el estado
            Input: estado (localidad)
            Output: Lista con las localidades no visitadas
        """
        return [x for x in self.localidades if x not in estado]

    def transicion(self, estado, accion):
        """ Retorna la lista actualizada con la accion realizada
            Input: estado (lista con el camino en el momento)
                   accion (desplazamiento)
            Output: Copia de la lista estado actualizada """
        lista = copy.deepcopy(estado)
        lista.append(accion)
        return lista

    def test_objetivo(self, estado):
        """ Verifica si ya fueron visitadas todas las localidades
            Input: estado (lista con el camino)
            Output: Boolean """
        return set(self.localidades) == set(estado)

    def codigo(self, estado):
        """ Actualiza el codigo
            Input: estado (lista con el camino)
            Output: cadena """
        cad = ""
        for i in estado:
            cad = cad + " - " + i
        return cad

    def costo(self, estado1, estado2):
        """ Peso entre el estado y la accion
            Input: estado (camino actual)
                   accion (desplazamiento)
            Output: Int """
        loc = self.rutas[estado1[-1]]
        return loc[estado2]

    def obtener_peso(self, estado1, estado2):
        """ Retorna el peso entre dos localidades
            Input: estado1, estado2, localidades
            Output: Int, distancia entre ambas localidades"""
        loc = self.rutas[estado1[-1]]
       # loc = self.rutas[estado1]
        return loc[estado2[-1]]

    
    def heuristica(self, estado):
        
        if len(estado)>1:
          estado_A=estado[-1]
          estado_B=estado[-2]
          num1=self.obtener_peso([estado_A], [estado_B])
        else:
            num1=0
        return num1
    
    def heuristica2(self, estado):
        
        dist= self.coords
        x,y=dist[estado[-1]]
        x2,y2=dist[estado[-2]]
        num1=np.sqrt((x2-x)**2+(y2-y)**2)
        
        return num1
    
    
# Maximum size of frontera
max_size = 1000
# Maximum recursion length
max_size_nodo = 100


Tupla = namedtuple('Tupla', ['elemento', 'valor'])
class ListaPrioritaria():
    
    def __init__(self):
        self.pila = []
        
    def __len__(self):
        return len(self.pila)

    def push(self, elemento, valor):
        tupla = Tupla(elemento, valor)
        self.pila.append(tupla)
        self.pila.sort(key=lambda x: x[1])
            
    def pop(self, with_value=False):
        if with_value:
            return self.pila.pop(0)
        else:
            return self.pila.pop(0)[0]
        
    def elementos(self):
        return [x[0] for x in self.pila]
    
    def is_empty(self):
        return len(self.pila) == 0
    
    def __len__(self):
        return len(self.pila)

    def __str__(self):
        cadena = '['
        inicial = True
        for elemento, valor in self.pila:
            if inicial:
                cadena += '(' + str(elemento) + ',' + str(valor) + ')'
                inicial = False
            else:
                cadena += ', (' + str(elemento) + ',' + str(valor) + ')'
        return cadena + ']'
    
class Nodo:

    # Clase para crear los nodos

    def __init__(self, estado, madre, accion, costo_camino, codigo):
        self.estado = estado
        self.madre = madre
        self.accion = accion
        self.costo_camino = costo_camino
        self.codigo = codigo

def nodo_hijo(problema, madre, accion):

    # Función para crear un nuevo nodo
    # Input: problema, que es un objeto de clase ocho_reinas
    #        madre, que es un nodo,
    #        accion, que es una acción que da lugar al estado del nuevo nodo
    # Output: nodo

    estado = problema.transicion(madre.estado, accion)
    costo_camino = madre.costo_camino + problema.costo(madre.estado, accion)
    codigo = problema.codigo(estado)
    return Nodo(estado, madre, accion, costo_camino, codigo)

def solucion(n):
    if n.madre == None:
        return []
    else:
        return solucion(n.madre) + [n.accion]

def depth(nodo):
    if nodo.madre == None:
        return 0
    else:
        return depth(nodo.madre) + 1

def camino_codigos(nodo):
    if nodo.madre == None:
        return [nodo.codigo]
    else:
        return camino_codigos(nodo.madre) + [nodo.codigo]

def is_cycle(nodo):
    codigos = camino_codigos(nodo)
    return len(set(codigos)) != len(codigos)

def anchura(problema):
    
    '''Método de búsqueda primero en anchura'''
    
    nodo = Nodo(problema.estado_inicial, None, None, 0, problema.codigo(problema.estado_inicial))
    if problema.test_objetivo(nodo.estado):
            return nodo
    frontera = [nodo]
    while len(frontera) > 0:
        nodo = frontera.pop(0)
        acciones = problema.acciones_aplicables(nodo.estado)
        for a in acciones:
            hijo = nodo_hijo(problema, nodo, a)
            if problema.test_objetivo(hijo.estado):
                return hijo
            if not is_cycle(hijo):
                frontera.append(hijo)
                assert(len(frontera) < max_size)
    return None

def profundidad(problema):

    '''Método de búsqueda primero en profundidad'''

    nodo = Nodo(problema.estado_inicial, None, None, 0, problema.codigo(problema.estado_inicial))
    if problema.test_objetivo(nodo.estado):
            return nodo
    frontera = [nodo]
    while len(frontera) > 0:
        nodo = frontera.pop()
        acciones = problema.acciones_aplicables(nodo.estado)
        for a in acciones:
            hijo = nodo_hijo(problema, nodo, a)
            if problema.test_objetivo(hijo.estado):
                return hijo
            if not is_cycle(hijo):
                frontera.append(hijo)
                assert(len(frontera) < max_size)
    return None

def backtracking(problema, nodo):

    '''Método de búsqueda backtracking'''

    if problema.test_objetivo(nodo.estado):
        return nodo
    assert(depth(nodo) < max_size_nodo)
    acciones = problema.acciones_aplicables(nodo.estado)
    for a in acciones:
        hijo = nodo_hijo(problema, nodo, a)
        if not is_cycle(hijo):
            resultado = backtracking(problema, hijo)
            if resultado is not None:
                return resultado    
    return None

def dijkstra(problema):
    s = problema.estado_inicial
    cod = problema.codigo(s)
    nodo = Nodo(s, None, None, 0, cod)
    frontera = ListaPrioritaria()
    frontera.push(nodo, 0)
    explorados = {}
    explorados[cod] = 0
    while not frontera.is_empty():
        nodo = frontera.pop()
        if problema.test_objetivo(nodo.estado):
            return nodo
        for a in problema.acciones_aplicables(nodo.estado):
            hijo = nodo_hijo(problema, nodo, a)
            s = hijo.estado
            cod = hijo.codigo
            c = hijo.costo_camino
            if (cod not in explorados.keys()) or (c < explorados[cod]):
                frontera.push(hijo, c)
                assert(len(frontera) < max_size)
                explorados[cod] = c
    return None


###############################################################################

def avara(problema, heuristica):
    s = problema.estado_inicial
    nodo = Nodo(s, None, None, 0, problema.codigo(s))
    v = 0
    frontera = ListaPrioritaria()
    frontera.push(nodo, v)
    
    while not frontera.is_empty():
        
        nodo = frontera.pop()
#        print('pop-->', nodo.estado)
        if problema.test_objetivo(nodo.estado):
            return nodo
        
        for a in problema.acciones_aplicables(nodo.estado):
            #print(nodo.estado)
            hijo = nodo_hijo(problema, nodo, a)
            
            if not is_cycle(hijo):
                s = hijo.estado
                v = problema.heuristica(s)
                #print('push-->', s)
                frontera.push(hijo, v)
           
                
    return None


##############################################################################
#Soluciones

prob = Vendedor('Chapinero')
b=profundidad(prob)
b.costo_camino



av1=avara(prob,prob.heuristica)
av1.costo_camino

av2=avara(prob,prob.heuristica2)
av2.costo_camino

###############################################################################
#Pruebas 

tiempos_Cpu= []
tiempos_Cpu1= []
tiempos_Cpu2= []
for i in range(0,300):
    inicio=time.time()
    c=profundidad(prob)
    c.costo_camino
    fin = time.time()
    intv=fin-inicio
    tiempos_Cpu.append(intv)

for i in range(0,300):
    inicio=time.time()
    c=avara(prob,prob.heuristica)
    c.costo_camino
    fin = time.time()
    intv=fin-inicio
    tiempos_Cpu1.append(intv)

for i in range(0,300):
    inicio=time.time()
    c=avara(prob,prob.heuristica2)
    c.costo_camino
    fin = time.time()
    intv=fin-inicio
    tiempos_Cpu2.append(intv)


    

Profundidad = tiempos_Cpu
Heuristica_Costo = tiempos_Cpu1
Heuristica_Distancia= tiempos_Cpu2


# Crear una lista con los tres conjuntos de datos
all_data = [Profundidad, Heuristica_Costo, Heuristica_Distancia]



# Custom colors for the boxplots
colors = ['#FF5733', '#3498DB', '#27AE60']

# Create the figure and axes
fig, ax = plt.subplots(figsize=(20, 12))

# Plot the boxplots
bp = ax.boxplot(all_data, patch_artist=True)

# Customize the colors of the boxplots
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Customize the style of the outliers
for flier in bp['fliers']:
    flier.set(marker='o', color='#34495E', alpha=0.5)

# Set axis labels and title
ax.set_xticklabels(['Profundidad', 'Avara con costo', 'Avara euclidiana'])
ax.set_ylabel('Tiempos de CPU')
ax.set_title('Rendimiento de los Algortimos')

#plt.ylim(0.000, 0.1)
# show plot
plt.show()

tiempos= [np.mean(Profundidad),np.mean(Heuristica_Costo),np.mean(Heuristica_Distancia)]
print(tiempos)

costos1=[]
costos2=[]
costos3=[]

for i in range(0,300):
    inicio=time.time()
    c=profundidad(prob)
    cc=c.costo_camino
    fin = time.time()
    intv=fin-inicio
    costos1.append(cc)

for i in range(0,300):
    inicio=time.time()
    c=avara(prob,prob.heuristica)
    cc=c.costo_camino
    fin = time.time()
    intv=fin-inicio
    costos2.append(cc)

for i in range(0,300):
    inicio=time.time()
    c=avara(prob,prob.heuristica2)
    cc=c.costo_camino
    fin = time.time()
    intv=fin-inicio
    costos3.append(cc)

costos= [np.mean(costos1),np.mean(costos2),np.mean(costos3)]
