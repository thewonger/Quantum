# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 01:30:44 2020

@author: Jorge
"""

import re
regex = r"m"
import cirq
from timeit import default_timer as timer
from cirq.ops import CZ, H
from scipy.stats import entropy

def qrandom(n): #n es el número de bits de la clave
    
    qubits = [cirq.GridQubit(i, 0) for i in range(n)]
    circuit = cirq.Circuit(

    #cirq.H(*qubits),
    cirq.H.on_each(*qubits),

    cirq.measure(*qubits, key='m')
    )
    #print("Circuit:")
    #print(circuit)

    # Simulamos el circuito las veces que queramos, en este caso como
    # es un generador de numeros aleatorios nos da igual.
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1)
    #print("Results:")
    #print(result)
    #print(result.data)
    a= result.data
    #cirq.plot_state_histogram(result)
    #con 10 bits tenemos 1024 números 

    #Lo normalizamos para que nos aparezcan valores de 0 a 1.
    #Cuantos más qubits tengamos mejor precisión obtendremos.
    global numero
    #print( a +" Asi es a")
        
       
    numero=str(a)

    return(numero)


import random
import numpy as np

qvector=np.array([])
vector = np.array([])

for i in range(50000):
 
  b= qrandom(4)   #16
  c= str(b)
  c=re.sub(regex, '', c)
  c=re.sub(r' ','',c)
  c = int(c)
  d=c/15
  qvector = np.append(qvector, d)

  normal = random.randint(0,15)
  normal=normal/15
  vector=np.append(vector, normal)



qmedia=np.mean(qvector)
media=np.mean(vector)


qdev=np.std(qvector)
dev=np.std(vector)

qvar=np.var(qvector)
var=np.var(vector)

qcoef = np.corrcoef(qvector)
coef=np.corrcoef(vector)

entropiaq = entropy(qvector, base=2)
entropia = entropy(vector, base=2)

print("Qmedia: " + str(qmedia))

print("media: " + str(media))


print("Qvar: " + str(qvar))


print("Var: " + str(var))

print("QDev: " + str(qdev))
print("Dev: " + str(dev))
print("QCor: " + str(qcoef))
print("Cor: " + str(coef))

print("Entropia: " + str(entropiaq))
print("Entropia" + str(entropia))

