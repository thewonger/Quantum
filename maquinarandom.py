# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 01:30:44 2020

@author: Jorge
"""

import re
regex = r"m"
import cirq
from cirq.ops import CZ, H
import random
import numpy as np

def split(word): 
    return list(word) 

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




def maquina():
    qvector=np.array([])
    vector = np.array([])


    for i in range(512):
        b= qrandom(16)   #16
        c= str(b)
        c=re.sub(regex, '', c)
        c=re.sub(r' ','',c)
        c = int(c)
        c=bin(c)
        c=re.sub(r'0b','',c)
          
        cstring=str(c)
        
        v=split(cstring)
        #print(len(v))
        while (len(v)<16):
            v.insert(0, 0)
        
        #print(len(v))
        vector= np.append(vector,v)

    vector=np.reshape(vector, (512, 16))
    arr = np.array(vector) 
    arr= arr.astype(np.int32)
  
    return(arr)

hola=maquina()
