# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 02:13:30 2020

@author: Jorge
"""

#numero de qubits
import cirq
from cirq.ops import CZ, H
import sympy
import re


qubits = [cirq.GridQubit(i, 0) for i in range(10)]
circuit = cirq.Circuit(
  
  #cirq.H(*qubits),
  cirq.H.on_each(*qubits),
 
  cirq.measure(*qubits, key='m')
)
print("Circuit:")
print(circuit)

# Simulamos el circuito las veces que queramos, en este caso como
# es un generador de numeros aleatorios nos da igual.
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1)
print("Results:")
print(result)
print(result.data)
a= result.measurements

#con 10 bits tenemos 1024 números 

#Lo normalizamos para que nos aparezcan valores de 0 a 1.
#Cuantos más qubits tengamos mejor precisión obtendremos.

#numeroAleatorio=a*1/1023
#print(numeroAleatorio)
cosa=""
for i in a['m']:
    cosa=i
    
    
