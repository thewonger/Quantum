import cirq

#numero de qubits
import cirq
from cirq.ops import CZ, H
global c
import pandas as pd
import re

regex = r"m"


def qrandom(n): #n es el número de bits de la clave
    
    qubits = [cirq.GridQubit(i, 0) for i in range(n)]
    
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
    a= result

    #con 10 bits tenemos 1024 números 

    #Lo normalizamos para que nos aparezcan valores de 0 a 1.
    #Cuantos más qubits tengamos mejor precisión obtendremos.
    #print( a +" Asi es a")
         
    global numero
    numero=(a.data)
    
    
    return(numero)
    

b= qrandom(10)
c= str(b)
c=re.sub(regex, '', c)
c=re.sub(r' ','',c)
c = int(c)


print(type(b))
