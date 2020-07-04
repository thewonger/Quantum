# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 01:57:24 2020

@author: Jorge
"""

import ctypes
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

import random


from keras.layers import LeakyReLU

import pydot
import tensorflow as tf
from tensorflow import keras

#from tensorflow.keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
#from tensorflow.keras.layers.InputLayer import Input
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.layers import Conv1D, SimpleRNN
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import load_model


#numero de qubits
import re
regex = r"m"
import cirq
from cirq.ops import CZ, H

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


def random_batch(X_train, y_train, batch_size):
    #aqui es donde cambiamos las cosas 
    index_set = np.random.randint(0, X_train.shape[0], batch_size)
    X_batch = X_train[index_set]
    print(X_batch)
    y_batch = y_train[index_set]
    print(index_set)
    return X_batch, y_batch


model_name = 'crypto1'


m_bits = 16
k_bits = 16
c_bits = 16
pad = 'same'

a=maquina()

# Calculamos el tamaño del espacio del mensaje
m_train = 2**(m_bits) #+ k_bits)

alice_file = 'models/crypto/' + model_name + '-alice'
bob_file = 'models/crypto/' + model_name + '-bob'
eve_file = 'models/crypto/' + model_name + '-eve'

K.clear_session()

##### Red neuronal de Alice #####
#
ainput0 = Input(shape=(m_bits,)) #mensaje

ainput1 = Input(shape=(k_bits,)) #clave

ainput = concatenate([ainput0, ainput1], axis=1)


adense1 = Dense(units=(m_bits + k_bits))(ainput) # cada capa de la red neuronal la unimos con la siguiente capa.

adense1a = Activation('tanh')(adense1)

areshape = Reshape((m_bits + k_bits, 1,))(adense1a)

aconv1 = Conv1D(filters=2, kernel_size=4, strides=1, padding=pad)(areshape)


aconv1a = Activation('tanh')(aconv1)
aconv2 = Conv1D(filters=4, kernel_size=2, strides=2, padding=pad)(aconv1a)
aconv2a = Activation('tanh')(aconv2)
aconv3 = Conv1D(filters=4, kernel_size=1, strides=1, padding=pad)(aconv2a)

aconv3a = Activation('tanh')(aconv3)

aconv4 = Conv1D(filters=1, kernel_size=1, strides=1, padding=pad)(aconv3a)

aconv4a = Activation('sigmoid')(aconv4)

aoutput = Flatten()(aconv4a)

alice = Model([ainput0, ainput1], aoutput, name='alice')



#alice.summary()




##### Red neuronal de Bob #####
##

##Le metemos el texto cifrado y la clave

binput0 = Input(shape=(c_bits,)) #texto cifrado
binput1 = Input(shape=(k_bits,)) #clave
binput = concatenate([binput0, binput1], axis=1)

bdense1 = Dense(units=(c_bits + k_bits))(binput)
bdense1a = Activation('tanh')(bdense1)

breshape = Reshape((c_bits + k_bits, 1,))(bdense1a)

bconv1 = Conv1D(filters=2, kernel_size=4, strides=1, padding=pad)(breshape)
bconv1a = Activation('tanh')(bconv1)
bconv2 = Conv1D(filters=4, kernel_size=2, strides=2, padding=pad)(bconv1a)
bconv2a = Activation('tanh')(bconv2)
bconv3 = Conv1D(filters=4, kernel_size=1, strides=1, padding=pad)(bconv2a)
bconv3a = Activation('tanh')(bconv3)
bconv4 = Conv1D(filters=1, kernel_size=1, strides=1, padding=pad)(bconv3a)
bconv4a = Activation('sigmoid')(bconv4)

boutput = Flatten()(bconv4a)

bob = Model([binput0, binput1], boutput, name='bob')

bob.summary()



######## Red neuronal de Eve ###########
#
einput = Input(shape=(c_bits,)) #solo tiene el cifrado

edense1 = Dense(units=(c_bits + k_bits))(einput)
edense1a = Activation('tanh')(edense1)

edense2 = Dense(units=(c_bits + k_bits))(edense1a)
edense2a = Activation('tanh')(edense2)

ereshape = Reshape((c_bits + k_bits, 1,))(edense2a)

econv1 = Conv1D(filters=2, kernel_size=4, strides=1, padding=pad)(ereshape)
econv1a = Activation('tanh')(econv1)
econv2 = Conv1D(filters=4, kernel_size=2, strides=2, padding=pad)(econv1a)
econv2a = Activation('tanh')(econv2)
econv3 = Conv1D(filters=4, kernel_size=1, strides=1, padding=pad)(econv2a)
econv3a = Activation('tanh')(econv3)
econv4 = Conv1D(filters=1, kernel_size=1, strides=1, padding=pad)(econv3a)
econv4a = Activation('sigmoid')(econv4)

eoutput = Flatten()(econv4a)# Eve's attempt at code guessing

eve = Model(einput, eoutput, name='eve')
eve.summary()

alice.compile(loss='mse', optimizer='sgd')
bob.compile(loss='mse', optimizer='sgd')
eve.compile(loss='mse', optimizer='sgd')


# Establecemos la comunicacion de los canales uniendo inputs y outputs.
#
alicesalida = alice([ainput0, ainput1])
bobsalida = bob( [alicesalida, binput1] )# bob ve el cifrado y la clave
evesalida = eve( alicesalida )# eve solo bve el cifrado.

# Loss for Eve is just L1 distance between ainput0 and eoutput. The sum
# is taken over all the bits in the message. The quantity inside the K.mean()
# is per-example loss. We take the average across the entire mini-batch
#
eveperdida = K.mean( K.sum(K.abs(ainput0 - evesalida), axis=-1)  )

# Loss for Alice-Bob communication depends on Bob's reconstruction, but
# also on Eve's ability to decrypt the message. Eve should do no better
# than random guessing, so on average she will guess half the bits right.
#
bobperdida = K.mean(  K.sum(K.abs(ainput0 - bobsalida), axis=-1)  )
aliceperdida = bobperdida + K.square(m_bits/2 - eveperdida)/( (m_bits//2)**2 )

# Optimizer and compilation
#
aliceoptim = RMSprop(lr=0.001)
eveoptim = RMSprop(lr=0.001) #default 0.001


# Build and compile the alice model, used for training Alice-Bob networks
#
alicemodel = Model([ainput0, ainput1, binput1], bobsalida, name='alicemodel')
alicemodel.add_loss(aliceperdida)
alicemodel.compile(optimizer=aliceoptim)


# Build and compile the EVE model, used for training Eve net (with Alice frozen)
#
alice.trainable = False
evemodel = Model([ainput0, ainput1], evesalida, name='evemodel')
evemodel.add_loss(eveperdida)
evemodel.compile(optimizer=eveoptim)


alicelosses = []
boblosses = []
evelosses = []


n_epochs = 20
batch_size = 512
n_batches = m_train // batch_size

alicecycles = 1
evecycles = 2

epoch = 0
print("Training for", n_epochs, "epochs with", n_batches, "batches of size", batch_size)

while epoch < n_epochs:
    alicelosses0 = []
    boblosses0 = []
    evelosses0 = []
    for iteration in range(n_batches):
        
        # Train the A-B+E network
        #
        alice.trainable = True
        for cycle in range(alicecycles):
            # Select a random batch of messages, and a random batch of keys
            #
            m_batch = np.random.randint(0, 2, m_bits * batch_size).reshape(batch_size, m_bits)
            #k_batch = np.random.randint(0, 2, k_bits * batch_size).reshape(batch_size, k_bits)
            k_batch = maquina()
            
            #print(k_batch)
            
            loss = alicemodel.train_on_batch([m_batch, k_batch, k_batch], None)
            
        alicelosses0.append(loss)
        alicelosses.append(loss)
        aliceavg = np.mean(alicelosses0)
            
        # Evaluate Bob's ability to decrypt a message
        m_enc = alice.predict([m_batch, k_batch])
        m_dec = bob.predict([m_enc, k_batch])
        loss = np.mean(  np.sum( np.abs(m_batch - m_dec), axis=-1)  )
        boblosses0.append(loss)
        boblosses.append(loss)
        bobavg = np.mean(boblosses0)
        
        # Train the EVE network
        #
        alice.trainable = False
        for cycle in range(evecycles):
            m_batch = np.random.randint(0, 2, m_bits * batch_size).reshape(batch_size, m_bits)
            k_batch = np.random.randint(0, 2, k_bits * batch_size).reshape(batch_size, k_bits)
            loss = evemodel.train_on_batch([m_batch, k_batch], None)
        
        evelosses0.append(loss)
        evelosses.append(loss)
        eveavg = np.mean(evelosses0)
        
        if iteration % max(1, (n_batches // 100)) == 0:
            print("\rEpoch {:3}: {:3}% | alice: {:2.3f} | eve: {:2.3f} | bob: {:2.3f}".format(
                epoch, 100 * iteration // n_batches, aliceavg, eveavg, bobavg), end="")
            sys.stdout.flush()
    
    print()
    epoch += 1
    
print('Training finished.')



steps = -1

plt.figure(figsize=(7, 4))
plt.plot(alicelosses[:steps], label='A-B', color = "blue")
plt.plot(evelosses[:steps], label='Eve', color= "red")
plt.plot(boblosses[:steps], label='Bob', color = "green")
plt.xlabel("Iterations", fontsize=13)
plt.ylabel("Loss", fontsize=13)
plt.legend(fontsize=13)

#plt.savefig("images/" + model_name + ".png", transparent=True) #dpi=100
plt.show()


n_examples = 10000

m_batch = np.random.randint(0, 2, m_bits * n_examples).reshape(n_examples, m_bits)
k_batch = np.random.randint(0, 2, m_bits * n_examples).reshape(n_examples, m_bits)

m_enc = alice.predict([m_batch, k_batch])
m_dec = (bob.predict([m_enc, k_batch]) > 0.5).astype(int)
m_att = (eve.predict(m_enc) > 0.5).astype(int)

bdiff = np.abs(m_batch - m_dec)
bsum = np.sum(bdiff, axis=-1)
ediff = np.abs(m_batch - m_att)
esum = np.sum(ediff, axis=-1)

print("Bob % correct: ", 100.0*np.sum(bsum == 0) / n_examples, '%')
print("Eve % correct: ", 100.0*np.sum(esum == 0) / n_examples, '%')