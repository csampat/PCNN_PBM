# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:08:45 2020

@author: Chaitanya Sampat

This module contains all data manipulation for PINN model

"""
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import keras.backend as K


from keras import regularizers
from keras.layers import Dense, Input, Concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD

class HelperFunction:
    def __init__(self, dataFile, sieveCut, yieldStrength=1e5):
        self.sieveCut = sieveCut
        self.dataFile = dataFile
        self.yieldStrength = yieldStrength
        self.normed_train_dataset, self.train_labels, self.normed_test_dataset, self.test_labels = self.normedDataSplitWithDensity(self.dataFile)

    def norm(self, x, train_stats):
        return (x - train_stats['min']) / (train_stats['max'] - train_stats['min'])
        #return (x - train_stats['mean']) / train_stats['std']

    
    def normedDataSplit(self, dataFile):
        train_dataset = dataFile.sample(frac=0.8,random_state=0)
        test_dataset = dataFile.drop(train_dataset.index)
        
        labels = ['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']
        
        train_labels = pd.DataFrame([train_dataset.pop(i) for i in labels]).T
        test_labels = pd.DataFrame([test_dataset.pop(i) for i in labels]).T
        
        train_stats = train_dataset.describe()
        train_stats = train_stats.transpose()
        
        test_stats = test_dataset.describe()
        test_stats = test_stats.transpose()
        
        normed_train_dataset = self.norm(train_dataset, train_stats)
        normed_test_dataset = self.norm(test_dataset, test_stats)
        
        return normed_train_dataset, train_labels, normed_test_dataset, test_labels
    
    def normedDataSplitWithDensity(self, dataFile):
        train_dataset = dataFile.sample(frac=0.8,random_state=0)
        test_dataset = dataFile.drop(train_dataset.index)
        
        labels = ['Granule_density','Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']
        
        train_labels = pd.DataFrame([train_dataset.pop(i) for i in labels]).T
        test_labels = pd.DataFrame([test_dataset.pop(i) for i in labels]).T
        
        train_stats = train_dataset.describe()
        train_stats = train_stats.transpose()
        
        test_stats = test_dataset.describe()
        test_stats = test_stats.transpose()
        
        normed_train_dataset = self.norm(train_dataset, train_stats)
        normed_test_dataset = self.norm(test_dataset, test_stats)

        train_label_stats = train_labels['Granule_density'].describe()
        train_label_stats = train_label_stats.transpose()
        

        test_label_stats = test_labels['Granule_density'].describe()
        test_label_stats = test_label_stats.transpose()

        train_labels['Granule_density'] = self.norm(train_labels['Granule_density'], train_label_stats)
        test_labels['Granule_density'] = self.norm(test_labels['Granule_density'], test_label_stats)

        return normed_train_dataset, train_labels, normed_test_dataset, test_labels


############# ANN MODELS ##################################


    def get_callbacks(self,pat=100):
      return [tfdocs.modeling.EpochDots(),
              tf.keras.callbacks.EarlyStopping(monitor='mse', patience=pat)]
    
    def build_train_model_2intLayers(self,normed_train_dataset, train_labels, patience_model, nOutput, nNodes=8, actFn='relu', EPOCHS=100):
        model = Sequential([
            Dense(nNodes, activation=actFn, input_shape=[len(normed_train_dataset.keys())]),
            Dense(nNodes, activation=actFn),
            Dense(nOutput, activation='linear')])
    
        model.compile(optimizer=SGD(learning_rate=0.001,momentum=0.0, nesterov=False),
                  loss='mse', metrics = ['mae','mse'])
        print(model.summary())
        history = model.fit(normed_train_dataset, train_labels, epochs=EPOCHS, 
                            validation_split = 0.33, verbose=0, callbacks=self.get_callbacks(patience_model))
        
        return model, history

    def build_train_model_3intLayers(self,normed_train_dataset, train_labels, patience_model, nOutput, nNodes=8, actFn='relu', EPOCHS=100):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(nNodes, activation=actFn, input_shape=[len(normed_train_dataset.keys())]),
            tf.keras.layers.Dense(nNodes, activation=actFn),
            tf.keras.layers.Dense(nNodes, activation=actFn),
            tf.keras.layers.Dense(nOutput, activation='sigmoid')])
    
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.0, nesterov=False),
                  loss='mse', metrics = ['mae','mse'])
        print(model.summary())
        history = model.fit(normed_train_dataset, train_labels, epochs=EPOCHS, 
                            validation_split = 0.33, verbose=0, callbacks=self.get_callbacks(patience_model))
        
        return model, history
    
    def build_train_regularizedModel_2intLayers(self,normed_train_dataset, train_labels, patience_model, nOutput, nNodes=8, actFn='relu', EPOCHS=100):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(nNodes, activation=actFn, 
                                  input_shape=[len(normed_train_dataset.keys())], 
                                  kernel_regularizer=regularizers.l2(0.001),),
            tf.keras.layers.Dense(nNodes, activation=actFn, kernel_regularizer=regularizers.l2(0.001),),
            tf.keras.layers.Dense(nOutput, activation='sigmoid')])
    
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.0, nesterov=False),
                  loss='mse', metrics = ['mae','mse'])
        print(model.summary())
        history = model.fit(normed_train_dataset, train_labels, epochs=EPOCHS, 
                            validation_split = 0.33, verbose=0, callbacks=self.get_callbacks(patience_model))
        
        return model, history
    
    
    def build_train_PINNstde(self,normed_train_dataset, train_labels, patience_model, nOutput, nNodes=8, actFn='relu', EPOCHS=100):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(nNodes, activation=actFn, 
                                  input_shape=[len(normed_train_dataset.keys())], 
                                  kernel_regularizer=regularizers.l2(0.001),),
            tf.keras.layers.Dense(nNodes, activation=actFn, kernel_regularizer=regularizers.l2(0.001),),
            tf.keras.layers.Dense(nOutput, activation='sigmoid')])
    
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.0, nesterov=False),
                  loss=self.lossFunc_StdeOnly, metrics = ['mae','mse'])
        print(model.summary())
        history = model.fit(normed_train_dataset, train_labels, epochs=EPOCHS, 
                            validation_split = 0.33, verbose=0, callbacks=self.get_callbacks(patience_model))
        
        return model, history
    
    def build_train_PINNStdeSmax(self,normed_train_dataset, train_labels, patience_model, nOutput, nNodes=8, actFn='relu', EPOCHS=100):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(nNodes, activation=actFn, 
                                  input_shape=[len(normed_train_dataset.keys())]),
            tf.keras.layers.Dense(nNodes, activation=actFn),
            tf.keras.layers.Dense(nNodes, activation=actFn),
            tf.keras.layers.Dense(nOutput, activation='sigmoid')])
    
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.0, nesterov=False),
                  loss=self.lossFunc_StdeSmax, metrics = ['mae','mse'])
        print(model.summary())
        history = model.fit(normed_train_dataset, train_labels, epochs=EPOCHS, 
                            validation_split = 0.33, verbose=0, callbacks=self.get_callbacks(patience_model))
        
        return model, history
    
    def build_train_PINNstde3(self,normed_train_dataset, train_labels, patience_model, nOutput, nNodes=8, actFn='relu', EPOCHS=100):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(nNodes, activation=actFn, 
                                  input_shape=[len(normed_train_dataset.keys())]),
            tf.keras.layers.Dense(nNodes, activation=actFn),
            tf.keras.layers.Dense(nNodes, activation=actFn),
            tf.keras.layers.Dense(nOutput, activation='sigmoid')])
    
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.0, nesterov=False),
                  loss=self.lossFunc_StdeOnly, metrics = ['mae','mse'])
        print(model.summary())
        history = model.fit(normed_train_dataset, train_labels, epochs=EPOCHS, 
                            validation_split = 0.33, verbose=0, callbacks=self.get_callbacks(patience_model))
        
        return model, history
    
    def build_train_PINNStdeSmax3(self,normed_train_dataset, train_labels, patience_model, nOutput, nNodes=8, actFn='relu', EPOCHS=100):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(nNodes, activation=actFn, 
                                  input_shape=[len(normed_train_dataset.keys())], 
                                  kernel_regularizer=regularizers.l2(0.001),),
            tf.keras.layers.Dense(nNodes, activation=actFn, kernel_regularizer=regularizers.l2(0.001),),
            tf.keras.layers.Dense(nOutput, activation='sigmoid')])
    
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.0, nesterov=False),
                  loss=self.lossFunc_StdeSmax, metrics = ['mae','mse'])
        print(model.summary())
        history = model.fit(normed_train_dataset, train_labels, epochs=EPOCHS, 
                            validation_split = 0.33, verbose=0, callbacks=self.get_callbacks(patience_model))
        
        return model, history
    
    
    def build_train_PINNwithDensity(self,normed_train_dataset, train_labels, patience_model, nOutput, nNodes=8, actFn='relu', EPOCHS=100):
        model = tf.keras.Sequential([
                tf.keras.layers.Dense(nNodes, activation=actFn, 
                                    input_shape=[len(normed_train_dataset.keys())], 
                                    kernel_regularizer=regularizers.l2(0.001),),
                tf.keras.layers.Dense(nNodes, activation=actFn, kernel_regularizer=regularizers.l2(0.001),),
                tf.keras.layers.Dense(nOutput, activation='linear')])
        
        
        model.add_loss(loss)
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.0, nesterov=False), loss=self.lossFunc_DensityStde(model.output[:,0]),  metrics = ['mae','mse'])
        print(model.summary())
        history = model.fit(normed_train_dataset, train_labels, epochs=EPOCHS, 
                                validation_split = 0.0, verbose=0, callbacks=self.get_callbacks(patience_model))
            
        return model, history

    
    def build_train_PINN_mul(self,normed_train_dataset, train_labels, patience_model, nOutput, nNodes=16, actFn='relu', EPOCHS=100):

        input_layer = Input(shape=(6,))
        dense_1 = Dense(nNodes, activation=actFn)(input_layer)
        dense_2 = Dense(nNodes, activation=actFn, kernel_regularizer=regularizers.l1(0.001))(dense_1)
        # separating outputs for density and GSD for custom loss function
        output_1 = Dense(1,activation='linear')(dense_2)
        output_2 = Dense(8,activation='sigmoid')(dense_2)

        train_labels_copy = pd.DataFrame.copy(train_labels)
        label1 = train_labels['Granule_density']
        labels = ['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']
        
        # label2 = ['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']
        label2 = pd.DataFrame([train_labels_copy.pop(i) for i in labels]).T
        model = Model(inputs=[input_layer], outputs=[output_1,output_2])

        model.compile(optimizer=SGD(learning_rate=0.001,momentum=0.0, nesterov=False),loss=self.lossFunc_DensityStde(output_1), metrics = ['mae','mse'])
        print(model.summary())
        history = model.fit(normed_train_dataset, [label1, label2], epochs=EPOCHS, 
                            verbose=1)
                
        return model, history

    
########### PHYSICS CALCULATIONS #############################################           
    def d50FromPSDCalculator(self,PSDvals):
        sieves = len(self.sieveCut)
        sieveCut = self.sieveCut
        dataPts = len(PSDvals)
        d50 = np.zeros((dataPts,))
        
        for i in range(dataPts):
            lowID = 0
            highID = 1
            # to calculate d50 we need the sieve value at 0.5
            for j in range(sieves):
                if(PSDvals[i,j] < 0.5):
                    lowID = j
                    if(j<6):
                        highID = j+1
                    else:
                        highID=j
                        lowID=j-1
            
            # matching slopes to get the intermidiate value at 0.5
            if (highID == lowID):
                d50[i] = sieveCut[highID]    
            else:
                d = (PSDvals[i,highID] - PSDvals[i,lowID]) # denominator
                if (d==0):
                    d50[i] = sieveCut[lowID]    
                else:
                    m = sieveCut[highID] * ((0.5 - PSDvals[i,lowID]) / d) #  term1
                    n = sieveCut[lowID] * ((PSDvals[i,highID] - 0.5) / d) # second term
                    
                    d50[i] = m + n
            
        d50[d50<0] = sieveCut[0]
        return d50
        
    
    def stdeCalculation(self):
        particleDensity = np.array(self.dataFile['solidDensity'])
        dImp = np.array(self.dataFile['impellerDiameter'])
        rpm = np.array(self.dataFile['rpm'])
        bins = np.array((self.dataFile[['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']]).values)
        yS = np.full((len(particleDensity),), 2 * self.yieldStrength)
        l = (len(particleDensity),)
        d50 = self.d50FromPSDCalculator(bins)*1e-2
        self.dataFile['d50'] = d50
        a = np.full(l,((np.pi * 0.15) / 60))
        # nParticles = np.divide(particleDensity,((4/3)*np.pi*np.power(d50/2,3)))
        Uc = np.multiply(a,np.multiply(rpm,dImp))# shear=pi*rpm*diameter/60
        # Uc = np.multiply(d50/2,shear)
        StDe = np.divide(np.multiply(particleDensity,(np.square(Uc))), yS)
        self.dataFile['StDe'] = StDe
        
        return self.dataFile
    
    def smaxCalculation(self):
        particleDensity = np.array(self.dataFile['solidDensity'])
        solidWt = np.array(self.dataFile['batch_amount'])
        liqWt = np.array(self.dataFile['liq_amount'])
        # iniPor = np.array(self.dataFile['Initial_Porosity'])
        minPorVal = 0.18
        minPor = np.ones(len(liqWt,)) * minPorVal
        liqDen = 1000
        n1 = (1 - minPor) # part of numerator
        d1 = liqDen * minPor # demonimator
        w = np.divide(liqWt, solidWt)
        num = np.multiply(w,np.multiply(particleDensity,n1))
        smax = np.divide(num,d1)
        self.dataFile['Smax'] = smax
        
        return self.dataFile
        
        
            
    def stdeBasedDataRemoval(self):
        
        dataFile1 = self.stdeCalculation()
        dataFile1 = dataFile1.dropna()
        dataFile1 = dataFile1[dataFile1['StDe'] <= 0.2]  
        dataFile1 = dataFile1.drop(columns=['StDe'])
        dataFile1 = dataFile1.drop(columns=['d50'])
        return dataFile1
    
    
    def stdesmaxDataRemoval(self):
      
        dataFile1 = self.stdeCalculation()
        dataFile1 = self.smaxCalculation()
        dataFile1 = dataFile1.dropna()
        dataFile1 = dataFile1[dataFile1['StDe'] <= 0.2]
        dataFile1 = dataFile1[dataFile1['Smax'] >= 0.2]
        dataFile1 = dataFile1.drop(columns=['StDe'])
        dataFile1 = dataFile1.drop(columns=['Smax'])
        dataFile1 = dataFile1.drop(columns=['d50'])
        return dataFile1
    
    
    def lossFunc_StdeOnly(self,yTrue,yPred):
        dataFile1 = self.stdeCalculation() # get values StDE calculated
        stdeVal = np.array(dataFile1['StDe']) # separating out the StDe values into an numpy array
        exVal = stdeVal[stdeVal>0.2]
        exVal = (exVal - 0.2) / len(exVal)
        addError = exVal.mean()
        
        return K.mean(K.square(yTrue - yPred)) + addError
    
    def lossFunc_StdeSmax(self,yTrue,yPred):
        dataFile1 = self.stdeCalculation() # get values StDE calculated
        stdeVal = np.array(dataFile1['StDe']) # separating out the StDe values into an numpy array
        stError = stdeVal[stdeVal>0.2]
        stError = (stError - 0.2) / len(stError)
        dataFile1 = self.smaxCalculation()
        smax = np.array(dataFile1['Smax'])
        smaxError = smax[smax<0.5]
        smaxError = (0.5 - smaxError) / len(smaxError)
        addError = K.mean(stError) + K.mean(smaxError)
        
        return K.mean(K.square(yTrue - yPred)) + addError
       
    def lossFunc_DensityStde(self,output_1):
        def loss(yTrue,yPred):
            # rpm = np.array(self.normed_train_dataset['rpm'])
            # granuleDensity = K.slice(yPred, 0, len(rpm))
            # granuleDensity = granuleDensity.numpy()
            
            Uc, Ys = self.stdePreCalc()
            
            Uc = np.array(Uc)
            Ys = np.array(Ys)
            b = np.power(Uc,2)
            c = np.dot(Ys,b)
            d = np.full(len(Uc),0.2)
            rho_comp = np.divide(d,c)
            
            rho_l = output_1 - rho_comp
            rho_l = rho_l > 0

            addError = K.mean(rho_l)
            
        
            return K.mean(K.square(yTrue - yPred)) + addError
        return loss
              
    def stdePreCalc(self):
       dImp = np.array(self.normed_train_dataset['impellerDiameter'])
       rpm = np.array(self.normed_train_dataset['rpm'])
       m = np.dot(rpm,dImp)
       l = len(rpm)
       a = np.full(l,((np.pi * 0.15) / 60))
       Uc = np.dot(m,a)
       ys = np.full(l,(1 / (2 * self.yieldStrength)))
       return Uc, ys
        
    
    def calculateR2(self, y_predicted, y_true):
        labels = ['Granule_Density','Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']
        y_predicted_mean = np.array(y_predicted).mean()
        y_true_reshaped = np.reshape(np.ravel(y_true),(len(y_true)*len(labels),))
        '''
        test['output_pred'] = model.predict(x=np.array(test[inputs]))
        output_mean = test['output'].mean()    # Is this the correct mean value for r2 here?
        test['SSres'] = np.square(test['output']-test['output_pred'])
        test['SStot'] = np.square(test['output']-output_mean)
        r2 = 1-(test['SSres'].sum()/(test['SStot'].sum()))
        '''    
        SSres = np.square(y_true_reshaped - y_predicted)
        SStol = np.square(y_true_reshaped - y_predicted_mean)
        
        r2 = 1 - (SSres.sum() / SStol.sum())          
        return r2
        

    
        
# dataFile = pd.read_csv('./PBM_simData_noSieve.csv')    

# bins = np.array((dataFile[['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']]).values)
# sieveCut = [150,250,350,600,850,1000,1400]
# a = HelperFunction(dataFile,sieveCut,1e5)
# st = a.stdeCalculation()
# st1 = a.smaxCalculation()

# st1.to_csv('inputdata.csv')

# a = HelperFunction(dataFile,sieveCut,1e5)
# p = [1,2,5,4,1,6,8]
# s = [1,2,4,3,2,5,8]
# r2 = a.calculateR2(p, s)
# print(r2)