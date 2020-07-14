# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:08:45 2020

@author: Chaitanya Sampat

This module contains all data plotting for all models

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PlotterClass:
    
    
    def history_plotter(self,history, name, valname, modelname):
        plt.plot(history.history[name])
        plt.plot(history.history[valname],'--')
        plt.title(modelname + name )
        plt.xlabel('epoch')
        plt.ylabel(name)
        plt.legend([modelname + 'train',modelname +'test'], loc='upper left')
        plt.show()

    def history_plotter_compare(self,history1, history2, name, valname, modelname1, modelname2):
        plt.plot(history1.history[name])
        plt.plot(history1.history[valname],'--')
        plt.plot(history2.history[name])
        plt.plot(history2.history[valname],'--')
        plt.title(name)
        plt.xlabel('epoch')
        plt.ylabel(name)
        plt.legend([modelname1 + 'train',modelname1 +'test',modelname2 + 'train',modelname2 +'test',], loc='upper right')
        plt.show()
    
    def history_plotter_compare3(self,history1, history2, history3, name, valname, modelname1, modelname2, modelname3):
        plt.plot(history1.history[name])
        plt.plot(history1.history[valname],'--')
        plt.plot(history2.history[name])
        plt.plot(history2.history[valname],'--')
        plt.plot(history3.history[name])
        plt.plot(history3.history[valname],'--')
        plt.title(name)
        plt.xlabel('epoch')
        plt.ylabel(name)
        plt.legend([modelname1 + 'train',modelname1 +'test',modelname2 + 'train',modelname2 +'test',modelname3 + 'train',modelname3 +'test'], loc='upper right')
        plt.show()
        
        

    def history_plotter_comparen(self,history, name, valname, modelname):
        for hist in history:
            plt.plot(hist.history[name])
            plt.plot(hist.history[valname],'--')
        
        plt.title(name)
        plt.xlabel('epoch')
        plt.ylabel(name)
        legstr = []
        for model in modelname:
            str1 = model + ' train'
            str2 = model + ' test'
            legstr.append(str1)
            legstr.append(str2)
        plt.legend(legstr, loc='upper right')
        plt.show()

    def history_plotter_comparen_noval(self,history, name, modelname):
        for hist in history:
            plt.plot(hist.history[name])
        
        plt.title(name)
        plt.xlabel('epoch')
        plt.ylabel(name)
        legstr = []
        for model in modelname:
            str1 = model + ' train'
            legstr.append(str1)
        plt.legend(legstr, loc='upper right')
    
    def parityPlot(self,test_data,test_predictions,title):
        plt.axes(aspect='equal')
        labels = ['Bin1','Bin2','Bin3','Bin4','Bin5','Bin6','Bin7','Coarse']
        plt.scatter(test_data,test_predictions)
        plt.title(title + ' Parity Plot')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        # lims = [np.floor(min(test_data)), np.round(max(test_data),decimals=0)]
        lims = [0,1]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims)
    
    def parityPlot_dens(self,test_data,test_predictions,title):
        plt.axes(aspect='equal')
        predicitons = test_predictions[0:10080]
        d = pd.DataFrame(test_data['Granule_density'])
        test = d.to_numpy().flatten()
        plt.scatter(test, predicitons)
        plt.title(title + ' Density Parity Plot')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        # lims = [np.floor(min(test_data)), np.round(max(test_data),decimals=0)]
        lims = [0,1]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims)
        
    def expDataPlot(self,test_conv,test_labels,testIdx,sieveCut,legend):
        for idx in testIdx:
            pltTest = np.transpose(np.array(test_labels.iloc[idx,1:8]))
            plt.figure()
            plt.plot(sieveCut,test_conv[idx][1:8],'-')
            plt.plot(sieveCut,pltTest,'bo')
            plt.xlabel('Sieve Fraction')
            plt.ylabel('Cummulative PSD')
            plt.legend(legend, loc='lower right')
            plt.title(str(idx))
    
            
    def expDataPlot_compare(self,test_conv1, test_conv2, test_labels1, test_labels2, testIdx,sieveCut,legend):
        for idx in testIdx:
            pltTest1 = np.transpose(np.array(test_labels1.iloc[idx,:7]))
            pltTest2 = np.transpose(np.array(test_labels2.iloc[idx,:7]))
            plt.figure()
            plt.plot(sieveCut,test_conv1[idx][:7],'-')
            plt.plot(sieveCut,test_conv2[idx][:7],'-')
            plt.plot(sieveCut,pltTest1,'bo')
            plt.plot(sieveCut,pltTest2,'bo')
            plt.xlabel('Sieve Fraction')
            plt.ylabel('Cummulative PSD')
            plt.title(str(idx))
            plt.legend(legend, loc='lower right')
            

    def expDataPlot_compare3(self,test_conv1, test_conv2, test_conv3, test_labels1, testIdx,sieveCut,legend):
        for idx in testIdx:
            pltTest1 = np.transpose(np.array(test_labels1.iloc[idx,:7]))
            # pltTest2 = np.transpose(np.array(test_labels2.iloc[idx,:7]))
            # pltTest3 = np.transpose(np.array(test_labels3.iloc[idx,:7]))
            plt.figure()
            plt.plot(sieveCut,test_conv1[idx][:7],'-')
            plt.plot(sieveCut,test_conv2[idx][:7],'-')
            plt.plot(sieveCut,test_conv3[idx][:7],'-')
            plt.plot(sieveCut,pltTest1,'bo')
            # plt.plot(sieveCut,pltTest2,'bo')
            # plt.plot(sieveCut,pltTest3,'bo')
            plt.xlabel('Sieve Fraction')
            plt.ylabel('Cummulative PSD')
            plt.title(str(idx))
            plt.legend(legend, loc='lower right')
            
    def expDataPlot_comparen(self,test_conv, test_labels1, testIdx,sieveCut,legend):
        fig, ax = plt.subplots(int(len(testIdx)/2), 2, sharey=True)
        counter = 0
        col=0
        for idx in testIdx:
            pltTest1 = np.transpose(np.array(test_labels1.iloc[idx,1:8]))
            # pltTest2 = np.transpose(np.array(test_labels2.iloc[idx,:7]))
            # pltTest3 = np.transpose(np.array(test_labels3.iloc[idx,:7]))
            # plt.figure()
            for tc in test_conv:
                ax[counter,col].plot(sieveCut,tc[idx][1:8],'-')
            ax[counter,col].plot(sieveCut,pltTest1,'bo')
            # plt.plot(sieveCut,pltTest2,'bo')
            # plt.plot(sieveCut,pltTest3,'bo')
            ax[counter,col].set_xlabel('Sieve Fraction')
            ax[counter,col].set_ylabel('Cummulative PSD')
            ax[counter,col].title.set_text(str(idx))
            if (counter < 2):
                counter = counter + 1
            else:
                counter = 0
                col = 1
           
        plt.legend(legend, loc='lower right')