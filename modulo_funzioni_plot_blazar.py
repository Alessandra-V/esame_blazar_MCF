"""
Modulo funzioni per realizzazione di grafici relativi allo studio della periodicità dei Blazar

Autore: Valenti Alessandra

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc
import math
from scipy import  fft, optimize
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib import colors
from matplotlib.colors import Normalize
import colorsys
plt.rcParams['text.usetex'] = True # per utilizzare il pacchetto di latex



def plot_all(diz1, diz2, diz3, diz4, base_temp, arr_col1, arr_col2):
    """
    Dati i dizionari delle 4 fonti da graficare (idealmente tutte su base mensile o settimanale)
    realizza il corrispettivo grafico contenente la curva di luce, gli erorri e gli upper limit

    Parametri:
    -----------------
    diz1, diz2, diz3, diz4 (dictionary) : dizionari contenenti i dati
    base_temp (string)                  : stringa che indica se i dati della fonte sono in base settimanale o mensile
                                          ATTENZIONE! i valori accettati sono (M , W)
    arr_col1, arr_col2 (array)          : array contenente i colori utilizzati per realizzare i grafici
    

    """
    if base_temp == "M":
        base = "mensile"
    if base_temp == "W":
        base = "settimanale"

    fig, ax = plt.subplots( 2 , 2 )
    fig.suptitle('Curve di Luce su base {} delle fonti'.format(base), fontsize=16)
        
    fig.subplots_adjust(hspace=0.2)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))

    ax[0,0].plot(diz1["tempo_data"], diz1["flusso"], color = arr_col1[0], alpha = 0.5, label = diz1["nome"] )
    ax[0,1].plot(diz2["tempo_data"], diz2["flusso"], color = arr_col1[1], alpha = 0.5, label = diz2["nome"] )
    ax[1,0].plot(diz3["tempo_data"], diz3["flusso"], color = arr_col1[2], alpha = 0.5, label = diz3["nome"] )
    ax[1,1].plot(diz4["tempo_data"], diz4["flusso"], color = arr_col1[3], alpha = 0.5, label = diz4["nome"] )

    ax[0,0].errorbar(diz1["tempo_data"], diz1["flusso"], yerr = diz1["flusso_err"], color = arr_col1[0], alpha = 0.5, fmt = 'o')
    ax[0,1].errorbar(diz2["tempo_data"], diz2["flusso"], yerr = diz2["flusso_err"], color = arr_col1[1], alpha = 0.5, fmt = 'o')
    ax[1,0].errorbar(diz3["tempo_data"], diz3["flusso"], yerr = diz3["flusso_err"], color = arr_col1[2], alpha = 0.5, fmt = 'o')
    ax[1,1].errorbar(diz4["tempo_data"], diz4["flusso"], yerr = diz4["flusso_err"], color = arr_col1[3], alpha = 0.5, fmt = 'o')

    ax[0,0].scatter(  diz1["upper_lim_data"],diz1['upper_lim_flusso'], color = arr_col2[0], marker = '*', s=100, zorder=3,  label= 'upper limits'  )
    ax[0,1].scatter(  diz2["upper_lim_data"],diz2['upper_lim_flusso'], color = arr_col2[1], marker = '*', s=100, zorder=3,  label= 'upper limits'  )
    ax[1,0].scatter(  diz3["upper_lim_data"],diz3['upper_lim_flusso'], color = arr_col2[2], marker = '*', s=100, zorder=3,  label= 'upper limits'  )
    ax[1,1].scatter(  diz4["upper_lim_data"],diz4['upper_lim_flusso'], color = arr_col2[3], marker = '*', s=100, zorder=3,  label= 'upper limits'  )

    ax[0,0].set_ylabel(r'$\textrm{Flusso di Fotoni} \quad  [0.1-100 GeV](\textrm{photons} \quad cm^{-2} s^{-1})$', fontsize=10)
    ax[0,1].set_ylabel(r'$\textrm{Flusso di Fotoni} \quad  [0.1-100 GeV](\textrm{photons} \quad cm^{-2} s^{-1})$', fontsize=10)
    ax[1,0].set_ylabel(r'$\textrm{Flusso di Fotoni} \quad  [0.1-100 GeV](\textrm{photons} \quad cm^{-2} s^{-1})$', fontsize=10)
    ax[1,1].set_ylabel(r'$\textrm{Flusso di Fotoni} \quad  [0.1-100 GeV](\textrm{photons} \quad cm^{-2} s^{-1})$', fontsize=10)

    ax[0,0].set_xlabel("Tempo")
    ax[0,1].set_xlabel("Tempo")
    ax[1,0].set_xlabel("Tempo")
    ax[1,1].set_xlabel("Tempo")
    
    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()  
    ax[1,1].legend()

    plt.show()




#--------------------------------------------------------------------------------
def plot_all_pwsp(diz1, diz2, diz3, diz4, base_temp, arr_col1, log = False, interp = False):
    """
    Dati i dizionari delle 4 fonti da graficare (idealmente tutte su base mensile o settimanale)
    realizza il corrispettivo grafico contenente gli spettri di potenza delle fonti 

    Parametri:
    -----------------
    diz1, diz2, diz3, diz4 (dictionary) : dizionari contenenti i dati
    base_temp              (string)     : stringa che indica se i dati della fonte sono in base settimanale o mensile
                                          ATTENZIONE! i valori accettati sono (M , W)
    arr_col1               (array)      : array contenente i colori utilizzati per realizzare i grafici
    log                    (boolean)    : valore booleano, se è uguale a True, restituisce i grafici logarimtici
    interp                 (boolean)    : variabile booleana che indica se il grafico dello spettro di potenza deve essere fatto sui dati interpolati o no
                                          interp = False =>  sui dati NON interpolati
                                          interp = True  =>  sui dati interpolati
    
    """
           
    if interp == False:
        freq1 = diz1["frequenza"]
        freq2 = diz2["frequenza"]
        freq3 = diz3["frequenza"]
        freq4 = diz4["frequenza"]
        pot1 = diz1["ck"]
        pot2 = diz2["ck"]
        pot3 = diz3["ck"]
        pot4 = diz4["ck"]
            
    if interp == True:
        freq1 = diz1["frequenza interp"]
        freq2 = diz2["frequenza interp"]
        freq3 = diz3["frequenza interp"]
        freq4 = diz4["frequenza interp"]
        pot1 = diz1["ck interp"]
        pot2 = diz2["ck interp"]
        pot3 = diz3["ck interp"]
        pot4 = diz4["ck interp"]

    if base_temp == "M":
        base = "mensile"
    if base_temp == "W":
        base = "settimanale"

    fig, ax = plt.subplots( 2 , 2 )
    
    fig.suptitle('Spettri di potenza su base {} delle fonti'.format(base), fontsize=16)

    ax[0,0].plot(freq1[:len(freq1)//2], np.abs(pot1[:len(pot1)//2])**2, color = arr_col1[0], alpha = 0.8, label = diz1["nome"] )
    ax[0,1].plot(freq2[:len(freq2)//2], np.abs(pot2[:len(pot2)//2])**2, color = arr_col1[1], alpha = 0.8, label = diz2["nome"] )
    ax[1,0].plot(freq3[:len(freq3)//2], np.abs(pot3[:len(pot3)//2])**2, color = arr_col1[2], alpha = 0.8, label = diz3["nome"] )
    ax[1,1].plot(freq4[:len(freq4)//2], np.abs(pot4[:len(pot4)//2])**2, color = arr_col1[3], alpha = 0.8, label = diz4["nome"] )
    
    ax[0,0].set_xlabel(r'Frequenza $[Hz]$ ', fontsize=16)
    ax[0,1].set_xlabel(r'Frequenza $[Hz]$ ', fontsize=16)
    ax[1,0].set_xlabel(r'Frequenza $[Hz]$ ', fontsize=16)
    ax[1,1].set_xlabel(r'Frequenza $[Hz]$ ', fontsize=16)
    
    ax[0,0].set_ylabel(r'$|C_k|^2$', fontsize=16)
    ax[0,1].set_ylabel(r'$|C_k|^2$', fontsize=16)
    ax[1,0].set_ylabel(r'$|C_k|^2$', fontsize=16)
    ax[1,1].set_ylabel(r'$|C_k|^2$', fontsize=16)

    if log == True:
        fig.text(0.5, 0.93, "In scala logaritmica", ha='center', va='center', fontsize=12)

        ax[0,0].set_xscale('log')
        ax[0,1].set_xscale('log')
        ax[1,0].set_xscale('log')
        ax[1,1].set_xscale('log')
        
        ax[0,0].set_yscale('log')
        ax[0,1].set_yscale('log')
        ax[1,0].set_yscale('log')
        ax[1,1].set_yscale('log')
    
    ax[0,0].legend()
    ax[1,0].legend()
    ax[0,1].legend()
    ax[1,1].legend()

    plt.show()


#-----------------------------------------------------------------------------------------

def plot_all_pwsp_fit(diz1, diz2, diz3, diz4, base_temp, arr_col1, arr_col2, interp = False, log = True):

    if interp == False:
        freq1 = diz1["frequenza"]
        freq2 = diz2["frequenza"]
        freq3 = diz3["frequenza"]
        freq4 = diz4["frequenza"]
        
        pot1  = diz1["ck"]
        pot2  = diz2["ck"]
        pot3  = diz3["ck"]
        pot4  = diz4["ck"]

    if interp == True:
        freq1 = diz1["frequenza interp"]
        freq2 = diz2["frequenza interp"]
        freq3 = diz3["frequenza interp"]
        freq4 = diz4["frequenza interp"]
        
        pot1  = diz1["ck interp"]
        pot2  = diz2["ck interp"]
        pot3  = diz3["ck interp"]
        pot4  = diz4["ck interp"]
        
    fit1 = diz1["dati_fit"]
    fit2 = diz2["dati_fit"]
    fit3 = diz3["dati_fit"]
    fit4 = diz4["dati_fit"]

    if base_temp == "M":
        base = "mensile"
    if base_temp == "W":
        base = "settimanale"

    fig , ax = plt.subplots(2,2)
    
    fig.suptitle('Fit dell analisi spettrale delle sorgenti su base {}'.format(base), fontsize=16)
    fig.subplots_adjust(hspace=0.08)
  
    if log == False:
        fig.subplots_adjust(hspace=0.3)
        a = 0.6
        b = 0.6
        

    ax[0,0].plot(freq1[1:len(freq1)//2], np.abs(pot1[1:len(pot1)//2]**2), color = arr_col1[0], alpha = 0.6, label = diz1["nome"] )
    ax[0,1].plot(freq2[1:len(freq2)//2], np.abs(pot2[1:len(pot2)//2]**2), color = arr_col1[1], alpha = 0.6, label = diz2["nome"] )
    ax[1,1].plot(freq3[1:len(freq3)//2], np.abs(pot3[1:len(pot3)//2]**2), color = arr_col1[2], alpha = 0.6, label = diz3["nome"] )
    ax[1,0].plot(freq4[1:len(freq4)//2], np.abs(pot4[1:len(pot4)//2]**2), color = arr_col1[3], alpha = 0.6, label = diz4["nome"] )

    ax[0,0].plot(freq1[1:len(freq1)//2], fit1, color = arr_col2[0], alpha = 0.9, label = "{} Fit".format(diz1["nome"]))
    ax[0,1].plot(freq2[1:len(freq2)//2], fit2, color = arr_col2[1], alpha = 0.9, label = "{} Fit".format(diz2["nome"]))
    ax[1,1].plot(freq3[1:len(freq3)//2], fit3, color = arr_col2[2], alpha = 0.9, label = "{} Fit".format(diz3["nome"]))
    ax[1,0].plot(freq4[1:len(freq4)//2], fit4, color = arr_col2[3], alpha = 0.9, label = "{} Fit".format(diz4["nome"]))

    ax[0,0].set_xlabel(r'Frequenza $[Hz]$ ')
    ax[0,1].set_xlabel(r'Frequenza $[Hz]$ ') 
    ax[1,1].set_xlabel(r'Frequenza $[Hz]$ ')
    ax[1,0].set_xlabel(r'Frequenza $[Hz]$ ')

    ax[0,0].set_ylabel(r'$|C_k|^2$')
    ax[0,1].set_ylabel(r'$|C_k|^2$')
    ax[1,1].set_ylabel(r'$|C_k|^2$')
    ax[1,0].set_ylabel(r'$|C_k|^2$')

    if log == True:     
        ax[0,0].set_xscale('log')
        ax[0,1].set_xscale('log')
        ax[1,1].set_xscale('log')
        ax[1,0].set_xscale('log')
    
        ax[0,0].set_yscale('log')
        ax[0,1].set_yscale('log')
        ax[1,1].set_yscale('log')
        ax[1,0].set_yscale('log')
        a = 0.1
        b = 0.1
        

    ax[0,0].text(a, b, r'$\beta$ = {:1.2f} $\pm$ {:1.2f}'.format(diz1["params fit"][1], math.sqrt(diz1["params covariance fit"][1,1])), fontsize=18, color =  arr_col2[0], transform=ax[0,0].transAxes)
    ax[0,1].text(a, b, r'$\beta$ = {:1.2f} $\pm$ {:1.2f}'.format(diz2["params fit"][1], math.sqrt(diz2["params covariance fit"][1,1])), fontsize=18, color =  arr_col2[1], transform=ax[0,1].transAxes)
    ax[1,1].text(a, b, r'$\beta$ = {:1.2f} $\pm$ {:1.2f}'.format(diz3["params fit"][1], math.sqrt(diz3["params covariance fit"][1,1])), fontsize=18, color =  arr_col2[2], transform=ax[1,1].transAxes)
    ax[1,0].text(a, b, r'$\beta$ = {:1.2f} $\pm$ {:1.2f}'.format(diz4["params fit"][1], math.sqrt(diz4["params covariance fit"][1,1])), fontsize=18, color =  arr_col2[3], transform=ax[1,0].transAxes)
    
    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,1].legend()
    ax[1,0].legend()

    plt.show()


#---------------------------------------------------------------------

def istogramma_singificatività(picchi, periodo, colore1, colore2, n_bins, base_temp ):

    if base_temp == "M":
        strng = "mensile"
    if base_temp == "W":
        strng = "settimanale"

    plt.title("Istogramma della distribuzione dei massimi delle curve sintetiche in base {}".format(strng))
    
    n, bis, p = plt.hist(np.abs(picchi)**2, bins = n_bins , color = colore1,  edgecolor = "black", density = True, linewidth = 0.2, label = "Massimi delle curve sintetiche")
    
    plt.axvline(np.abs(periodo)**2, color = colore2, linestyle = '--',linewidth = 3, label = "Picco originale" )
    plt.xlabel(r"$|C_k^2|$", fontsize=15)
    plt.ylabel(r"Densità di Probabilità", fontsize=15)
    plt.legend()
    plt.show()


def plot_all_hist(picchi1, picchi2, picchi3, picchi4, p1, p2, p3, p4, base_temp, colori2, n_bins):

    fig , ax = plt.subplots(2,2)
    plt.subplots_adjust(wspace=0.1)

    if base_temp == "M":
        base = "mensile"
    if base_temp == "W":
        base = "settimanale"

    fig.suptitle("Istogrammi della distribuzione dei massimi delle curve sintetiche su base {}".format(base))

    ax[0,0].axvline(np.abs(p1)**2, color = colori2[0], linestyle = '--',linewidth = 3, label = "Picco originale" )
    n, bis, p = ax[0,0].hist(np.abs(picchi1)**2, bins = n_bins, edgecolor = "black", linewidth = 0.2, label = "fonte 1" , density = True)
    fracs = n / n.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, p):
        color = plt.cm.cool_r(norm(thisfrac))
        thispatch.set_facecolor(color)

    ax[0,1].axvline(np.abs(p2)**2, color = colori2[1], linestyle = '--',linewidth = 3, label = "Picco originale" )
    n, bis, p = ax[0,1].hist(np.abs(picchi2)**2, bins = n_bins, edgecolor = "black", linewidth = 0.1, label = "fonte 2", density = True)
    fracs = n / n.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, p):
        color = plt.cm.autumn(norm(thisfrac))
        thispatch.set_facecolor(color)

    ax[1,1].axvline(np.abs(p3)**2, color = colori2[2], linestyle = '--',linewidth = 3, label = "Picco originale" )
    n, bis, p = ax[1,1].hist(np.abs(picchi3)**2, bins = n_bins, edgecolor = "black", linewidth = 0.2, label = "fonte 3", density = True)
    fracs = n / n.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, p):
        color = plt.cm.summer(norm(thisfrac))
        thispatch.set_facecolor(color)

    ax[1,0].axvline(np.abs(p4)**2, color = colori2[3], linestyle = '--',linewidth = 3, label = "Picco originale" )
    n, bis, p = ax[1,0].hist(np.abs(picchi4)**2, bins = n_bins, edgecolor = "black", linewidth = 0.2, label = "fonte 4", density = True)
    fracs = n / n.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, p):
        color = plt.cm.spring(norm(thisfrac))
        thispatch.set_facecolor(color)
    
    ax[0,0].set_ylabel(r"Densità di Probabilità", fontsize=15)
    ax[0,1].set_ylabel(r"Densità di Probabilità", fontsize=15) 
    ax[1,1].set_ylabel(r"Densità di Probabilità", fontsize=15)
    ax[1,0].set_ylabel(r"Densità di Probabilità", fontsize=15)

    ax[0,0].set_xlabel(r'$|C_k|^2$', fontsize=15)
    ax[0,1].set_xlabel(r'$|C_k|^2$', fontsize=15)
    ax[1,1].set_xlabel(r'$|C_k|^2$', fontsize=15)
    ax[1,0].set_xlabel(r'$|C_k|^2$', fontsize=15)
    ax[0,0].legend()
    ax[0,1].legend(loc = 'upper center')
    ax[1,1].legend()
    ax[1,0].legend()

    plt.show()

