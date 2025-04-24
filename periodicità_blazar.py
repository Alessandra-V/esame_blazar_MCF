#####################################################################
# Alessandra Valenti                                                #
#                                                                   #
# Università degli Studi di Perugia                                 #
# Progetto per il corso di Metodi Computazionali per la Fisica      #
#-------------------------------------------------------------------#
#    Studio della Periodicità di Blazar                             #
#                                                                   #
#####################################################################

import modulo_funzioni_blazar as fbl
import modulo_funzioni_plot_blazar  as blplt
import argparse
import sys, os
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import colorsys
import math
from matplotlib import colors
from matplotlib.colors import Normalize
from scipy.stats import mode
from scipy import integrate
from scipy import  fft, optimize
from datetime import datetime, timedelta


#############################################
# Funzione per la gestione delle opzioni    #
#############################################

def parse_arguments():
    parser = argparse.ArgumentParser(description='Studio della periodicità dei Blazar', usage = 'python3 periodicità_blazar.py --option')
    
    parser.add_argument('-a', '--plotlc', action='store_true', help='Realizza il plot delle curve di luce delle fonti ')
    parser.add_argument('-b', '--pwsp'  , action='store_true', help='Realizza il plot degli spettri di potenza delle fonti')
    parser.add_argument('-c', '--fit'   , action='store_true', help='Realizza il plot del fit delle curve di luce e stampa una tabella con i parametri ottenuti')
    parser.add_argument('-d', '--period', action='store_true', help='Effettua lo studio della periodicità delle fonti e stampa una tabella con i relativi dati')
    parser.add_argument('-e', '--sint'  , action='store_true',
                        help='Realizza il plot degli istogrammi della distribuzione delle potenze delle curve sintetiche e restituisce la significatività ')

    return   parser.parse_args(args=None if sys.argv[1:] else ['--help'])


def main():

    args = parse_arguments()

                   ##########################################
                   #    Import dei dati e prima analisi     #
                   ##########################################
                   
    #import dei dati da CSV

    data1M = pd.read_csv('4FGL_J1229.0+0202_monthly_12_23_2024.csv')
    data2M = pd.read_csv('4FGL_J1555.7+1111_monthly_12_23_2024.csv')
    data3M = pd.read_csv('4FGL_J2202.7+4216_monthly_12_23_2024.csv')
    data4M = pd.read_csv('4FGL_J2253.9+1609_monthly_12_23_2024.csv')


    data1W = pd.read_csv('4FGL_J1229.0+0202_weekly_12_23_2024.csv')
    data2W = pd.read_csv('4FGL_J1555.7+1111_weekly_12_23_2024.csv')
    data3W = pd.read_csv('4FGL_J2202.7+4216_weekly_12_23_2024.csv')
    data4W = pd.read_csv('4FGL_J2253.9+1609_weekly_12_23_2024.csv')


    #creazione dizionari delle fonti

    data1M_diz = fbl.crea_dizionario_fonte(data1M, "3C 273 (FSRQ)")
    data2M_diz = fbl.crea_dizionario_fonte(data2M, "PG 1553 + 113 (BL Lac)")
    data3M_diz = fbl.crea_dizionario_fonte(data3M, "BL Lacertae (BL Lac)")
    data4M_diz = fbl.crea_dizionario_fonte(data4M, "3C 454.3 (FSRQ)" )

    data1W_diz = fbl.crea_dizionario_fonte(data1W, "3C 273 (FSRQ)")
    data2W_diz = fbl.crea_dizionario_fonte(data2W, "PG 1553 + 113 (BL Lac)")
    data3W_diz = fbl.crea_dizionario_fonte(data3W, "BL Lacertae (BL Lac)")
    data4W_diz = fbl.crea_dizionario_fonte(data4W, "3C 454.3 (FSRQ)")


    #estrazione degli upper limit dalle curve di luce

    fbl.agg_upper_limit(data1M_diz)
    fbl.agg_upper_limit(data2M_diz)
    fbl.agg_upper_limit(data3M_diz)
    fbl.agg_upper_limit(data4M_diz)

    fbl.agg_upper_limit(data1W_diz)
    fbl.agg_upper_limit(data2W_diz)
    fbl.agg_upper_limit(data3W_diz)
    fbl.agg_upper_limit(data4W_diz)

    #conversione da array di stringhe ad array di float per i flussi e gli errori

    fbl.converti_to_float(data1M_diz)
    fbl.converti_to_float(data2M_diz)
    fbl.converti_to_float(data3M_diz)
    fbl.converti_to_float(data4M_diz)

    fbl.converti_to_float(data1W_diz)
    fbl.converti_to_float(data2W_diz)
    fbl.converti_to_float(data3W_diz)
    fbl.converti_to_float(data4W_diz)


    #conversione da MET a data

    fbl.MET_to_data_diz(data1M_diz)
    fbl.MET_to_data_diz(data2M_diz)
    fbl.MET_to_data_diz(data3M_diz)
    fbl.MET_to_data_diz(data4M_diz)

    fbl.MET_to_data_diz(data1W_diz)
    fbl.MET_to_data_diz(data2W_diz)
    fbl.MET_to_data_diz(data3W_diz)
    fbl.MET_to_data_diz(data4W_diz)



                 ###############################################
                 #    Plots Flusso di Fotoni vs Tempo (data)   #
                 ###############################################

    c_grafici   = ['rebeccapurple', 'firebrick', 'darkorange', 'deeppink' ]
    c_secondari = ['forestgreen',"lightseagreen", "darkmagenta",  "darkslateblue"]


    #plot raggruppati per base temporale (M/W)
    if args.plotlc == True: 
        blplt.plot_all(data1M_diz, data2M_diz, data3M_diz, data4M_diz, "M", c_grafici, c_secondari)
        blplt.plot_all(data1W_diz, data2W_diz, data3W_diz, data4W_diz, "W", c_grafici, c_secondari)
        sys.exit()





              ###############################################
              #    Analisi di Fourier delle curve di Luce   #
              ###############################################

    #interpolazione dei dati 

    fbl.interpolazione(data1M_diz)
    fbl.interpolazione(data2M_diz)
    fbl.interpolazione(data3M_diz)
    fbl.interpolazione(data4M_diz)

    fbl.interpolazione(data1W_diz)
    fbl.interpolazione(data2W_diz)
    fbl.interpolazione(data3W_diz)
    fbl.interpolazione(data4W_diz)


    #calcolo della trasformata di Fourier per dati originali

    fbl.fft_diz(data1M_diz)
    fbl.fft_diz(data2M_diz)
    fbl.fft_diz(data3M_diz)
    fbl.fft_diz(data4M_diz)

    fbl.fft_diz(data1W_diz)
    fbl.fft_diz(data2W_diz)
    fbl.fft_diz(data3W_diz)
    fbl.fft_diz(data4W_diz)


    #calcolo della trasformata di Fourier per flussi e tempi completi

    fbl.fft_diz(data1M_diz, interp = True)
    fbl.fft_diz(data2M_diz, interp = True)
    fbl.fft_diz(data3M_diz, interp = True)
    fbl.fft_diz(data4M_diz, interp = True)

    fbl.fft_diz(data1W_diz, interp = True)
    fbl.fft_diz(data2W_diz, interp = True)
    fbl.fft_diz(data3W_diz, interp = True)
    fbl.fft_diz(data4W_diz, interp = True)



               ############################################
               #   Spettri di Potenza delle curve di Luce #
               ############################################

    #plot degli spettri di potenza su base mensile e settimanale:

    if args.pwsp == True:
        blplt.plot_all_pwsp(data1M_diz, data2M_diz, data3M_diz, data4M_diz, "M", c_grafici, log = True, interp = True)
        blplt.plot_all_pwsp(data1W_diz, data2W_diz, data3W_diz, data4W_diz, "W", c_grafici, log = True, interp = True)
        sys.exit()



                      ##############################
                      #  Fit con rumore            #
                      ##############################


    #initial guesses per i fit:

    p0_1M = [1e-9, 0.9]
    p0_2M = [1e-11,1.2]
    p0_3M = [1e-8, 1.3]
    p0_4M = [1e-9, 0.7]

    p0_1W = [1e-8, 0.9]
    p0_2W = [1e-10,1]
    p0_3W = [1e-7, 1.3]
    p0_4W = [1e-6, 0.8]

    #fit:
    fbl.fit_pwsp(data1M_diz, fbl.fit, p0_1M, interp = True)
    fbl.fit_pwsp(data2M_diz, fbl.fit, p0_1M, interp = True)
    fbl.fit_pwsp(data3M_diz, fbl.fit, p0_1M, interp = True)
    fbl.fit_pwsp(data4M_diz, fbl.fit, p0_1M, interp = True)

    fbl.fit_pwsp(data1W_diz, fbl.fit, p0_1M, interp = True)
    fbl.fit_pwsp(data2W_diz, fbl.fit, p0_1M, interp = True)
    fbl.fit_pwsp(data3W_diz, fbl.fit, p0_1M, interp = True)
    fbl.fit_pwsp(data4W_diz, fbl.fit, p0_1M, interp = True)



    if args.fit == True:
        
        print("\033[95m     Tabella dei parametri ricavati dal Fit   \033[0m")
        print("")
        print("      Sorgente | Parametro  | Valore Parametro | Errore")
        print("     ----------|------------|------------------|----------------")
        print("      1M       |     N      | {:.3f}            |+- {:.3e} ".format(data1M_diz["params fit"][0], math.sqrt(data1M_diz["params covariance fit"][0,0])))
        print("      1M       |    Beta    | {:.3f}            |+- {:.3f} ".format(data1M_diz["params fit"][1], math.sqrt(data1M_diz["params covariance fit"][1,1])))
        print("     ----------|------------|------------------|----------------")
        print("      2M       |     N      | {:.3f}            |+- {:.3e} ".format(data2M_diz["params fit"][0], math.sqrt(data2M_diz["params covariance fit"][0,0])))
        print("      2M       |    Beta    | {:.3f}            |+- {:.3f} ".format(data2M_diz["params fit"][1], math.sqrt(data2M_diz["params covariance fit"][1,1])))
        print("     ----------|------------|------------------|---------------")
        print("      3M       |     N      | {:.3f}            |+- {:.3e} ".format(data3M_diz["params fit"][0], math.sqrt(data3M_diz["params covariance fit"][0,0])))
        print("      3M       |    Beta    | {:.3f}            |+- {:.3f} ".format(data3M_diz["params fit"][1], math.sqrt(data3M_diz["params covariance fit"][1,1])))
        print("     ----------|------------|------------------|-----------------")
        print("      4M       |     N      | {:.3f}            |+- {:.3e} ".format(data4M_diz["params fit"][0], math.sqrt(data4M_diz["params covariance fit"][0,0])))
        print("      4M       |    Beta    | {:.3f}            |+- {:.3f} ".format(data4M_diz["params fit"][1], math.sqrt(data4M_diz["params covariance fit"][1,1])))
        print("     ----------|------------|------------------|----------------")
        print("      1W       |     N      | {:.3f}            |+- {:.3e} ".format(data1W_diz["params fit"][0], math.sqrt(data1W_diz["params covariance fit"][0,0])))
        print("      1W       |    Beta    | {:.3f}            |+- {:.3f} ".format(data1W_diz["params fit"][1], math.sqrt(data1W_diz["params covariance fit"][1,1])))
        print("     ----------|------------|------------------|----------------")
        print("      2W       |     N      | {:.3f}            |+- {:.3e} ".format(data2W_diz["params fit"][0], math.sqrt(data2W_diz["params covariance fit"][0,0])))
        print("      2W       |    Beta    | {:.3f}            |+- {:.3f} ".format(data2W_diz["params fit"][1], math.sqrt(data2W_diz["params covariance fit"][1,1])))
        print("     ----------|------------|------------------|---------------")
        print("      3W       |     N      | {:.3f}            |+- {:.3e} ".format(data3W_diz["params fit"][0], math.sqrt(data3W_diz["params covariance fit"][0,0])))
        print("      3W       |    Beta    | {:.3f}            |+- {:.3f} ".format(data3W_diz["params fit"][1], math.sqrt(data3W_diz["params covariance fit"][1,1])))
        print("     ----------|------------|------------------|-----------------")
        print("      4W       |     N      | {:.3f}            |+- {:.3e} ".format(data4W_diz["params fit"][0], math.sqrt(data4W_diz["params covariance fit"][0,0])))
        print("      4W       |    Beta    | {:.3f}            |+- {:.3f} ".format(data4W_diz["params fit"][1], math.sqrt(data4W_diz["params covariance fit"][1,1])))


        # plot dei dati + fit:

        blplt.plot_all_pwsp_fit(data1M_diz,data2M_diz, data3M_diz, data4M_diz, "M", c_grafici, c_secondari, interp = True)
        blplt.plot_all_pwsp_fit(data1W_diz,data2W_diz, data3W_diz, data4W_diz, "W", c_grafici, c_secondari, interp = True)
        sys.exit()




                         #############################
                         #   Periodi e Periodicità   # 
                         #############################

    #ricerca dei picchi associati al periodo:

    frequenza_taglio = 1e-8

    periodo_1M = fbl.picco_periodo(data1M_diz , frequenza_taglio, interp = True)
    periodo_2M = fbl.picco_periodo(data2M_diz , frequenza_taglio, interp = True)
    periodo_3M = fbl.picco_periodo(data3M_diz , frequenza_taglio, interp = True)
    periodo_4M = fbl.picco_periodo(data4M_diz , frequenza_taglio, interp = True)

    periodo_1W = fbl.picco_periodo(data1W_diz , frequenza_taglio, interp = True)
    periodo_2W = fbl.picco_periodo(data2W_diz , frequenza_taglio, interp = True)
    periodo_3W = fbl.picco_periodo(data3W_diz , frequenza_taglio, interp = True)
    periodo_4W = fbl.picco_periodo(data4W_diz , frequenza_taglio, interp = True)


    if args.period == True:
        
        print("\033[95m     Tabella delle frequenze e periodi individuati nelle fonti   \033[0m")
        print("")
        print(" Fonte e base temporale   | frequenza del picco [Hz] | potenza associata [u.a.] |    periodo[gg]  "  )
        print("--------------------------|--------------------------|--------------------------|-----------------")
        print(" Fonte 1, Mensile         |{:.3e}                 |{:.3e}                 |{:.2f}  ".format(periodo_1M[0], np.abs(periodo_1M[1])**2, 1/(periodo_1M[0]*86400) ) )
        print(" Fonte 1, Settimanale     |{:.3e}                 |{:.3e}                 |{:.2f}  ".format(periodo_1W[0], np.abs(periodo_1W[1])**2, 1/(periodo_1W[0]*86400)) )
        print("--------------------------|--------------------------|--------------------------|-----------------")
        print(" Fonte 2, Mensile         |{:.3e}                 |{:.3e}                 |{:.2f}  ".format(periodo_2M[0], np.abs(periodo_2M[1])**2, 1/(periodo_2M[0]*86400)) )
        print(" Fonte 2, Settimanale     |{:.3e}                 |{:.3e}                 |{:.2f}  ".format(periodo_2W[0], np.abs(periodo_2W[1])**2, 1/(periodo_2W[0]*86400)) )
        print("--------------------------|--------------------------|--------------------------|-----------------")
        print(" Fonte 3, Mensile         |{:.3e}                 |{:.3e}                 |{:.2f}  ".format(periodo_3M[0], np.abs(periodo_3M[1])**2, 1/(periodo_3M[0]*86400)) )
        print(" Fonte 3, Settimanale     |{:.3e}                 |{:.3e}                 |{:.2f}  ".format(periodo_3W[0], np.abs(periodo_3W[1])**2, 1/(periodo_3W[0]*86400)) )
        print("--------------------------|--------------------------|--------------------------|-----------------")
        print(" Fonte 4, Mensile         |{:.3e}                 |{:.3e}                 |{:.2f}  ".format(periodo_4M[0], np.abs(periodo_4M[1])**2, 1/(periodo_4M[0]*86400)) )
        print(" Fonte 4, Settimanale     |{:.3e}                 |{:.3e}                 |{:.2f}  ".format(periodo_4W[0], np.abs(periodo_4W[1])**2, 1/(periodo_4W[0]*86400)) )
        sys.exit()




                              ################################
                              #   Curve di Luce Sintetiche   #
                              ################################


    N = 10000

    #Generazione delle curve sintetiche: 

    curve_sint_1M = fbl.curve_sintetiche_diz(data1M_diz, N)
    curve_sint_2M = fbl.curve_sintetiche_diz(data2M_diz, N)
    curve_sint_3M = fbl.curve_sintetiche_diz(data3M_diz, N)
    curve_sint_4M = fbl.curve_sintetiche_diz(data4M_diz, N)



    curve_sint_1W = fbl.curve_sintetiche_diz(data1W_diz, N)
    curve_sint_2W = fbl.curve_sintetiche_diz(data2W_diz, N)
    curve_sint_3W = fbl.curve_sintetiche_diz(data3W_diz, N)
    curve_sint_4W = fbl.curve_sintetiche_diz(data4W_diz, N)



    #Trasformata di Fourier delle curve sintetiche



    fft_sint_1M = fbl.fft_curve_sintetiche_diz(curve_sint_1M)
    fft_sint_2M = fbl.fft_curve_sintetiche_diz(curve_sint_2M)
    fft_sint_3M = fbl.fft_curve_sintetiche_diz(curve_sint_3M)
    fft_sint_4M = fbl.fft_curve_sintetiche_diz(curve_sint_4M)



    fft_sint_1W = fbl.fft_curve_sintetiche_diz(curve_sint_1W)
    fft_sint_2W = fbl.fft_curve_sintetiche_diz(curve_sint_2W)
    fft_sint_3W = fbl.fft_curve_sintetiche_diz(curve_sint_3W)
    fft_sint_4W = fbl.fft_curve_sintetiche_diz(curve_sint_4W)



                                    #######################
                                    #   Significatività   #
                                    #######################

    # Invidivuazione del picco di periodo per curve sintetiche:

    picchi_sint_1M = fbl.ar_picchi_sintetici(fft_sint_1M, frequenza_taglio)
    picchi_sint_2M = fbl.ar_picchi_sintetici(fft_sint_2M, frequenza_taglio)
    picchi_sint_3M = fbl.ar_picchi_sintetici(fft_sint_3M, frequenza_taglio)
    picchi_sint_4M = fbl.ar_picchi_sintetici(fft_sint_4M, frequenza_taglio)

    picchi_sint_1W = fbl.ar_picchi_sintetici(fft_sint_1W, frequenza_taglio)
    picchi_sint_2W = fbl.ar_picchi_sintetici(fft_sint_2W, frequenza_taglio)
    picchi_sint_3W = fbl.ar_picchi_sintetici(fft_sint_3W, frequenza_taglio)
    picchi_sint_4W = fbl.ar_picchi_sintetici(fft_sint_4W, frequenza_taglio)


    #realizzazione degli istogrammi della distribuzione dei massimi delle trasformate delle curve sintetiche

    n_bins = 100


    #calcolo dell'area (valore-p )

    pval_1M = fbl.significatività_int(picchi_sint_1M, periodo_1M[1], n_bins)
    pval_2M = fbl.significatività_int(picchi_sint_2M, periodo_2M[1], n_bins)
    pval_3M = fbl.significatività_int(picchi_sint_3M, periodo_3M[1], n_bins)
    pval_4M = fbl.significatività_int(picchi_sint_4M, periodo_4M[1], n_bins)

    pval_1W = fbl.significatività_int(picchi_sint_1W, periodo_1W[1], n_bins)
    pval_2W = fbl.significatività_int(picchi_sint_2W, periodo_2W[1], n_bins)
    pval_3W = fbl.significatività_int(picchi_sint_3W, periodo_3W[1], n_bins)
    pval_4W = fbl.significatività_int(picchi_sint_4W, periodo_4W[1], n_bins)

    if args.sint == True:

        print("\033[95m  \t                     Tabella della Significatività dei periodi delle Fonti  \033[0m")
        print(" ")
        print("\033[4m       Nome Fonte       | Base Temporale | Periodo[gg]  |   p-value  | Significatività [%]  \033[0m")
       
        print(" {:<25} Mensile         {:.2f}\t   {:.5f}\t     {:.2f}%  \u00B10.01%      ".format(data1M_diz["nome"],1/(periodo_1M[0]*86400),pval_1M,(1-pval_1M)*100))
        print(" {:<25} Mensile         {:.2f}\t  <{:.5f}\t    >{:.2f}%  \u00B10.01%      ".format(data2M_diz["nome"],1/(periodo_2M[0]*86400),pval_2M,(1-pval_2M)*100))
        print(" {:<25} Mensile         {:.2f}\t   {:.5f}\t     {:.2f}%  \u00B10.01%      ".format(data3M_diz["nome"],1/(periodo_3M[0]*86400),pval_3M,(1-pval_3M)*100))
        print(" {:<25} Mensile         {:.2f}\t   {:.5f}\t     {:.2f}%  \u00B10.01%      ".format(data4M_diz["nome"],1/(periodo_4M[0]*86400),pval_4M,(1-pval_4M)*100))
        print("-----------------------------------------------------------------------------------------------------")
        print(" {:<25} Settimanale     {:.2f}\t   {:.5f}\t     {:.2f}%  \u00B10.01%      ".format(data1W_diz["nome"],1/(periodo_1W[0]*86400),pval_1W,(1-pval_1W)*100))
        print(" {:<25} Settimanale     {:.2f}\t  <{:.5f}\t    >{:.2f}%  \u00B10.01%      ".format(data2W_diz["nome"],1/(periodo_2W[0]*86400),pval_2W,(1-pval_2W)*100))
        print(" {:<25} Settimanale     {:.2f}\t  <{:.5f}\t    >{:.2f}%  \u00B10.11%      ".format(data3W_diz["nome"],1/(periodo_3W[0]*86400),pval_3W,(1-pval_3W)*100))
        print(" {:<25} Settimanale     {:.2f}\t  <{:.5f}\t    >{:.2f}%  \u00B10.01%      ".format(data4W_diz["nome"],1/(periodo_4W[0]*86400),pval_4W,(1-pval_4W)*100))


        blplt.plot_all_hist(picchi_sint_1M, picchi_sint_2M, picchi_sint_3M, picchi_sint_4M, periodo_1M[1], periodo_2M[1], periodo_3M[1], periodo_4M[1], "M", c_secondari, n_bins)
        blplt.plot_all_hist(picchi_sint_1W, picchi_sint_2W, picchi_sint_3W, picchi_sint_4W, periodo_1W[1], periodo_2W[1], periodo_3W[1], periodo_4W[1], "W", c_secondari, n_bins)



if __name__ == "__main__":

    main()
