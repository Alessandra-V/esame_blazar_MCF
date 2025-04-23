"""
Modulo funzioni per analisi della periodicità dei Blazar

Autore: Valenti Alessandra


Nel seguente modulo sono contenute tutte le funzioni utilizzate per lo studio della periodicità dei blazar

Elenco delle funzioni contenute per categoria di utilizzo:

1) Analisi preliminare dei dati:
     - crea_dizionario_fonte............. r.58             
     - flusso_to_float................... r.98                        
     - flusso_err_to_float............... r.120             
     - trova_upper_limit................. r.144                
     - agg_upper_limit................... r.182                 
     - converti_to_float................. r.217                   
     - MET_to_data_array................. r.243         
     - MET_to_data_diz................... r.268           
     
2) Analisi di Fourier delle curve di luce                      
     - dt_control_bool.................. r.294          
     - dt_medio......................... r.321                  
     - dt_moda ......................... r.350                         
     - interpolazione................... r.373                       
     - fft_diz.......................... r.447                          
             
3) Fit dei dati
    - fit    .......................... r. 494                                                              
    - fit_pwsp ........................ r. 514                

4) Periodicità
    - picco_periodo ................... r. 562            

5) Curve sintetiche e significatività
    - curve_sintetiche_diz .............. r.612                
    - fft_curve_sintetiche_diz........... r.650 
    - picco_periodo_sint................. r.702    
    - ar_picchi_sintetici................ r.742   
    - significatività_int................ r.776            

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc
import math
from scipy import  fft, optimize
from scipy.stats import mode
from datetime import datetime, timedelta
import matplotlib.dates as mdates


                                      ###########################################
                                      #     Analisi preliminare dei dati        #
                                      ###########################################

def crea_dizionario_fonte(df, nome_fonte):
    """
    Funzione che crea un dizionario contenente tutti i dati una fonte
    Parametri:
    ---------------
    df  (dataframe pandas) :  contenente i dati da analizzare, il dataframe deve avere le seguenti colonne:
                             ['MET'] , ['Photon Flux [0.1-100 GeV](photons cm-2 s-1)'] , ['Photon Flux Error(photons cm-2 s-1)']
    nome_fonte (string)    : nome associato alla fonte 
    

    Restituisce:
    ----------------
    Dizionario contenente i dati della fonte.
    Il dizionario ha la seguente struttura:

    fonte = {
         "nome"      : ... ,
         "flusso"    : ... ,
         "flusso_err : ... ,
         "tempo"     :
    
    }
                    
    Note:
    ------------
    La funzione, oltre a creare il dizionario, converte le colonne dei dataframe in array di numpy con la funzione  .to_numpy()

    """
    diz_fonte = {
        "nome"       : nome_fonte, 
        "flusso"     : df['Photon Flux [0.1-100 GeV](photons cm-2 s-1)'].to_numpy(),
        "flusso_err" : df['Photon Flux Error(photons cm-2 s-1)'].to_numpy(),
        "tempo"      : df['MET'].to_numpy()
    }
   

    return diz_fonte

#-----------------------------------------------------------------------------------------------------------------------

def flusso_to_float(ar):
    """
    Funzione che si occupa della conversione dei dati da stringhe a float, con la conversione del carattere
    "<", che indica gli upper limit del flusso, in uno spazio vuoto "".

    Parametri:
    -------------
    ar  (array) :  Array contenente i dati di tipo (string)

    Restituisce:
    -------------
    array :  Array contenente dati di tipo (float)
    
    """

    for i in range (0, len(ar)):
        ar[i] = ar[i].replace("<", "")

    return ar.astype(float)
#------------------------------------------------------------------------------------------------------------------------


def flusso_err_to_float(ar):
    """
    Funzione che si occupa della conversione dei dati da stringhe a float, con la conversione del carattere
    "-", che indica gli upper limit dell'errore del flusso , in uno spazio vuoto "".

    Parametri:
    -------------
    ar (array):  Array contenente i dati di tipo (string)

    Restituisce:
    -------------
    array :  Array contenente dati di tipo (float)
 

    """

    for i in range (0, len(ar)):
        if ar[i].startswith("-"):
            ar[i] = "NaN"

    return ar.astype(float)

#-------------------------------------------------------------------------------------------------------------------------

def trova_upper_limit(flusso, tempo):
    """
    Funzione che individua gli upper limit del flusso.

    Parametri:
    --------------
    flusso (array):  di (string) contenente i dati del flusso
    tempo  (array):  di (float)  contenente i dati temporali

    Restituisce:
    -------------
    upper_limits_flusso (array): array di (float) che contiene i valori del flusso degli upper limit
    upper_limits_tempo (array) : array di (float) che contiene i valori del tempo degli upper limit


    Note:
    -----------
    -  utilizza la funzione flusso_to_float() definita in questo modulo per convertire l'array dei flussi in dati di tipo (float)
    -  nel caso in cui non siano presenti upper limit, restituisce un array vuoto.
       La condizione imposta per questa casistica si basa sul fatto che l'array che contiene i dati del flusso
       non è un array di stringhe, ma di float (o meglio, è un array di tipo object)
    """
    upper_limits_flusso = np.empty(0)
    upper_limits_tempo  = np.empty(0)

    for i in range(0, len(flusso)):
                
        if flusso[i].startswith("<"):
            
            upper_limits_flusso = np.append(upper_limits_flusso, flusso[i])
            upper_limits_tempo  = np.append(upper_limits_tempo, tempo[i])

    upper_limits_flusso = flusso_to_float(upper_limits_flusso)

    return upper_limits_flusso, upper_limits_tempo

#------------------------------------------------------------------------------------------------------------------------------------

def agg_upper_limit(diz):
    """
    Funzione che individua gli upper limit dei dati delle fonti e li aggiunge come chiavi al dizionario della fonte
    In particolare, trova i valori del flusso corrispondenti agli upper limit e i relativi dati temporali

    Parametri:
    -------------
    diz (dictionary): dizionario contenente i dati della fonte

    Restituisce:
    --------------
    dizionario modificato con i dati degli upper limit

    Note:
    --------------
    Utilizza la funzione trova_upper_limit() definita nel modulo
    ATTENZIONE! dato che la funzione upper limit sfrutta il fatto che i dati siano di tipo stringa è obbligatorio trovare gli upper limits
    prima di convertire le colonne del flusso, errore e tempo in float
    """

    flusso = diz["flusso"]
    tempo  = diz["tempo"]
  
    if type(flusso[0]) == str:        

        diz['upper_lim_flusso'] = trova_upper_limit(flusso,tempo)[0]
        diz['upper_lim_tempo']  = trova_upper_limit(flusso,tempo)[1]

    else:
        diz['upper_lim_flusso'] = np.empty(0)
        diz['upper_lim_tempo']  = np.empty(0)


#--------------------------------------------------------------------------------------------------------------

def converti_to_float(diz):
    """
    Funzione che converte i dati del flusso e del relativo errore in float

    Parametri:
    -------------
    diz (dictionary) : contenente le informazioni di flusso e errore

    Restituisce:
    ---------------
     dizionario con le chiavi di flusso e errore modificate

    Note:
    --------
    - la funzione controlla se i dati di flusso e errore siano effettivamente in formato stringa prima di convertire
    
    """

    if type(diz['flusso'][0]) == str:        

        diz['flusso'] =      flusso_to_float(diz['flusso'])
        diz['flusso_err']  = flusso_err_to_float(diz['flusso_err'])


#----------------------------------------------------------------------------------------------------------

def MET_to_data_array(array):
    """
    Funzione che dato un array di dati temporali espresso in Mission Elapsed Time (MET)
    lo converte in date con formato dd/mm/yy

    Parametri:
    ------------
    array (array): contente i dati temporali

    Restituisce:
    --------------
    date_array (array) : contente i dati temporali nel formato dd/yy/mm
    """
    
    start_time = datetime(2001, 1, 1, 0, 0, 0)
    date_array = np.empty(0)
    
    for i in range(0,len(array)):
        date_i =  start_time + timedelta(seconds=int(array[i]))
        date_array = np.append(date_array, date_i)
        
    return date_array

#-------------------------------------------------------------------------------------

def MET_to_data_diz(diz):
    """
    Funzione che aggiunge al dizionario delle fonti, la chiave contenente il tempo indicato con
    una data del tipo dd/mm/yy

    Parametri:
    -------------
    diz (dictionary): dizionario contenente i dati

    Restituisce:
    -------------
    dizionario con le chiavi del tempo e degli upper limit nel formato di data
    
    """

    diz["tempo_data"] = MET_to_data_array(diz["tempo"])
    diz["upper_lim_data"] = MET_to_data_array(diz['upper_lim_tempo'])



                       #######################################
                       #     Analisi di Fourier dei dati     #
                       #######################################

#-------------------------------------------------------------------

def dt_control_bool(tempo):
    """
    Funzione di controllo per verificare che gli intervalli temporali siano costanti
    

    Parametri:
    -----------
    tempo (array):  array che contiene i dati temporali (float) della curve di luce

    Restituisce :
    ------------
    (bool)  : True se gli intervalli temporali sono tutti uguali tra loro , altrimenti False

    Note:
    ---------
    La funzione è stata utilizzata unicamente nelle fasi iniziali del progetto
    
    """
    
    dt = np.diff(tempo)

    return np.all(dt == dt[0])


#-------------------------------------------------------------------------------------------------


def dt_medio(tempo):
    """
    Calcola l'intervallo temporale medio tra un elemento e un altro

    Parametri:
    --------------------
    tempo (array)   : contenente i timestamp temporali

    Restituisce:
    ----------------
    dt_med (float) : intervallo temporale medio

    Note:
    ---------
    La funzione è stata utilizzata unicamente nelle fasi iniziali del progetto
    
    """
    dt_arr = np.diff(tempo)
    
    som = 0

    for i in range(0, len( dt_arr)):
        som = som +  dt_arr[i]

    dt_med = som / len(dt_arr)
    return dt_med

#--------------------------------------------------------------------------------------------

def dt_moda(tempo):
    """
    Dato un array di dati temporali restituisce il valore dell'intervallo temporale più presente

    Parametri:
    ----------------
    tempo    : contenente i timestamp temporali

    Restituisce:
    ----------------
    dt (float) : intervallo temporale maggioritario

    """
    dt_arr = np.diff(tempo)
    
    valori_unici, frequenze = np.unique(dt_arr, return_counts=True)
    most_common = valori_unici[np.argmax(frequenze)]

    return most_common


#----------------------------------------------------------------------------------------------------------

def interpolazione(diz):
    """
    Funzione che effettua l'interpolazione dei dati di flusso e tempo

    Parametri:
    ----------------
    diz (dictionary) : contenente almeno le chiavi di ["flusso"] e ["tempo"] associate ai dati

    Restituisce:
    -------------
    dizionario aggiornato con le chiavi del flusso e del tempo interpolati

    Note:
    ------------
    - utilizza la funzione dt_moda() definita in questo modulo
                  
    """
    flusso = diz["flusso"]
    tempo  = diz["tempo"]

    ar_dt = np.diff(tempo)
    dt_ok = dt_moda(tempo)

    flussi_completi = flusso.copy()

    indici_flussi_mancanti = np.empty(0)
    valori_flussi_mancanti = np.empty(0)
 

    for i in range(0, len(ar_dt)):
        
        if ar_dt[i] != dt_ok :
           
            n_mancanti = int(( ar_dt[i] / dt_ok) -1)
            t_1 = tempo[i - 1]

            if n_mancanti == 1:
                
                t_buco = t_1 +  dt_ok
                y = np.interp(t_buco, tempo, flusso)
                
                valori_flussi_mancanti = np.append(valori_flussi_mancanti, y)
                indici_flussi_mancanti = np.append(indici_flussi_mancanti, i)

  
            if n_mancanti > 1:
                timestamp_mancanti = np.empty(0)
                indici_mancanti_float = np.empty(0)

                for j in range(0, n_mancanti):
                    
                    t_buco = t_1 + (j + 1) * dt_ok
                    
                    timestamp_mancanti = np.append(timestamp_mancanti, t_buco)
                    indici_mancanti_float = np.append(indici_mancanti_float, i + 1)
                    indici_mancanti = indici_mancanti_float.astype(int)
                    

                y_ar = np.interp(timestamp_mancanti, tempo, flusso)
                
                valori_flussi_mancanti = np.append(valori_flussi_mancanti, y_ar)
                indici_flussi_mancanti = np.append(indici_flussi_mancanti, indici_mancanti)

    tempi_completi = np.arange(tempo[0] , tempo[-1] + dt_ok , dt_ok)

    indici_flussi_mancanti = indici_flussi_mancanti.astype(int)
    flussi_completi = np.insert(flussi_completi, indici_flussi_mancanti, valori_flussi_mancanti, )

    diz["flussi completi"] = flussi_completi
    diz["tempi completi"]  = tempi_completi

    
#-----------------------------------------------------------------------------------------------------------

def fft_diz(diz, interp = False):
    """
    Funzione che effettua lo studio in frequenza delle curve di luce, calcola le frequenze e le potenze
    e le aggiunge come chiavi del dizionario

    Parametri:
    ----------------
    diz (dictionary)  : dizionario contenente almeno le informazioni di flusso e tempo della curva di luce
    interp (boolean)  : booleano che mi indica se i dati di cui voglio fare la trasformata sono interpolati o meno
                        se interp = True, allora faccio la trasformata dei dati interpolati

    Restituisce:
    --------------
    aggiorna il dizionario aggiungendo le chiavi della potenza e della frequenza con i relativi dati
    
    """
    
    if interp == False:
        flusso = diz["flusso"]
        tempo  = diz["tempo"]

        fft_a = fft.fft(flusso)
        diz["ck"] = fft_a

        dt = dt_moda(tempo)
        diz["frequenza"] = fft.fftfreq(len(fft_a), d = dt )
        
    if interp == True:
        flusso_interp = diz["flussi completi"]
        tempo_interp  = diz["tempi completi"]

        fft_interp = fft.fft(flusso_interp)
        diz["ck interp"] = fft_interp

        dt = np.diff(tempo_interp)[0]
        diz["frequenza interp"] = fft.fftfreq(len(fft_interp), d = dt)
           




                           ###############################
                           #        Fit dei dati         #
                           ###############################



def fit(x , N, beta) :
    """
    Funzione di Fit con normalizzazione

    Parametri:
    ------------
    x (float/array): variabile indipendente della funzione, in questo caso, il tempo (float)
    N (float)      : fattore di normalizzazione (float)
    beta           : esponente della funzione (float)


    Restituisce:
    -------------
    N x 1/x^beta (float/array) : valore della funzione calcolata in base ai dati in input  
    
    """
    return N*(1/x**(beta))

#-------------------------------------------------------------------

def fit_pwsp(diz, fit_func, p0_guess, interp = False):
    """
    Funzione che effettua il fit con una funzione definita sui dati dell'analisi in frequenza
    delle fonti

    Parametri:
    ----------
    diz      (dictionary)    : contenente almeno i dati dell'analisi in frequenza della fonte
    fit_func (function)      : funzione di fit 
    p0_guess (list)          : contenente le initial guesses per il fit
    interp   (boolean)       : variabile booleana che indica se il fit deve essere fatto sui dati interpolati o no
                                interp = False => fit sui dati NON interpolati
                                interp = True  => fit sui dati interpolati

    Restituisce:
    --------------
    - il dizionario viene aggiornato aggiungendo le chiavi:
        diz["params fit"]            : contente l'array con il valore ottimizzato dei parametri nell'ordine in cui sono definiti nella funzione soggetta al fit
        diz["params covariance fit"] : la matrice di covarianza dei parametri ottimizzati per cui la diagonale principale contiene la varianza dei parametri

    Note:
    -----------
    - utilizza la funzione optimize.curve_fit() di Scipy optimize
    - per il fit viene escluso il primo punto dei dati a disposizione
    """
    
    if interp == False:
        freq = diz["frequenza"]
        pot  = diz["ck"]

    if interp == True:
        freq = diz["frequenza interp"]
        pot  = diz["ck interp"]

    params , params_covariance = optimize.curve_fit(fit_func, freq[1:len(freq)//2], np.abs(pot[1:len(pot)//2])**2, p0 = p0_guess, maxfev = 1200000)

    diz["params fit"] = params
    diz["params covariance fit"]= params_covariance

    diz["dati_fit"] = fit_func(freq[1:len(freq)//2],params[0], params[1] )


    

                           ##########################
                           #      Periodicità       #
                           ##########################

def picco_periodo(diz , f_taglio, interp = True, return_pot = True):
    """
    Funzione che individua il picco associato al periodo a partire dallo spettro di potenza di una fonte

    Parametri:
    -----------
    diz (dictionary)     : contenente almeno i dati relativi alla frequenza e potenza dello spettro
    f_taglio (float)     : valore in frequenza al di sotto della quale il contributo viene considerato costante
    interp (boolean)     : variabile booleana che indica se la ricerca del periodo deve essere fatta sui dati interpolati o no
                                interp = False =>  sui dati NON interpolati
                                interp = True  =>  sui dati interpolati

    Restituisce:
    ------------
    periodo (list)  : contenente la frequenza e la potenza del picco del periodo

    
    """
    if interp == False:
        freq = diz["frequenza"]
        pot  = diz["ck"]
    if interp == True:
        freq = diz["frequenza interp"]
        pot  = diz["ck interp"]       
    
    freq = freq[:len(freq)//2]
    pot  = pot[:len(pot)//2]

    mask = freq > f_taglio
    freq_tgl = freq[mask]

    n = len(freq) - len(freq_tgl)
    pot_tgl = pot[n:]

    pot_picco = np.max(pot_tgl)

    for i in range(0, len(pot_tgl)):
        if pot_tgl[i] == pot_picco:
            freq_picco = freq_tgl[i]
    
    periodo = [freq_picco, pot_picco]
    
    return periodo
   

                      #########################################
                      #  Curve sintetiche e Significatività   #
                      #########################################

                      
def curve_sintetiche_diz(diz, N):
    """
    Funzione per la creazione di curve sintetiche a partire dai dati di una curva di luce.
    Restituisce un dizionario contenente le curve sintetiche

    Parametri:
    -------------------
    diz (dictionary)   : contenente le informazioni di flusso e tempo della curva di luce
    N   (int)          : numero di curve sintetiche da generare

    Restituisce:
    ----------------------
    curve_sintetiche (dictionary) : contenente nella chiave ["tempo"] i dati temporali comuni a tutte le curve di luce,
                                    e nelle N chiavi ["flusso i"] con i in [1, N] i dati del flusso di ogni curva sintetica

    Note:
    --------------
    - utilizza la funzione np.random.shuffle() di numpy

    """

    flusso = diz["flussi completi"]
    tempo  = diz["tempi completi"]

    curve_sintetiche = {}
    curve_sintetiche["tempo"] = tempo

    for i in range(0 , N + 1):

        flusso_shuffle = flusso.copy()
        np.random.shuffle(flusso_shuffle)

        curve_sintetiche["flusso {}".format( i + 1)] = flusso_shuffle   

    return curve_sintetiche

#---------------------------------------------------------------------------------------------------------------------------------

def fft_curve_sintetiche_diz(diz):
    """
    Funzione che effettua lo studio in  frequenza delle curve sintetiche

    Parametri:
    ---------------------
    diz (dictionary)  : contenente le curve sintetiche generate con la funzione curve_sintetiche_diz()

    Restituisce:
    --------------------
    diz_fft (dictionary) : contenente  le chiavi [ck i] per i in [1,N] che contengono le potenze di ogni curva di luce sintetica
                           e le chiave ["freq"] con le frequenze associate al segnale (dato che i valori della frequenza dipendono
                           unicamente dal numero di dati associati alla trasformata di Fourier e dalla distanza temporale che sono uguali per tutte
                           le curve sintetiche, allora è sufficiente avere un'unica chiave per le frequenze di tutte le trasfromate delle curve sintetiche )

    Note:
    --------------------
    - utilizza la funzione dt_moda() definita in questo modulo per l'individuazione del dt

    """
    
    tempo = diz["tempo"]

    dt_T = dt_moda(tempo)

    diz_fft = {}

    i = 0 

    for chiave, dati in diz.items():
        
        if chiave != "tempo":
            
            flusso = dati
            ffts  = fft.fft(flusso)
            diz_fft["ck {}".format(i + 1)] = ffts

            i = i + 1 

    fft_1 = diz_fft["ck 1"]

    freq = fft.fftfreq(len(fft_1), d = dt_T)

    diz_fft["freq"] = freq

    return diz_fft


#----------------------------------------------------------------------------------------------------



def picco_periodo_sint( freq_sint , pot_i_sint, f_taglio, return_freq = True):
    """
    Funzione per la ricerca del massimo di una curva di luce sintetica

    Parametri:
    ---------------
    freq_sint     (array) : contenente le frequenze relative allo studio in frequenza della curve sintetica
    pot_i_sint    (array) : contenente le potenze relative allo studio in frequenza della curve sintetica
    f_taglio      (float) : valore della frequenza al di sotto della quale il contributo viene considerato costante 
    return_freq (boolean) : variabile booleana per determinare se si desidera o meno ottenere la frequenza associata al picco
    
    """
    freq = freq_sint
    pot  = pot_i_sint
    
    freq = freq[:len(freq)//2]
    pot  = pot[:len(pot)//2]

    mask = freq > f_taglio
    freq_tgl = freq[mask]

    n = len(freq) - len(freq_tgl)
    pot_tgl = pot[n:]

    pot_picco = np.max(pot_tgl)


    for i in range(0, len(pot_tgl)):
        if pot_tgl[i] == pot_picco:
            freq_picco = freq_tgl[i]
    
    periodo = [freq_picco, pot_picco]

    if return_freq == True:
        return periodo
    
    if return_freq == False:
        return pot_picco
   

def ar_picchi_sintetici(diz_fft, f_taglio):
    """
    Funzione che dato il dizionario delle trasformate di Fourier trova la potenza relativa ai picchi massisimi
    individuati in ciascun spettro di potenza sintetico

    Parametri:
    ---------------
    diz_fft (dictionary) : contenente lo studio in frequenza delle curve di luce sintetiche
    f_taglio (float)     : valore della frequenza al di sotto della quale il contributo viene considerato costante

    Restituisce:
    ---------------
    ck_picchi_sintetici (array) : contenente le potenze dei picchi sintetici
    
    """

    ck_picchi_sintetici = np.empty(0)

    frequenze_sint = diz_fft["freq"]

    for chiave, dati in diz_fft.items():

        if chiave != "freq":
            
            potenze_sint = dati
            pot_i = picco_periodo_sint( frequenze_sint , potenze_sint, f_taglio, return_freq = False)
            ck_picchi_sintetici = np.append(ck_picchi_sintetici, pot_i )
            
    return ck_picchi_sintetici



#--------------------------------------------------------------

def significatività_int(picchi_sint, picco_orig, n_bin):
    """
    Funzione che calcola la significativtià del periodo associato ad una fonte
    ATTENZIONE! Restituisce unicamente il valore-p ovvero l'are sottesa alla curva che va dalla potenza del periodo originale in poi

    Parametri:
    -------------
    picchi_sint (array) : contenente i valori di potenza dei picchi degli spettri di potenza sintetici
    picco_orig  (float) : valore del picco di potenza originale
    n_bin       (int)   : numero di bin in cui suddividere l'istogramma

    Restituisce:
    --------------
    area        (float) : area sottesa alla curva che va dalla potenza del periodo originale in poi
                          se l'area è nulla restituisce comunque un valore che dipende dalla sensibilità del processo e quindi dal numero di curve simulate  
    
    """
    picchi_sint = np.abs(picchi_sint)**2
    picco_orig  = np.abs(picco_orig)**2

    n , bis, p = plt.hist(picchi_sint, bins = n_bin , density = True)  
    plt.close() 

    centri_bins = 0.5 * (bis[1:] + bis[:-1])
    larg_bins = bis[1] - bis[0]
    
    mask = centri_bins >= picco_orig

    area = np.sum(n[mask]*larg_bins)

    if area == 0:
        area = 1/np.sqrt(len(picchi_sint))
        
    return area

