# esame_blazar_MCF
Repository per la consegna dell'esame di Metodi Computazionali per la Fisica di Valenti Alessandra.

Il progetto riguarda l'analisi della periodictà di fonti astrofisiche, in questo caso i Blazar.
I file di dati sono delle curve di luce fornite dall'esperimento LAT Fermi, satellite che si occupa della rilevazione 
di raggi gamma emessi da queste sorgenti astofisiche.

Oltre ai file contenenti i dati il repository contiene: 

### periodicità_blazar.py 
File che contiene l'analisi della periodicità delle fonti fornite e suddivisa in:
- Analisi preliminare dei dati
- Realizzazione delle curve di luce
- Studio in frequenza e realizzazione dello spettro di potenza
- Fit con la funzione di rumore
- Generazione di curve sintetiche
- Studio della periodicità delle curve sintetiche e calcolo della Significativtià

Tutte le funzioni utilizzate per l'analisi sono contenute nei seguenti file:
### modulo_funzioni_blazar.py
Che contiene le funzioni volte all'analisi dei dati
### modulo_funzioni_plot_blazar.py
Che contiene le funzioni utilizzate per la realizzazione dei grafici 

## Indicazioni di utilizzo
Per eseguire correttamente il programma è necessaio scaricare correttamente sia i file di dati che i moduli di analisi e apporli in un apposita cartella.
Il file di analisi, *periodicità_blazar.py* è strutturato in modo da eseguire solo delle parti di programma sotto indicazione dell'utente tramite l'utilizzo di _argparse_.
L'utente è così in grado di visualizzare indipendentemente le varie fasi del progetto utilizzando l'interfaccia testuale




