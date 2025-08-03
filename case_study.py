
# **CASE STUDY**

Con questo progetto vogliamo individuare
1. le variabili che hanno maggior peso e che garantiscano un maggior successo.
2. quali clienti/lead contattare per tutto il prossimo mese.

Ci vengono forniti due dataset: Historical_Data, su cui basare la nostra analisi, e Actual_Data, da cui scegliere i lead da contattare per il prossimo mese.

Sapendo che il team è composto da 40 advisor, e ogni advisor riesce a contattare 100 leads a settimana, ne deduciamo che il numero di clienti da scegliere è 16.000.

Per iniziare, procediamo a caricare i dati e pulire entrambi i dataset (denominati hist e actual).

## 0. DATA LOAD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
hist = pd.read_excel("/content/drive/MyDrive/Historical_Data.xlsx")
actual = pd.read_excel("/content/drive/MyDrive/Actual_Data.xlsx")

"""## 1. DATA CLEANING SU hist

Iniziamo da una panoramica con le funzioni .info() e .describe() per capire come è strutturato il dataset. A primo impatto non notiamo valori mancanti, ma la funzione .describe() ci aiuterà a visualizzare meglio i valori per le variabili numeriche.
"""

print(hist.head())
print(hist.info())

print("Missing values: \n", hist.isnull().sum())

hist.describe().T

"""Notiamo che
- l'**età** rientra in parametri normali (tra 18 e 75 anni);
- per il r**eddito disponibile** abbiamo un valore minimo negativo. Dovremo andare a individuare queste righe e imputarle.
- il **numero di intestatari** rientra in valori normali, cioè tra 1 e 2.
- lo stesso vale per la **presenza di finanziamenti**, i cui valori saranno 0 o 1.
- il **prezzo dell'immobile** parte da 15000 e arriva a 4 milioni, il che è plausibile.
-  il **valore del mutuo** parte da 0, il che significa dovremo esplorare i casi inferiori a un valore soglia che stabiliremo, e dovremo controllare che il rapporto tra il prezzo dell'immobile e il valore del mutuo rispecchino la colonna LTV
- **LTV** ha un valore minimo di 0 e un massimo superiore a 7
- la colonna **Converted** rientra, come previsto, tra 0 e 1.


Andiamo ora a individuare i valori per le variabili categoriche con .unique()
"""

for col in hist.columns:
    uniques = hist[col].unique()
    if len(uniques) < 20:
        print(f"{col}: {uniques}")

"""Le variabili categoriche rispecchiano i parametri e le definizioni di partenza.

Adesso andiamo a lavorare sulle discrepanze che abbiamo riscontrato.

Iniziamo con il rimuovere valori dove LTV è superiore a 1.2.

In base alle ricerche effettuate, ci sono casi in cui le banche rilasciano mutui per valori superiori al prezzo dell'immobile, laddove si includano spese di ristrutturamento.
"""

hist = hist[hist["LTV"] <= 1.2]

"""Continuiamo con l'individuazione e imputazione dei valori dove reddito disponibile e valore del mutuo sono uguali o inferiori a 0 e cerchiamo di capire se ci sono casi in cui, sebbene rientri nella norma, il valore dell'LTV non rispecchia il rapporto reale tra il valore del mutuo e il prezzo dell'immobile."""

neg_income  = hist["Reddito_disp"] <= 0
zero_mutuo  = hist["valore_mutuo"] <= 0


calc_ltv = hist["valore_mutuo"] / hist["prezzo_immobile"]
bad_ltv = (hist["LTV"] - calc_ltv).abs() > 1e-4

print(f"Reddito ≤ 0   : {neg_income.sum()} rows")
print(f"Mutuo ≤ 0     : {zero_mutuo.sum()} rows")
print(f"LTV mismatch  : {bad_ltv.sum()} rows")

hist.loc[neg_income,  "Reddito_disp"]  = np.nan
hist.loc[zero_mutuo,  "valore_mutuo"]  = np.nan

"""Per il reddito disponibile, utilizzeremo la mediana per tipologia di contratto, mentre per il valore del mutuo utilizzeremo la mediana globale.

La scelta della mediana sta nel fatto che essa è meno sensibile agli outliers e si adatta meglio a distribuzioni asimmetriche rispetto alla media.
"""

hist['Reddito_disp'] = hist.groupby('Tipologia_contratto')['Reddito_disp'].transform(
    lambda x: x.fillna(x.median())
)
hist["valore_mutuo"] = hist["valore_mutuo"].fillna(hist["valore_mutuo"].median())

"""Alla luce dei nuovi valori per la colonna 'valore_mutuo', ricalcoliamo anche la colonna LTV."""

hist["LTV"] = hist["valore_mutuo"] / hist["prezzo_immobile"]

hist.describe().T

"""Adesso, non resta che imputare i casi in cui il reddito è troppo basso. Come valore soglia cautelativo abbiamo scelto 300."""

low_income = hist["Reddito_disp"] < 300
hist.loc[low_income, "Reddito_disp"] = np.nan

hist["Reddito_disp"] = (
    hist.groupby("Tipologia_contratto")["Reddito_disp"]
        .transform(lambda s: s.fillna(s.median()))
)

hist.describe().T

"""Infine, non ci resta che fare un 'sense-check' sui Converted, e assicurarci che non ci siano falsi positivi. Per questo creiamo una colonna rate_approx, che rappresenti il 0.4% del valore del mutuo. Questa soglia è molto approssimativa e cautelativa.
Dopodiché evidenziamo come sospette le righe dove Converted = 1 e la rata approssimativa rappresenta il 120% del reddito disponibile, ed è quindi palesemente impossibile.

Poiché le righe sono poche, decidiamo di eliminarle.
"""

hist["rate_approx"] = 0.004 * hist["valore_mutuo"]

mask_suspect = (
      (hist["Converted"] == 1)
   & (hist["Reddito_disp"] < hist["rate_approx"]* 1.2)
)

print(f"Converted rows clearly unaffordable: {mask_suspect.sum()}")

hist = hist[~mask_suspect]

hist.drop(columns="rate_approx", inplace=True)

"""Un ultimo caso non plausibile è rappresentato da coloro che hanno un reddito disponibile inferiore a 700, hanno richiesto un mutuo superiore a 100.000 e questo gli è stato accettato. è evidente che anche questi siano falsi positivi."""

implausible = (
    (hist['Converted'] == 1) &
    (
        ((hist['Reddito_disp'] < 700) & (hist['valore_mutuo'] > 100000))
    )
)


bad_cases = hist[implausible]
bad_cases

hist = hist[~implausible]

"""## DATA CLEANING SU actual

Procediamo ora alla pulizia del dataset actual. Seguiamo gli stessi step di panoramica del dataset hist.
"""

print(actual.head())
print(actual.info())

print("Missing values: \n", actual.isnull().sum())

"""Notiamo che c'è un solo missing value su LTV, quindi lo visualizziamo per capire a cosa sia dovuto. Il prezzo dell'immobile è pari a 0."""

actual[actual["LTV"].isna()]

actual.describe().T

for col in actual.columns:
    uniques = actual[col].unique()
    if len(uniques) < 20:
        print(f"{col}: {uniques}")

"""Il periodo di acquisizione riporta un valore appartenente alla colonna immobili visitati. Capiamo quanti 'now' ci sono."""

actual['periodo_acquisizione'].value_counts()

"""Come avevamo fatto nel dataset hist, ci occupiamo allo stesso modo, con la mediana, dei casi in cui reddito disponibile e prezzo dell'immobile siano negativi o pari a zero."""

neg_income  = actual["Reddito_disp"] <= 0
zero_prezzo  = actual["prezzo_immobile"] <= 0


calc_ltv = actual["valore_mutuo"] / actual["prezzo_immobile"]
bad_ltv = (actual["LTV"] - calc_ltv).abs() > 1e-4

print(f"Reddito ≤ 0   : {neg_income.sum()} rows")
print(f"Prezzo ≤ 0     : {zero_prezzo.sum()} rows")
print(f"LTV mismatch  : {bad_ltv.sum()} rows")

actual.loc[neg_income,  "Reddito_disp"]  = np.nan
actual.loc[zero_prezzo,  "prezzo_immobile"]  = np.nan

actual['Reddito_disp'] = actual.groupby('Tipologia_contratto')['Reddito_disp'].transform(
    lambda x: x.fillna(x.median())
)
actual["prezzo_immobile"] = actual["prezzo_immobile"].fillna(actual["prezzo_immobile"].median())

actual["LTV"] = actual["valore_mutuo"] / actual["prezzo_immobile"]

print("Missing values: \n", actual.isnull().sum())

actual.describe().T

"""## **EXPLORATORY DATA ANALYSIS**

---
Procediamo ora con una breve EDA, con una matrice di correlazione su Converted per le variabili numeriche e Cramer's V per quelle categoriche.

In questo modo abbiamo un'idea del rapporto che le variabili hanno rispetto a Converted.

Una conclusione finale la otterremo con il metodo SHAP più avanti.

"""

df = hist.copy()
target = "Converted"

num_cols = df.select_dtypes("number").columns

corr = df[num_cols].corr()
corr_with_target = corr["Converted"].drop("Converted")

print(corr_with_target.sort_values(ascending=False))

def cramers_v(cat, target):
    confusion = pd.crosstab(cat, target)
    chi2 = ss.chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    return np.sqrt(chi2 / (n * (min(confusion.shape)-1)))

cat_corr = {col: cramers_v(df[col], df["Converted"]) for col in df.select_dtypes("object").columns}
cat_corr = pd.Series(cat_corr).sort_values(ascending=False)
print(cat_corr)

"""## **FEATURE ENGINEERING**

Aggiungiamo 3 nuove variabili:

- DTI (Debt To Income): il valore del mutuo sul reddito disponibile

- income_pc: per normalizzare il reddito per numero di intestatari

- LTV bucket: così da dividere LTV in fasce (bins)

- Highltv Finanz: in caso di LTV elevato e presenza di finanziamenti
"""

def add_features(df):
    df = df.copy()


    df["DTI"] = df["valore_mutuo"] / df["Reddito_disp"]


    df["income_pc"] = df["Reddito_disp"] / df["numero_intestatari"]


    df["LTV_bucket"] = pd.cut(df["LTV"],
                              bins=[0,0.6,0.8,0.9,1],
                              labels=["0_60","60_80","80_90","90_100"])


    df["HighLTV_Finanz"] = (df["LTV"] > .8) & (df["Presenza_finanziamenti"] == 1)

    return df

hist   = add_features(hist)
actual = add_features(actual)

"""## TRAIN/TEST SPLIT

Procediamo ora alla preparazione degli algoritmi.

Iniziamo con un train/test split stratificato.
"""

X = hist.drop(columns=["Converted",
                       "prezzo_immobile","valore_mutuo"])
y = hist["Converted"]

from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

"""Procediamo a normalizzare le variabili, applicando lo Standard Scaler sulle variabili numeriche e lo OneHotEncoder su quelle categoriche."""

from sklearn.compose       import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num = X_tr.select_dtypes("number").columns
cat = X_tr.select_dtypes(exclude="number").columns

prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
      ])

"""## LOGISTIC REGRESSION L1

Iniziamo con il classico algoritmo per classificazione Logistic Regression.

Definiamo un modello di regressione logistica con penalizzazione L1 (Lasso)
- L1 aiuta a selezionare automaticamente le variabili più rilevanti (sparse model)

Creiamo una pipeline composta da:
- preprocessamento (scaling + one-hot encoding)
- selezione automatica delle feature più importanti
- classificatore logistic regression

Calcoliamo le probabilità di conversione sul set di test e valutiamo la performance del modello usando average precision visto che abbiamo dati sbilanciati.
"""

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

l1 = LogisticRegression(
        penalty="l1", solver="saga", class_weight="balanced",
        max_iter=5000,
        tol=1e-4)

pipe = Pipeline([
        ("prep", prep),
        ("sel" , SelectFromModel(l1, threshold="median")),
        ("clf" , l1)
      ])

param_grid = {"clf__C": [0.05, 0.1, 0.3, 1, 3]}
cv = StratifiedKFold(5, shuffle=True, random_state=42)

gs = GridSearchCV(pipe, param_grid, cv=cv,
                  scoring="average_precision", n_jobs=-1)
gs.fit(X_tr, y_tr)

print("Best C :", gs.best_params_["clf__C"])
print("CV AP  :", gs.best_score_.round(3))

"""Applichiamo poi un modello di Random Forest per predire la probabilità di conversione dei lead. La Random Forest è un algoritmo robusto e non lineare, molto efficace in presenza di variabili eterogenee (numeriche e categoriche) e interazioni complesse tra le feature.

La pipeline è composta da due blocchi principali:

prep: il preprocessamento dei dati (scaling per le variabili numeriche e one-hot encoding per le categoriche), identico a quello usato nel training degli altri modelli

rf: il classificatore Random Forest vero e proprio, configurato con:

n_estimators=400: numero di alberi nella foresta

min_samples_leaf=50: previene l’overfitting imponendo un numero minimo di osservazioni per foglia

class_weight="balanced_subsample": bilancia automaticamente le classi sbilanciate in ogni albero

Dopo il training, calcoliamo la probabilità di conversione per ogni lead nel set di test (X_te) e valutiamo il modello tramite Average Precision Score, una metrica adatta a problemi sbilanciati come questo (molti più lead non convertiti rispetto a quelli convertiti).



"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

rf_pipe = Pipeline([
        ("prep", prep),
        ("rf", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=50,
                class_weight="balanced_subsample", random_state=42))
      ]).fit(X_tr, y_tr)

proba_rf = rf_pipe.predict_proba(X_te)[:,1]
ap_rf = average_precision_score(y_te, proba_rf)
print("AP random-forest :", round(ap_rf, 3))

"""Infine, implementiamo un modello di XGBoost (Extreme Gradient Boosting), uno degli algoritmi più potenti per l’analisi predittiva su dati strutturati. XGBoost costruisce una sequenza di alberi decisionali ottimizzati uno dopo l’altro, correggendo gli errori commessi dagli alberi precedenti, e riesce a catturare relazioni non lineari complesse tra le variabili.

I parametri principali del modello sono:

objective="binary:logistic": problema di classificazione binaria (Converted = 1 o 0)

eval_metric="aucpr": metrica di valutazione basata sulla precision-recall (adatta per dati sbilanciati)

learning_rate=0.05, n_estimators=400: controllano il numero e la dimensione dei passi dell’algoritmo

max_depth=4: limita la profondità degli alberi per evitare overfitting

subsample=0.8, colsample_bytree=0.8: usano solo una parte delle righe/colonne per ogni albero, che aumenta la generalizzazione

scale_pos_weight: bilancia la classe positiva (Converted = 1) in base alla proporzione presente nei dati

Dopo il training, calcoliamo le probabilità di conversione sui dati di test e valutiamo la performance con l’Average Precision Score.

Infine, utilizziamo la libreria SHAP per spiegare il contributo di ogni feature alla predizione. Il grafico SHAP mostra le variabili più influenti per il modello, consentendo un’interpretazione trasparente anche di un algoritmo complesso come XGBoost.


"""

import xgboost as xgb, shap
gbm = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        learning_rate=0.05,
        n_estimators=400,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(1-y_tr).sum()/y_tr.sum(),
        random_state=42)

pipe_GBM = Pipeline([("prep", prep), ("gbm", gbm)]).fit(X_tr, y_tr)
proba_GBM = pipe_GBM.predict_proba(X_te)[:,1]
print("AP XGBoost:", average_precision_score(y_te, proba_GBM).round(3))


explainer = shap.Explainer(pipe_GBM.named_steps["gbm"])
shap_values = explainer(pipe_GBM.named_steps["prep"].transform(X_te))
shap.summary_plot(shap_values, feature_names=pipe_GBM.named_steps["prep"].get_feature_names_out())

"""Dall’analisi emergono alcune evidenze chiave:

La variabile immobili_visitati_compromise è la più influente: i lead che hanno già fatto un'offerta per una determinata casa hanno una probabilità di conversione molto più alta.

Anche Età, LTV, DTI e Reddito_disp risultano fondamentali: indicano il profilo finanziario del richiedente e la sostenibilità della richiesta di mutuo.

La presenza di un contratto a tempo indeterminato (Tipologia_contratto_open_ended_term) è anch’essa un forte indicatore positivo, segnale di stabilità lavorativa.

Variabili temporali come il periodo di acquisizione influenzano la conversione, suggerendo che la "freschezza" del lead può incidere sul comportamento.

In generale, il modello premia:

la motivazione del lead (comportamenti attivi come la visita agli immobili),

la solidità economica (reddito elevato, basso LTV/DTI),

e la stabilità occupazionale.

Definiamo adesso la soglia di punteggio (cutoff) da utilizzare per selezionare i lead più promettenti da contattare.

Poiché il team commerciale può gestire al massimo 16.000 lead al mese (CALL_CAP = 16000), dobbiamo individuare i lead con la probabilità di conversione più alta (calcolata dal modello XGBoost).

Il codice ordina tutte le probabilità predette (proba_GBM) in ordine crescente e seleziona il valore della 16.000ª probabilità più alta. Questo valore rappresenta la soglia minima sopra la quale un lead viene considerato tra i "top" da chiamare.

Tutti i lead con un punteggio maggiore o uguale a cutoff formeranno la lista finale da contattare.
"""

CALL_CAP = 16000
cutoff   = np.sort(proba_GBM)[-min(CALL_CAP, len(proba_GBM))]

"""Procediamo a valutare la performance del modello  tenendo conto del limite reale di lead contattabili.

Invece di misurare la precisione su tutte le predizioni, la funzione calcola:

la precisione e il recall solo sui migliori k lead con la probabilità di conversione più alta,

dove k viene scalato proporzionalmente in base alla dimensione del set di test rispetto al dataset completo (23.199 righe nel training).


"""

def precision_k(model_pipe, X_test, y_test,
                k_full=16000,
                n_full=23199):

    proba = model_pipe.predict_proba(X_test)[:, 1]


    k_sub = min(len(proba), int(np.ceil(k_full * len(proba) / n_full)))

    thresh = np.sort(proba)[-k_sub]
    y_hat  = (proba >= thresh).astype(int)

    tp   = (y_hat & y_test).sum()
    prec = tp / y_hat.sum()
    rec  = tp / y_test.sum()
    return prec, rec, thresh, k_sub

"""Utilizziamo adesso la funzione precision_k() per confrontare la performance dei tre modelli principali (Logistic Regression, Random Forest e XGBoost) nel contesto operativo realistico: possiamo contattare solo un numero limitato di lead (ad esempio 16.000 su tutto il dataset).

Poiché stiamo lavorando su un set di test (X_te), k viene adattato proporzionalmente alla sua dimensione. Per ciascun modello vengono calcolati:

Precision: la percentuale di lead contattati che si sono effettivamente convertiti (quanto è "pulita" la lista)

Recall: la percentuale dei lead realmente convertiti che siamo riusciti a "catturare" nella top-k (quanto ne prendiamo)

Questo confronto è fondamentale per decidere quale modello utilizzare in produzione, non solo in base alla metrica globale, ma anche all’efficacia concreta nel contesto del business.
"""

prec_log, rec_log, t_log, k_sub = precision_k(gs.best_estimator_, X_te, y_te)
prec_rf , rec_rf , t_rf ,  _    = precision_k(rf_pipe           , X_te, y_te)
prec_gbm, rec_gbm, t_gbm, _     = precision_k(pipe_GBM         , X_te, y_te)

print(f"Subset size   : {len(X_te)} rows")
print(f"k used        : {k_sub} (≈ 69 % of rows)")

print(f"Logit  : Precision {prec_log:.2%} | Recall {rec_log:.2%}")
print(f"Forest : Precision {prec_rf :.2%} | Recall {rec_rf :.2%}")
print(f"XGBoost: Precision {prec_gbm:.2%} | Recall {rec_gbm:.2%}")

"""Ripresentiamo la funzione precision_k adattata però alle dimensioni del dataset actual, per capire quanto bene funzioni il modello se applicato realmente. Ripetiamo anche il confronto tra modelli."""

def precision_k(model_pipe, X_test, y_test,
                k_full=16000, n_full_actual=34503):

    frac  = k_full / n_full_actual
    k_sub = int(np.ceil(frac * len(X_test)))

    proba  = model_pipe.predict_proba(X_test)[:, 1]
    thresh = np.sort(proba)[-k_sub]
    y_hat  = (proba >= thresh).astype(int)

    tp   = (y_hat & y_test).sum()
    prec = tp / y_hat.sum()
    rec  = tp / y_test.sum()
    return prec, rec, thresh, k_sub

prec_log, rec_log, t_log, k_sub = precision_k(gs.best_estimator_, X_te, y_te)
prec_rf , rec_rf , t_rf , _     = precision_k(rf_pipe           , X_te, y_te)
prec_gbm, rec_gbm, t_gbm,_      = precision_k(pipe_GBM         , X_te, y_te)

print(f"Subset size : {len(X_te)} rows  |  k used = {k_sub}")
print(f"Logit  : P {prec_log:.2%}  |  R {rec_log:.2%}")
print(f"Forest : P {prec_rf :.2%}  |  R {rec_rf :.2%}")
print(f"XGB    : P {prec_gbm:.2%}  |  R {rec_gbm:.2%}")

"""Abbiamo quindi scelto come modello finale la Random Forest perché ha mostrato un ottimo equilibrio tra precisione e recall nei test effettuati con la funzione precision_k(), risultando il più adatto al nostro obiettivo pratico: identificare i lead con maggiore probabilità di conversione, entro i vincoli di capacità del team commerciale.

Rispetto ad altri modelli provati (regressione logistica, XGBoost), la Random Forest offre:

- una buona capacità predittiva

- maggiore robustezza rispetto agli outlier

- e una buona interpretabilità tramite le importanze delle feature.

Per sfruttare al massimo i dati disponibili, eseguiamo il retraining del modello finale su tutto il dataset storico (X, y), così che possa apprendere dai pattern presenti anche nel set di test e fornire predizioni più accurate sui lead attuali da contattare.
"""

best_model = rf_pipe

best_model.fit(X, y)

"""In questa cella applichiamo il modello finale Random Forest allenato su tutto il dataset storico al dataset actual, che contiene i lead attualmente da valutare."""

actual_scores = best_model.predict_proba(
        actual.drop(columns=["prezzo_immobile", "valore_mutuo"]))[:, 1]

actual["score"] = actual_scores
call_list = (actual
             .sort_values("score", ascending=False)
             .head(16000))

call_list.to_excel("Leads_to_call.xlsx", index=False)
print("✓ Leads_to_call.xlsx creato (16 000 righe)")
