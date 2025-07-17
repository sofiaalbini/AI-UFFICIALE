#DATASET 
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import  load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np

#IRIS 
iris = load_iris()
X_iris = iris.data
y_iris = iris.target 

y_iris = np.eye(3)[y_iris] 




X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)  

print(X_train.shape)

print(X_test.shape)



input_iris=4 

output_iris=3 


#VISUALIZZAZIONE DATASET 
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Aggiunta della colonna target (le etichette)
df['target'] = iris.target

# Visualizza le prime 5 righe
#print(df.head())



"""
#REGRESSIONE 


diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

""" 

