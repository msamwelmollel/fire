from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn import metrics



#print(tf.__version__)

c = [20]
degree=[1,2,3,4,6,10,20,100]

for p in degree:
    for k in c:
        print(k)
        
        #%%
        path = "forestfires.csv"
        
        dataset = pd.read_csv(path,sep=',')
        
        #print(dataset.head())
        dataset = dataset.drop(["month","day","X","Y","FFMC","DMC","DC","ISI"], axis=1)
        #print(dataset.head())
        
        dataset.isna().sum()
        
        
        train_dataset = dataset.sample(frac=0.8,random_state=0)
        test_dataset = dataset.drop(train_dataset.index)
        #print(train_dataset.shape)
        #print(test_dataset.shape)
        
        
        #sns.pairplot(train_dataset[["X", "Y", "FFMC", "DMC","ISI","temp","RH","wind","rain","area"]], diag_kind="kde")
        
        #%%
        train_labels = train_dataset.pop('area')
        train_labels = np.log((train_labels + 1)) 
        labels= train_labels.to_numpy().ravel()
        #print(labels.shape)
        
        test_labels = test_dataset.pop('area')
        test_labels= np.log((test_labels + 1))  # purposely to remove the skewness of the labe;
        
        #print(test_labels)
        
        # Normalize the data
        max_abs_scaler = preprocessing.MaxAbsScaler()
        X_train_maxabs = max_abs_scaler.fit_transform(train_dataset)
        np.reshape(X_train_maxabs,(414,4))
        X_train = pd.DataFrame(X_train_maxabs, index=range(X_train_maxabs.shape[0]),
                                  columns=range(X_train_maxabs.shape[1]))
        #print((X_train_maxabs[:2]))
        
        
        X_test_maxabs = max_abs_scaler.fit_transform(test_dataset)
        
        
        X_test = pd.DataFrame(X_test_maxabs, index=range(X_test_maxabs.shape[0]),
                                  columns=range(X_test_maxabs.shape[1]))
        #print((X_train_maxabs[:2]))
        
        #%%
        # Fit regression model
        svr_rbf = SVR(kernel='rbf', C=k, gamma=0.1, epsilon=.1)
        svr_lin = SVR(kernel='linear', C=k, gamma='auto')
        svr_poly = SVR(kernel='poly', C=k, gamma='auto', degree=p, epsilon=.1,coef0=1)
        
        svrs = [svr_rbf, svr_lin, svr_poly]
        labels[:10]
        
        
        
        svr_rbf.fit(X_train_maxabs, labels)
        svr_lin.fit(X_train_maxabs, labels)
        svr_poly.fit(X_train_maxabs, labels)
        
        
        y_pred_rbf = svr_rbf.predict(X_test_maxabs)
        y_pred_lin = svr_lin.predict(X_test_maxabs)
        y_pred_poly = svr_poly.predict(X_test_maxabs)
        
        Mrbf = metrics.mean_squared_error(test_labels,y_pred_rbf)
        Mlin = metrics.mean_squared_error(test_labels,y_pred_lin)
        Mpoly = metrics.mean_squared_error(test_labels,y_pred_poly)
        
        print("MSE RBF:",metrics.mean_squared_error(test_labels,y_pred_rbf))
        print("MSE lin:",metrics.mean_squared_error(test_labels,y_pred_lin))
        print("MSE poly:",metrics.mean_squared_error(test_labels,y_pred_poly))


