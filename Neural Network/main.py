#Suppress tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from io import StringIO

import re
import json
import sys
import time

import numpy as np
import pandas as pd

import tensorflow as tf

from matplotlib import pyplot as plt

#Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_regression

#Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from keras.callbacks import EarlyStopping

#====================================================================================================

#Accepts a data frame containing a text column named "text"
#Returns a numpy array where the text column is replaced by the tf-idf matrix
class TextColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, min_df_arg):
        self.vectorizer = TfidfVectorizer(min_df=min_df_arg)

    def fit(self, X):
        if "text" in X:
            self.vectorizer.fit(X["text"].values)

        #print(f"Number of text dimensions: {len(self.vectorizer.get_feature_names_out())}")

        return self

    def transform(self, X):
        if "text" in X:
            tfidf_matrix = self.vectorizer.transform(X["text"].values)
            return pd.concat([pd.DataFrame(tfidf_matrix.toarray()).reset_index(inplace=False, drop=True), X.drop("text", inplace=False, axis=1).reset_index(inplace=False, drop=True)], axis=1, join="inner").to_numpy()
        else:
            return X.to_numpy()
    
#====================================================================================================

#Accepts a dataframe with two columns
#Scales the two columns with StandardScaler together, considering them as one
#Returns a numpy array with the two scaled columns, after separating them again
class TwoColumnScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.Y_concatenated = None
        self.scaler = StandardScaler()

    def fit(self, Y):
        self.Y_concatenated = np.vstack((Y['date_min'].values.reshape(-1, 1), Y['date_max'].values.reshape(-1, 1)))
        self.scaler.fit(self.Y_concatenated)

        return self

    def transform(self, Y):
        scaled = self.scaler.transform(Y.values.reshape(-1, 1))

        return scaled.reshape(-1,2, order="F")
    
    def inverse_transform(self, Y):
        unscaled = self.scaler.inverse_transform(Y.reshape(-1, 1, order="F"))

        return unscaled.reshape(-1,2, order="C")
    

#====================================================================================================    

#Loss function (also used as a metric)
def rmse(y_true, y_pred):
    date_min = y_true[:,0]
    date_max = y_true[:,1]
    y_pred = y_pred[:,0]

    within_range = tf.logical_and(y_pred >= date_min, y_pred <= date_max)
    loss_within_range = tf.zeros_like(y_pred)
    loss_outside_range = tf.minimum(tf.abs(date_min - y_pred), tf.abs(date_max - y_pred))
    loss = tf.where(within_range, loss_within_range, loss_outside_range)

    loss = tf.sqrt(tf.reduce_mean(tf.square(loss)))

    return loss
    

#====================================================================================================

def run_experiment(X, Y, hidden_neurons, hyperparams, rate, k, kfold, pipe, outscaler, num_epochs, experiment, early_stopping):

    rmseList = []

    avg_loss_train = np.zeros((num_epochs, 1))
    avg_loss_valid = np.zeros((num_epochs, 1))

    callbacks = None

    if early_stopping:
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=5, verbose=0)]

    #Cross Validation
    for i, (train, test) in enumerate(kfold.split(X)):

        #Split into train and test
        X_train = X.iloc[train]
        X_test = X.iloc[test]
        Y_train = Y.iloc[train]
        Y_test = Y.iloc[test]

        #Apply transformations
        X_train = pipe.fit_transform(X_train)
        X_test = pipe.transform(X_test)
        Y_train = outscaler.fit_transform(Y_train)
        Y_test = outscaler.transform(Y_test)

        #Keep best features
        selector = SelectKBest(f_regression, k=min(1000, X_train.shape[1]))
        selector.fit(X_train, np.mean(Y_train, axis=1))

        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)

        
        #Model structure
        #______________________________________________________________________________________________________________
        
        model = Sequential()

        if experiment == 4: #Dropout
            model.add(Dropout(rate[0]))
            model.add(Dense(hidden_neurons[0], activation="relu", input_dim=X_train.shape[1]))
            model.add(Dropout(rate[1]))
            model.add(Dense(hidden_neurons[1], activation="relu", input_dim=hidden_neurons[0]))
            model.add(Dropout(rate[1]))
            model.add(Dense(hidden_neurons[2], activation="relu", input_dim=hidden_neurons[1]))
            model.add(Dropout(rate[1]))
            model.add(Dense(1, activation="linear", input_dim=hidden_neurons[2]))

        else:
            if len(hidden_neurons) == 1:
                model.add(Dense(hidden_neurons[0], activation="relu", input_dim=X_train.shape[1]))
                model.add(Dense(1, activation="linear", input_dim=hidden_neurons[0]))
            
            elif len(hidden_neurons) == 2:
                model.add(Dense(hidden_neurons[0], activation="relu", input_dim=X_train.shape[1]))
                model.add(Dense(hidden_neurons[1], activation="relu", input_dim=hidden_neurons[0]))
                model.add(Dense(1, activation="linear", input_dim=hidden_neurons[1]))

            elif len(hidden_neurons) == 3:
                model.add(Dense(hidden_neurons[0], activation="relu", input_dim=X_train.shape[1]))
                model.add(Dense(hidden_neurons[1], activation="relu", input_dim=hidden_neurons[0]))
                model.add(Dense(hidden_neurons[2], activation="relu", input_dim=hidden_neurons[1]))
                model.add(Dense(1, activation="linear", input_dim=hidden_neurons[2]))

        #model.summary()

        #______________________________________________________________________________________________________________

        #Specify learning rate and momentum
        optimizer = tf.keras.optimizers.SGD(learning_rate=hyperparams[0], momentum=hyperparams[1])

        #Compile model
        model.compile(loss=rmse, optimizer=optimizer, metrics=[rmse])

        #Fit model
        history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=num_epochs, batch_size=32, verbose=0, shuffle=True, callbacks=callbacks)

        avg_loss_train[:, 0] += np.pad(history.history["loss"], (0, max(0, num_epochs - len(history.history["loss"]))), mode="edge")
        avg_loss_valid[:, 0] += np.pad(history.history["val_loss"], (0, max(0, num_epochs - len(history.history["val_loss"]))), mode="edge")

        #Evaluate
        scores = model.evaluate(X_test, Y_test, verbose=0)
        rmseList.append(scores[0])
        print(f"Fold: {i}\nRMSE: {scores[0]}\n")
        
        '''
        #For scaled
        predictions = model.predict(X_test, verbose=0)
        Y_test = outscaler.inverse_transform(Y_test)
        predictions = outscaler.scaler.inverse_transform(predictions)
        df = pd.DataFrame({'Lower': np.floor(Y_test[:,0]), 'Upper': np.floor(Y_test[:,1]), 'Predictions': np.floor(predictions.flatten())})
        print(f'Number of correct answers: {len(df.loc[(df["Predictions"] >= df["Lower"]) & (df["Predictions"] <= df["Upper"])])} out of {df.shape[0]}')
        #df.to_excel(f'results/{j}_{i}.xlsx', index=False)
        '''

        '''
        #For unscaled

        predictions = model.predict(X_test, verbose=0)
        #Y_test = outscaler.inverse_transform(Y_test)
        #predictions = outscaler.scaler.inverse_transform(predictions)
        Y_test = Y_test.values
        df = pd.DataFrame({'Lower': np.floor(Y_test[:,0]), 'Upper': np.floor(Y_test[:,1]), 'Predictions': np.floor(predictions.flatten())})
        print(f'Number of correct answers: {len(df.loc[(df["Predictions"] >= df["Lower"]) & (df["Predictions"] <= df["Upper"])])} out of {df.shape[0]}')
        #df.to_excel(f'results/{j}_{i}.xlsx', index=False)
        '''

    return rmseList, avg_loss_train/k, avg_loss_valid/k

#================================================================================================================================
    
if __name__ == "__main__":

    #experiment = 0 -> hidden layer neurons 
    #experiment = 1 -> +1 hidden layer
    #experiment = 2 -> +2 hidden layers
    #experiment = 3 -> momentum
    #experiment = 4 -> dropout
    
    experiment = 0
    early_stopping = False
    
    #Dataset
    #=======================================================================================================

    #Load the file
    dataset = pd.read_csv("iphi2802.csv", delimiter="\t", converters={"text": lambda x: x.replace("[","").replace("]","")}, usecols=["text", "date_min", "date_max"])

    #Split into input and output
    X = dataset[["text"]]
    Y = dataset[["date_min", "date_max"]]

    #Transformations
    #=======================================================================================================

    #Input pipeline
    
    pipe = Pipeline([
        ("vectorizer", TextColumnTransformer(5)),
        ("scaler", StandardScaler())
    ])

    '''
    pipe = Pipeline([
        ("vectorizer", TextColumnTransformer(5))
    ])
    '''
    
    #Target scaler
    outscaler = TwoColumnScaler()

    k = 5
    kfold = KFold(n_splits = k, shuffle=False)
    
    #Experiment params
    #===================================================================================================================
    num_epochs = 50

    num_neurons = None
    hyperparams = [(0.001, 0.0)] #(learning rate, momentum)
    rate=None

    #______________________________________________________________________________________________________________

    if experiment == 0:
        #num_neurons = [(5,), (100,), (1000,)]
        num_neurons = [(5,), (40,), (70,)]

    elif experiment == 1:
        num_neurons = [(5, 3), (5, 5), (5, 50)]

    elif experiment == 2:
        num_neurons = [(5, 50, 20), (5, 50, 50), (5, 50, 70)]

    elif experiment == 3:
        num_neurons = [(5, 50, 70)]
        hyperparams = [(0.001, 0.2), (0.001, 0.6), (0.05, 0.6), (0.1, 0.6)]

    elif experiment == 4:
        num_neurons = [(5, 50, 70)]
        hyperparams = [(0.001, 0.2)]
        #rate = [(0.8, 0.5),(0.5, 0.5),(0.8, 0.2)]
        rate = [(0.8, 0.2), (0.0, 0.0)]

    #______________________________________________________________________________________________________________


    #Testing number of hidden layers and nodes
    #================================================================================================================
    if experiment < 3:

        avg_loss_train = np.zeros((num_epochs, len(num_neurons)))
        avg_loss_valid = np.zeros((num_epochs, len(num_neurons)))

        for j, hidden_neurons in enumerate(num_neurons):

            rmseList, loss, val_loss = run_experiment(X, Y, hidden_neurons, hyperparams[0], rate, k, kfold, pipe, outscaler, num_epochs, experiment, early_stopping)
            avg_loss_train[:, j] += loss[:,0]
            avg_loss_valid[:, j] += val_loss[:,0]
            print(f"Average RMSE for {'-'.join(str(x) for x in hidden_neurons)} neurons: {np.mean(rmseList)}\n")

            print("===================================================================\n")

        #Plot
        for j, num in enumerate(num_neurons):
            plt.plot(avg_loss_train[:,j], label=f"Train Loss for {'-'.join(str(x) for x in num)} neurons")
            plt.plot(avg_loss_valid[:,j], label=f"Valid Loss for {'-'.join(str(x) for x in num)} neurons")
    
    #Hyperparameter tuning
    #================================================================================================================
    elif experiment == 3:
        
        avg_loss_train = np.zeros((num_epochs, len(hyperparams)))
        avg_loss_valid = np.zeros((num_epochs, len(hyperparams)))

        for j, params in enumerate(hyperparams):

            rmseList, loss, val_loss = run_experiment(X, Y, num_neurons[0], params, rate, k, kfold, pipe, outscaler, num_epochs, experiment, early_stopping)
            avg_loss_train[:, j] += loss[:,0]
            avg_loss_valid[:, j] += val_loss[:,0]
            print(f"Average RMSE for (h={params[0]}, m={params[1]}): {np.mean(rmseList)}\n")

            print("===================================================================\n")

        #Plot
        for j, params in enumerate(hyperparams):
            plt.plot(avg_loss_train[:,j], label=f"Train Loss for (h={params[0]}, m={params[1]})")
            plt.plot(avg_loss_valid[:,j], label=f"Valid Loss for (h={params[0]}, m={params[1]})")

    #Dropout
    #================================================================================================================
    elif experiment == 4:
        
        avg_loss_train = np.zeros((num_epochs, len(rate)))
        avg_loss_valid = np.zeros((num_epochs, len(rate)))

        for j, r in enumerate(rate):

            rmseList, loss, val_loss = run_experiment(X, Y, num_neurons[0], hyperparams[0], r, k, kfold, pipe, outscaler, num_epochs, experiment, early_stopping)
            avg_loss_train[:, j] += loss[:,0]
            avg_loss_valid[:, j] += val_loss[:,0]
            print(f"Average RMSE for (Rin={r[0]}, Rh={r[1]}): {np.mean(rmseList)}\n")

            print("===================================================================\n")

        #Plot
        for j, r in enumerate(rate):
            plt.plot(avg_loss_train[:,j], label=f"Train Loss for (Rin={r[0]}, Rh={r[1]})")
            plt.plot(avg_loss_valid[:,j], label=f"Valid Loss for (Rin={r[0]}, Rh={r[1]})")

    #================================================================================================================

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Convergence Plot per Training Cycle')
    plt.legend()
    plt.show()