from sklearn.metrics import accuracy_score, f1_score
import keras_tuner as kt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime

def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int("num_layers", 2, 5)):
        model.add(keras.layers.Dense(
            hp.Choice('units', [16, 32, 64]),
            activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer="adam")
    return model

def train_nn(X_train, y_train, X_test, y_test, dir):

    now = datetime.now()

    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory=dir,
        project_name='intro_to_kt'
    )

    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    tuner.search(X_train, y_train, epochs=25, validation_data=(X_test, y_test))
    best_model = tuner.get_best_models(num_models=1)[0]

    preds = best_model.predict(X_test)

    # Calculate the accuracy score on the test data
    accuracy_test = accuracy_score(y_test, (preds > 0.5))
    f1 = f1_score(y_test, (preds > 0.5), average='macro')

    best_params = tuner.get_best_hyperparameters(num_trials=1)[0].values

    # Print the best hyperparameters
    print("Best hyperparameters:", best_params)
    print("Test accuracy:", accuracy_test)
    print("Test f1:", f1)

    return accuracy_test, f1
