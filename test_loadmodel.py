import numpy as np
import pandas as pd
import os
from skimage import io
from preprocess import preprocess_img
import keras


def test_model(model):
    test = pd.read_csv('GTSRB/GT-final_test.csv', sep=';')

    # Load test dataset
    X_test = []
    y_test = []

    for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('GTSRB/Final_Test/Images/', file_name)
        X_test.append(preprocess_img(io.imread(img_path)))
        y_test.append(class_id)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # predict and evaluate
    y_pred = model.predict_classes(X_test)
    accuracy = float(np.sum(y_pred == y_test)) / float(np.size(y_pred))
    return accuracy


def load_model(model_path='./trained_models/baseline_30_32_0.01.h5'):
    model=keras.models.load_model(model_path)
    return model

model = load_model()
accuracy = test_model(model)
print "Test accuracy = {:.2%}".format(accuracy)
