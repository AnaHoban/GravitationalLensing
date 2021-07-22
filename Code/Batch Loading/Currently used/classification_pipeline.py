import os
import shutil
import h5py
import numpy as np
from astropy.nddata.utils import Cutout2D
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt
from astropy.visualization import (ZScaleInterval, ImageNormalize)
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from astropy.wcs import WCS,utils
import umap.umap_ as umap
import seaborn as sns


scratch = os.path.expandvars('$SCRATCH') + /

####### 1 ####### this will be part of the script
#Input
#getting cutouts
file_directory = scratch + 'class_dataset'
cutouts = tf.data.Dataset.experimental.load(file_directory, element_spec = tf.TensorSpec(shape=(64,64,1), dtype=tf.float64))

####### 2 ######
#classification
classifier = keras.models.load_model("../Models/binary_classifier")
for i in range(len(classifier.layers)):
    classifier.layers[i].trainable = False

predict_labels = classifier.predict(cutouts)


###### 3 ######
#stats
plt.hist(predict_labels)
plt.title('Histogram of predicted labels on 5000 cutouts #' + str(n)) #*** add batch number 
plt.save(scratch + 'Classification/' + 'hist_' + str(n))



###### 4 ######
#lenses
lenses = cutouts[y_pred > 0.5]
lens_score = y_pred[y_pred > 0.5]

for i, score in enumerate(lens_score):
    f, ax = plt.subplots(1,2)
    ax[0].imshow(lenses[i,...,0])
    ax[0].title('u band ')

    ax[1].imshow(lenses[i,...,1])
    ax[1].title('r band')

    f.suptitle('score: ' + str(i))

    plt.savefig(scratch + 'Classification/Lenses/' + str()) #*** need to find a way to keep track of tile and location, dictionnary?



####### 5 #######
#latent space rep
## ****put draw umap in functions file

reducer = umap.UMAP(random_state=42, n_neighbors = 5, min_dist = 0.1, metric = 'euclidian')
mapper = reducer.fit(test_x.reshape(len(cutouts),64*64*1))
embedding = reducer.transform(test_x.reshape(len(cutouts),64*64*4))

plt.scatter(embedding[:,0], embedding[:,1], c = pred_label)
