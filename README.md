# Projet-6

Creating image-classification models.
Applied to different dog breeds. (120 in total, check class_names.npy to see them) 

# Which model is which ?

model5.h5 is a trial hand-made model, generated with the 5 first dog breeds in the database.

wholemodel.h5 is the best hand-made model I could create for the 120 dog breeds.

wholemodelvgg.h5 is a VGG16 re-train.

wholemodelres.h5 is a ResNet50 re-train. It is the best model I could make.

# Main File

GOUBERT_Antoine_1_notebook_072022.ipynb is the main file used to generate the models, and extract the data.

You will first see my first trials, using only 5 classes, and trying to fine tune my model accordingly.

Then the real fun begins with the whole dataframe, data-augmented. If you plan on running the project, get ready to wait.

# App

GOUBERT_Antoine_2_programme_072022.py is a streamlit app that uses wholemodel.h5 to predict the race of a dog on an individual picture (no folder).

# Tracking

trackandrep.csv keeps in memory my tries before data-augmentation.