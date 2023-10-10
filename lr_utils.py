import numpy as np
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    #Ouvre le fichier HDF5 contenant l'ensemble d'apprentissage en mode lecture ("r") 
    # et stocke l'objet fichier dans train_dataset.

    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    #Lit le tableau "train_set_x" du fichier HDF5 et le convertit en un dataframe numpy.
    #Lit le dataframe "train_set_y" du fichier HDF5 et le convertit en un dataframe numpy.

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r") 
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    #Ouvre le fichier HDF5 contenant l'ensemble de test en mode lecture ("r").
    #Lit le dataframe "test_set_x" du fichier HDF5 et le convertit en un dataframe numpy.
    #Lit le dataframe "test_set_y" du fichier HDF5 et le convertit en un dataframe numpy.


    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    #Lit le dataframe "list_classes" du fichier HDF5 qui contient les différentes classes possibles.
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    #train_set_y_orig : Redimensionne le dataframe d'étiquettes d'entraînement pour qu'il ait une forme de (1, n), 
    # où n est le nombre d'exemples d'entraînement.
    #test_set_y_orig : Redimensionne le dataframe d'étiquettes de test pour qu'il ait une forme de (1, m), 
    # où m est le nombre d'exemples de test.

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
    #Retourne les dataframe numpy et les classes chargés.
