import pandas as pd
import numpy as np
from matplotlib import pyplot as pp
import tkinter as tk

from sklearn.metrics import mean_squared_error


def afficher_resultats_modele (self):
    '''Cette fonction sert à afficher dans la frame résultats les résultats (métrics) du modèle généré par l'appentissage.'''
    if self.frame_resultats_modele != None:
        self.frame_resultats_modele.forget()

    self.frame_resultats_modele = tk.Frame(self.frame_resultat)
    self.frame_resultats_modele.pack()
    self.label = tk.Label(self.frame_resultats_modele, text = "Coefficient R2 : " + str(self.R2))
    self.label.pack()
    self.label = tk.Label(self.frame_resultats_modele, text = "Theta : " + str(self.theta))
    self.label.pack()
    self.label = tk.Label(self.frame_resultats_modele, text = "Fonction coût : " + str(self.cout))
    self.label.pack()

    self.label = tk.Label(self.frame_resultats_modele, text = "MSE : " + str(mean_squared_error(self.y, self.y_modele)))
    self.label.pack()
    

def options_reg (self, variables_multiples) :
    #Création du frame pour les options de la régression multiple.
    self.frame_options_modele = tk.Frame(self.frame_modele)
    self.frame_options_modele.pack()

    #Choix de la ou les target(s) pour X :
    if variables_multiples == True :
        self.label = tk.Label(self.frame_options_modele, text = "Choisir les features X :")
    else :
        self.label = tk.Label(self.frame_options_modele, text = "Choisir la feature X :")
    self.label.pack()

    #On crée ici la liste de features potentielles pour le modèle de régression multiple :
    if variables_multiples == True :
        self.liste_features_potentielles = tk.Listbox(self.frame_options_modele, selectmode  = "extended", height = 5, exportselection = 0)
    else :
        self.liste_features_potentielles = tk.Listbox(self.frame_options_modele, selectmode  = "single", height = 5, exportselection = 0)

    #On énumère sur les indices et titres de colonnes du dataframe :
    for i, e in enumerate(self.liste_variables_candidates):
        # i = indice de colonne du dataframe
        # e = string correspondant à un titre de colonne du dataframe
        self.liste_features_potentielles.insert(tk.END, e)
        self.liste_features_potentielles.pack()

    #Choix de la target Y :
    self.label = tk.Label(self.frame_options_modele, text = "Choisir la target Y : ")
    self.label.pack()

    #On crée ici la liste de cibles potentielles pour le modèle de régression multiple :
    self.liste_targets_potentielles = tk.Listbox(self.frame_options_modele, selectmode  = "single", height = 5, exportselection = 0)

    #On énumère sur les indices et titres de colonnes du dataframe :
    for i, e in enumerate(self.liste_variables_candidates):
        # i = indice de colonne du dataframe
        # e = string correspondant à un titre de colonne du dataframe
        self.liste_targets_potentielles.insert(tk.END, e)
        self.liste_targets_potentielles.pack()

def choix_learning_rate(self):
    '''Fonction qui affiche la zone de saisie pour le learning rate.'''
    self.label = tk.Label(self.frame_options_modele, text = "Learning rate :")
    self.label.pack()

    self.entrybox_learning_rate = tk.Entry(self.frame_options_modele, textvariable = self.learning_rate, bd =5)
    self.entrybox_learning_rate.pack()

def choix_nb_iter(self):
    '''Fonction qui affiche la zone de saisie pour le nombre d'itérations.'''
    self.label = tk.Label(self.frame_options_modele, text = "Nombre d'itérations :")
    self.label.pack()

    self.entrybox_nb_iter = tk.Entry(self.frame_options_modele, textvariable = self.nb_iter, bd =5)
    self.entrybox_nb_iter.pack()

def choix_degre_pol(self):
    self.label = tk.Label(self.frame_options_modele, text = "Degré polynomial :")
    self.label.pack()

    self.entrybox_degre_pol = tk.Entry(self.frame_options_modele, textvariable = self.degre_pol, bd =5)
    self.entrybox_degre_pol.pack()


def lancer_modele (self):
    '''Cette fonction lance l'apprentissage du modèle avec les paramètres renseignés par l'utilisateur dans la frame "frame_options_modele".'''
    learning_rate = float(self.learning_rate.get())
    nb_iter = int(self.nb_iter.get())
    
    if self.degre_pol.get() != '':
        degre_pol = int(self.degre_pol.get())
    else : degre_pol = False 

    print("Le learning rate est :", learning_rate)
    #On crée un string features_choisies à partir d'un get sur le stringvar issu de la sélection par l'utilisateur dans la entrybox de features potentielles.
    #features_choisies = self.liste_features_potentielles.get(self.liste_features_potentielles.curselection())
    features_choisies = [self.liste_features_potentielles.get(index) for index in self.liste_features_potentielles.curselection()]
    target_choisie = [self.liste_targets_potentielles.get(index) for index in self.liste_targets_potentielles.curselection()]

    #Création des features et de la target :
    self.X, self.y = creer_X_et_y(features = features_choisies, dataframe = self.df, target = target_choisie)

    #Création d'une feature polynomiale si nécessaire :
    if degre_pol != False :
        self.X = ajout_feature_poly(self.X, degre_pol)

    #Création du biais :
    self.X = ajout_biais(self.X, self.y)

    #Initialisation d'un theta aléatoire :
    self.theta = creer_theta(self.X)
    
    #Création d'une fonction coût :
    self.cout = fonc_cout(self.X, self.y, self.theta)
    print("Fonction coût avant apprentissage :", self.cout)

    #Apprentissage automatique : création d'un theta optimisé et d'un array d'évolution de la fonction coût
    self.theta, self.evolution_cout = desc_grad(self.X, self.y, self.theta, alpha = learning_rate, nb_iter = nb_iter)

    #Nouveau test d'une fonction coût :
    self.cout = fonc_cout(self.X, self.y, self.theta)
    print("Fonction coût après apprentissage :", self.cout)

    #Prédiction d'un y à l'aide du modèle entraîné :
    self.y_modele =  modele(self.X, self.theta)
    
    #Calcul du coeff de détermination R2 :
    self.R2 = coef_R2(self.y, self.y_modele)

    #Affichage tkinter des résultats (metrics et graphique) :
    afficher_resultats_modele(self)
    graphique_cout(self.evolution_cout)

def creer_dataframe (adresse_fichier):
    '''Cette fonction retourne un dataframe pandas créé à partir d'un fichier csv dont on renseigne l'adresse en paramètre.'''
    df = pd.read_csv(adresse_fichier)
    return df

def creer_X_et_y (features, dataframe, target):
    '''Cette fonction prend un dataframe ainsi que des indices de colonnes en argument et 
    retourne un tableau de features sous forme de ndarray. Les indices de colonne sont les
    variables x1, x2, etc.
    Pour rappel, le premier index de colonne dans le dataframe pandas est zéro.'''
    
    #Création d'une liste qui recueillera des ndarrays pour chaque variable x, créés avec les colonnes pandas :
    liste_features = []
    
    #On itère sur la liste issue du curselection de l'utilisateur dans la listbox de features possibles :
    for entete in features:
        liste_features.append(np.array(dataframe[entete]).reshape(-1,1))
    
    #On crée la première colonne de feature :
    X = liste_features[0]
    #On ajoute et concatène les éventuelles colonnes suplémentaires en sautant la première :
    for i in range(1, len(liste_features)):
        X = np.concatenate((liste_features[i],X), axis = 1)
    
    #Création de la target :
    y = np.array(dataframe[target]).reshape(-1,1)
    
    return X, y

def ajout_biais (X, y):
    '''Cette fonction sert à rajouter une colonne de biais (remplie de 1) à une matrice de features.'''
    #Création de la colonne de biais :
    biais = np.ones_like(y)
    
    #Concaténation du tout en un tableau de features X :
    X = np.concatenate((X,biais), axis = 1)
    return X

def ajout_feature_poly (x, degre):
    '''Cette fonction modifie un array de feature pour lui ajouter des features polynomiales du degré spécifié.
    Le biais doit être rajouté APRES l'ajout de features polynomiales.'''

    if degre >= 2:
        for d in range(2,degre+1):
            feat_pol = x**d
            x = np.concatenate((feat_pol, x),axis = 1)
    print("le x est ", x, x.shape)
    return x

def creer_theta (X):
    '''Cette fonction crée un theta aléatoire pour initialiser les paramètres d'un modèle d'apprentissage.
    Pour avoir un theta de la bonne taille, il faut renseigner son array de feature en argument.'''
    theta = np.random.randn(X.shape[1],1)
    return theta

def fonc_cout (X, y, theta):
    '''Cette fonction renvoie une fonction coût pour une association target-feature-theta donnée.'''
    m = len(y)
    return 1/(2*m) * np.sum(((X.dot(theta)) - y )**2)

def grad (X, y, theta):
    '''Cette fonction retourne un gradient calculé à partir de feature(s) X, d'une cible y et d'un
    vecteur paramètre theta.'''
    m = len(y)
    return 1/m * X.T.dot(modele(X, theta) - y)

def desc_grad (X, y, theta, alpha, nb_iter) :
    '''Cette fonction réalise l'algorithme de descente de gradient et retourne un theta optimisé
    en conséquence ainsi qu'un array numpy contenant les valeurs successives de la fonction coût
    durant l'apprentissage (pour visualisation éventuelle).'''
    evolution_cout = np.zeros(nb_iter)

    for i in range(0, nb_iter):
        theta = theta - alpha*grad(X, y, theta)
        evolution_cout[i] = fonc_cout(X,y,theta)

    return theta, evolution_cout

def modele (X, theta):
    '''Cette fonction retourne un vecteur colonne y prédit à partir d'une ou plusieurs feature(s) X et d'un theta.'''
    y_modele = np.dot(X,theta)
    return y_modele

def coef_R2 (y, y_modele):
    '''Cette fonction prend un vecteur target y et un vecteur y prédit par un modèle et retourne
    le coefficient de détermination R2 de ce modèle.'''
    R2 = 1-((y-y_modele)**2).sum()/((y-y.mean())**2).sum()
    return R2

def graphique_cout (evolution_cout):
    '''Cette fonction prend en argument une liste contenant le coût à chaque itération du modèle.
    Elle renvoie un graphique '''
    pp.plot(range(len(evolution_cout.reshape(-1,1))), evolution_cout.reshape(-1,1))
    pp.xlabel("nombre d'itérations")
    pp.ylabel("fonction coût")
    pp.title("Evolution du coût du modèle dans le temps")