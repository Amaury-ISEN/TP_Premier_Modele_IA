import tkinter as tk
from tkinter import ttk
from fonctions import *
from tkinter import filedialog

def action_import (self):
    '''Cette fonction sert de commande au bouton d'import de fichier. Elle retourne l'adresse du fichier sélectionné.'''
    
    #Ici on vide la frame qui contient la visualisation du dataframe pour faire place à un nouvel import si jamais elle existe déjà.
    if self.frame_dataframe_importe != None :
        self.frame_dataframe_importe.forget()

    
    self.adresse_fichier = filedialog.askopenfilename()
    print('Fichier sélectionné :', self.adresse_fichier)

    self.df = creer_dataframe(self.adresse_fichier)

    self.frame_dataframe_importe = tk.Frame(self.frame_import)
    self.frame_dataframe_importe.pack(fill='both', expand="no")

    ######################
    ######################
    #Ajout à prendre en compte pour merge avec Eva
    ######################
    self.label = tk.Label(self.frame_dataframe_importe, text = self.adresse_fichier)
    self.label.grid(column=0, row=2, sticky=tk.W)

    self.liste_variables_candidates = []
    for col in self.df.columns:
        self.liste_variables_candidates.append(col)
    ######################
    ######################
    ######################

    # Création d'une Treeview avec barres de défilement pour notre dataframe.
    tree = ttk.Treeview(self.frame_dataframe_importe, show="headings", columns=self.df.columns)
    hsb = tk.Scrollbar(self.frame_dataframe_importe, orient="horizontal", command=tree.xview)
    vsb = tk.Scrollbar(self.frame_dataframe_importe, orient="vertical", command=tree.yview)
    tree.configure(xscrollcommand=hsb.set, yscrollcommand=vsb.set)
    tree.grid(column=0, row=0, sticky=tk.NSEW)
    vsb.grid(column=1, row=0, sticky=tk.NS)
    hsb.grid(column=0, row=1, sticky=tk.EW)
    self.frame_dataframe_importe.grid_columnconfigure(0, weight=1)
    self.frame_dataframe_importe.grid_rowconfigure(0, weight=1)

    for i, header in enumerate(self.df.columns):
        print(i)
        print(self.df.shape)
        tree.column(i, minwidth=0, width=50, anchor='center')
        tree.heading(i, text=header)
        
    for row in range(self.df.shape[0]):
        print(row)
        tree.insert('', 'end', values=list(self.df.iloc[row]))

def afficher_options_reg (self, nouveau_type):
    '''Cette fonction à utiliser avec un bouton dans la frame modele vient changer la variable type_reg pour lui
    attribuer la valeur de nouveau_type. Ensuite, elle affiche dans la frame "modele" les options pour ce type de
    régression-là.'''

    #Si la frame d'options de modèle est déjà remplie par les options d'un modèle quand on clique sur un bouton de modèle, cela la vide.
    if self.frame_options_modele != None :
        self.frame_options_modele.forget()

    self.type_reg.set(nouveau_type)
    print("Type reg changé pour : ", self.type_reg.get())


    ##############################
    # REGRESSION LINEAIRE SIMPLE #
    ##############################
    if self.type_reg.get() == "lin" :
        options_reg(self, variables_multiples = False)
        
        #Choix du learning rate alpha :
        choix_learning_rate(self)

        #Choix du nombre d'itérations :
        choix_nb_iter(self)



    #######################
    # REGRESSION MULTIPLE #
    #######################
    if self.type_reg.get() == "mult" :
        options_reg(self, variables_multiples = True)

        #Choix du learning rate alpha :
        choix_learning_rate(self)

        #Choix du nombre d'itérations :
        choix_nb_iter(self)



    ###########################
    # REGRESSION POLYNOMIALE  #
    ###########################
    if self.type_reg.get() == "poly" :
        options_reg(self, variables_multiples = False)
    
        #Choix du learning rate alpha :
        choix_learning_rate(self)

        #Choix du nombre d'itérations :
        choix_nb_iter(self)

        #Choix du degré :
        choix_degre_pol(self)

    #Bouton pour lancer l'apprentissage:
    self.button = tk.Button(self.frame_options_modele, text = "Lancer l'apprentissage !", command = lambda : lancer_modele(self))
    self.button.pack()


def afficher_type_reg (self):
    print("La régression choisie est :", self.type_reg.get())

class App (tk.Tk):

    def __init__ (self):
        tk.Tk.__init__(self)
        self.geometry("1024x720")

        #############
        # VARIABLES #
        #############
        self.v=tk.IntVar()
        self.type_reg = tk.StringVar()
        self.learning_rate = tk.StringVar()
        self.nb_iter = tk.StringVar()
        self.degre_pol = tk.StringVar()
        self.evolution_cout = []

        #Liste destinée à recueillir les choix possibles de variables à partir du dataframe généré via import par l'utilisateur.
        self.liste_variables_candidates = None
        
        ##########
        # FRAMES #
        ##########
        #Création de la frame principale qui contient toutes les autres de l'appli :
        self.frame_principale = tk.Frame(self)
        self.frame_principale.pack(fill = "both", expand="yes", padx = 10, pady = 10)

        #Création d'un titre pour la fenêtre tkinter :
        self.wm_title("Machine Learning Deluxe")

        #Frames crées par l'usage de boutons. Initialisées sur None pour pouvoir vérifier ailleurs si elles sont déjà remplies et les vider le cas échéant. 
        self.frame_dataframe_importe = None   
        self.frame_options_modele = None
        self.frame_resultats_modele = None

        #Création des frames secondaires :
        self.frame_import = tk.Frame(self.frame_principale, borderwidth=2, relief="groove")
        self.titre_import = tk.Label(self.frame_import, text = "Import")
        self.frame_import.grid(row = 0, column = 0, sticky=tk.N)
        self.titre_import.pack() 
        self.button = tk.Button(self.frame_import, text='Open', command = lambda : action_import(self))
        self.button.pack()


        #Frame pour le modèle (choix du modèle et affichage des options par type de modèle choisi) : 
        self.frame_modele = tk.Frame(self.frame_principale, borderwidth=2, relief="groove")
        self.titre_modele = tk.Label(self.frame_modele, text = "Modele")
        self.frame_modele.grid(row = 0, column = 1, sticky=tk.N)
        self.titre_modele.pack()

        self.button = tk.Button(self.frame_modele, text='Régression linéaire simple', command = lambda : (afficher_options_reg(self, "lin")) )
        self.button.pack()
        self.button = tk.Button(self.frame_modele, text='Régression linéaire multiple', command = lambda : (afficher_options_reg(self, "mult")) )
        self.button.pack()
        self.button = tk.Button(self.frame_modele, text='Régression polynomiale', command = lambda : (afficher_options_reg(self, "poly")) )
        self.button.pack()

        #Frame pour le résultat (visualisation des résultats du modèle et affichage des indicateurs de performance) : 
        self.frame_resultat = tk.Frame(self.frame_principale, borderwidth=2, relief="groove")
        self.frame_resultat.grid(row = 0, column = 2, sticky=tk.N)
        self.titre_resultat = tk.Label(self.frame_resultat, text = "Résultat")
        self.titre_resultat.pack()
    
appli = App()
appli.mainloop()

app = tk.Tk()
#app.geometry("300x300")
#app.resizable(False, False)
app.wm_title("Machine Learning Deluxe")

########################
# Variables générales  #
########################

type_reg = tk.StringVar()

recup_type_reg = type_reg.get()