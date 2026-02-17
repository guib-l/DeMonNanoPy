#!/usr/bin/env python3
import __future__

# Import standard de python3
import os
import sys

import json

import copy
import numpy as np
from copy import deepcopy



class deMonEnviron:
    """
    Définit un environnement de calcul pour deMon.
    Cette classe permet de configurer et gérer les paramètres nécessaires
    pour exécuter des calculs avec le logiciel deMon.
    """
    def __init__(
            self, 
            workdir="",
            config="parameters.json",
            exec=None,
            omp_threads=1 ):
        """
        Initialise l'environnement de calcul deMon.
        Args:
            workdir (str): Répertoire de travail pour les calculs deMon.
            config (str): Chemin vers le fichier de configuration JSON.
            omp_threads (int): Nombre de threads OpenMP à utiliser.
        """
        self.workdir = workdir
        _config = json.load(open(config))

        self.config  = _config

        if exec is not None:
            self.config["DEMON_EXECUTABLE"] = exec

        self.executable  = "cd %s && "%self.workdir + self.config["DEMON_EXECUTABLE"] 
        self.prefix      = self.config["PREFIX"] 
        self.omp_threads = omp_threads

    def build(self,
              calculator,
              title="Calcualtion with deMonNano"):
        """
        Construit les paramètres de calcul pour deMon à partir du fichier 
        de configuration.
        Returns:
            calc: Instance de la classe calculator configurée avec les 
                  paramètres de calcul.
        """
        input_arguments = {
            'TITLE':title.uppercase(),
            'DFTB': self.config["DEMON_PARAMETERS"]["DFTB"],
            'PARAM': 'PTYPE=%s'%self.config["DEMON_PARAMETERS"]["SKFILE"],
            'OTHER':{},
        } 
        
        # Ajouter les paramètres supplémentaires
        for key in self.config["DEMON_PARAMETERS"]:
            if key not in ["DFTB","SKFILE","WMULL"]:
                input_arguments[key] = self.config["DEMON_PARAMETERS"][key]


        calc = calculator(
            workdir=self.workdir,
            command=self.executable,
            basis_path=self.config["BASIS"],
            input_arguments=input_arguments,
        )
        
        return calc

    def setup_environment(self):
        """
        Configure l'environnement de calcul en créant les répertoires nécessaires
        et en définissant les variables d'environnement.
        
        Returns:
            bool: True si la configuration a réussi, False sinon
        """
        try:
            # Créer le répertoire de travail s'il n'existe pas
            if self.workdir and not os.path.exists(self.workdir):
                os.makedirs(self.workdir)
                                
            # Définir les variables d'environnement
            os.environ["OMP_NUM_THREADS"] = str(self.omp_threads)
            return True
        except Exception as e:
            print(f"Erreur lors de la configuration de l'environnement: {e}")
            return False







