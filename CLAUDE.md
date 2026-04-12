# CLAUDE.md - DeMonNanoPy

## Projet

DeMonNanoPy est un wrapper Python autour du moteur de chimie quantique **deMonNano** (DFTB).
Il permet de construire des fichiers d'entrée, lancer des calculs et parser les résultats depuis Python.

- **Statut** : Alpha (v0.1.0)
- **Python** : >= 3.10
- **Dépendances** : numpy, ase (Atomic Simulation Environment)
- **Licence** : MIT

## Architecture

Trois couches principales dans `deMonPy/` :

1. **Input** (`input.py`) : construit `deMon.inp` depuis des dictionnaires Python imbriqués
2. **Execution** (`deMonNano.py`) : orchestre écriture, exécution du binaire, lecture des résultats
3. **Output** (`output.py`) : parse `deMon.out` et `deMon.mol` (énergies, géométries, trajectoires)

Couche supplémentaire :

4. **Modules** (`modules/`) : workflows de haut niveau (optimisation, MD, PTMC) qui enchaînent des calculs

Fichiers de support :
- `profile.py` : exécution subprocess + décorateurs `@assert_flags` / `@exclude_flags`
- `molden.py` : lecture de fichiers XYZ étendu
- `encoder.py` : sérialisation JSON pour numpy/ase
- `__init__.py` : registre des modules disponibles avec leurs paramètres par défaut

## Conventions de code

### Nommage

- Les classes suivent le nommage du domaine, pas PEP8 strict : `deMonNano`, `Module_DeMonNano`, `write_input`, `read_output`
- Les modules internes sont préfixés par `_` : `_relax_geometry`, `_ptmc`, `_dyn`
- Les clés de configuration sont en **MAJUSCULES** : `DEMON_EXECUTABLE`, `DEMON_PARAMETERS`, `BASIS`, `DFTB`, `SCC`
- Les flags internes sont en **minuscules** : `"opt"`, `"md"`, `"dftb"`, `"ci"`, `"traj"`
- Ne pas renommer les classes/fonctions existantes pour les rendre PEP8 : le nommage actuel est intentionnel

### Style

- Langue du code : **anglais** (variables, classes, docstrings)
- Langue des commentaires : anglais ou français selon le contexte existant
- Indentation : 4 espaces, pas de tabs
- Imports : `os`, `sys`, `numpy as np` en haut de fichier ; imports internes `deMonPy.*` après
- Les docstrings utilisent le format **Google style** (Args, Returns, Raises)
- Pas de type hints systématiques dans le code existant ; ne pas en ajouter sauf si explicitement demandé

### Configuration par dictionnaires

La configuration suit une structure imbriquée stricte :

```python
{
    "DEMON_EXECUTABLE": "path/to/deMon.x",
    "BASIS": {"PTYPE": "...", "SKFILE": "..."},
    "DEMON_PARAMETERS": {"ACTIVE": { ... }},
    "DEMON_MODULE": {"ACTIVE": { ... }},
}
```

- Les paramètres actifs sont sous la clé `"ACTIVE"`
- Les flags sont dérivés automatiquement des clés de `ACTIVE` (en minuscules)
- Les décorateurs `@assert_flags` / `@exclude_flags` contrôlent conditionnellement l'exécution des méthodes d'écriture et de lecture

### Pattern module

Chaque module de workflow (`modules/`) :
- Hérite de `modules` (classe de base dans `module.py`)
- Reçoit le calculateur via `context`
- Implémente une méthode `forward()` comme point d'entrée principal
- Met à jour les paramètres via `self.update_parameters()`
- Lance le calcul via `self.context.calculate()`

Pour ajouter un nouveau module :
1. Créer un fichier dans `deMonPy/modules/`
2. Hériter de `modules`
3. Implémenter `forward(**kwds)`
4. L'enregistrer dans `deMonPy/__init__.py` sous `available_modules`

## Commandes

### Installation

```bash
python -m venv .env
source .env/bin/activate
pip install -e .
```

### Tests

```bash
# Depuis la racine du projet
pytest

# Un fichier spécifique
pytest test/test_demon.py

# Un test spécifique
pytest test/test_demon.py::TestBasicUsage::test_single_point
```

Les tests nécessitent :
- Le binaire deMonNano installé (chemin dans `test/configs.py`)
- Les fichiers de base/Slater-Koster dans `test/basis/`
- Les tests valident des valeurs numériques exactes (énergies)

### Exécution des exemples

```bash
python exemple/exemple_opt.py
python exemple/exemple_md.py
python exemple/exemple_ptmc.py
```

## Fichiers deMonNano

- `deMon.inp` : fichier d'entrée généré (ne pas éditer manuellement)
- `deMon.out` : sortie principale parsée pour les énergies, états, fréquences
- `deMon.mol` : géométries en format XYZ étendu (entrée, sortie, trajectoire)
- `deMon.freq` : résultats d'analyse vibrationnelle

## Points d'attention

- **Typo connue** : le paramètre s'appelle `properies` (pas `properties`) dans l'API existante. Ne pas corriger sans demande explicite car cela casserait la compatibilité.
- **Stubs** : les parsers PTMC, NEB, debug, et certaines parties fréquence sont des placeholders. L'ASE calculator (`ase_calculator.py`) est quasi vide.
- **Exécution** : le code utilise `os.system()` par défaut (`system=True`). Le mode subprocess (`system=False`) utilise `subprocess.Popen`.
- **Flags** : le système de flags pilote toute la logique conditionnelle. Toute nouvelle section input/output doit s'intégrer via ce mécanisme.

