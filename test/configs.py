import os
import sys



DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to sys.path to allow imports 
# from the main project
sys.path.append(os.path.dirname(DIRECTORY))



