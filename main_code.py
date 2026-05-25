import logging
import warnings
import os
import pandas as pd
import numpy as np
from functions.main_func import obtain_features
from functions.eval_f import evaluate

# to remove warnings
logger = logging.getLogger("radiomics.glcm")
logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

# to access to data
tables_path = r"/home/natalia/Documents/BUS-20250226T233013Z-001/BUS"
dataset_path = os.path.join(tables_path, "dataset_info.xlsx")

db = pd.read_excel(dataset_path, engine='openpyxl')

# paths to imgs and masks
data_path = 'BUS/all_images'
mask_path = 'BUS/all_masks'

# features
fclasses = ['firstorder', 'shape2D', 'glcm', 'glrlm', 'gldm'] 
args = {'label':1, 'all_features':False, 'feature_classes':fclasses} 

# model arguments
nfolds = 5 
nfolds_cv = 10
seed = 42
cols = {'classes': 'label',
        'features': 'features',
        'group': 'patient_id'
       }

params = {
    'penalty': ['l2'], 
    'C': np.logspace(0, 4, 5), 
    'solver': ['liblinear']  
}


db_ext = obtain_features(db, 'patient_id', data_path, mask_path, args)
evaluate(db_ext, cols, 'LR_custom', params, nfolds, nfolds_cv, seed, seed);

