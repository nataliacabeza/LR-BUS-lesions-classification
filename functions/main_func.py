import re
import cv2
from radiomics import featureextractor
import SimpleITK as sitk
import numpy as np
import os
import six

# to correct de ID format
def correct_patient_id(df):
    rgx = r'([A-Z]{1})(-+)?([A-Z]{2})(-+)?([A-Z]{0,1})(-+)?(\d{4})'
    df['patient_id'] = df.patient_id.str.replace(rgx, r'\1-\3-\5-\7', regex=True)
    df['patient_id'] = df.patient_id.str.replace(r'--+', '-', regex=True)

    rgx = r'[A-Z]{1}-[A-Z]{2}-\d{4}|[A-Z]{1}-[A-Z]{2}-[A-Z]{1}-\d{4}'
    df['patient_id'] = df.patient_id.str.findall(rgx, flags=re.IGNORECASE).str[0]

    return df

# to convert array to image
def array_to_image(arr, dtype):
    arr = sitk.GetImageFromArray(arr) 
    arr = sitk.Cast(arr, dtype) 
    return arr


def mask_redim(arr, mask): 
    h, w = arr.shape 
    if arr.shape != mask.shape: 
        mask = cv2.resize(mask, (w, h)) 
    return (mask>1).astype(np.uint8) 

# to extract features
def feature_extractor(img, mask, label=1, all_features=True, feature_classes=None):
    extractor = featureextractor.RadiomicsFeatureExtractor()

    if ~all_features: 
        extractor.disableAllFeatures() 
        for feature_class in feature_classes:
            extractor.enableFeatureClassByName(feature_class)

    features_dict = extractor.execute(img, mask, label) 
    features = np.array([]) 

    for key, value in six.iteritems(features_dict): 
        if key.startswith("original_"): 
            features = np.append(features, features_dict[key]) 

    return features

# to obtain features
def obtain_features(db, file_col, data_path, mask_path, args):
    db = db.copy()  
    db['features'] = [[] for _ in range(len(db))]  

    for i in range(len(db)): 
        file = str(db[file_col].loc[i])  

       
        data = cv2.imread(os.path.join(data_path, file + '.bmp'), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(mask_path, file + '.tif'), cv2.IMREAD_GRAYSCALE)

        
        mask = mask_redim(data, mask) 
        mask = array_to_image(mask.T, sitk.sitkInt8) 

        
        features = feature_extractor(array_to_image(data.T, sitk.sitkFloat64),  
                                     mask,
                                     label=args['label'],
                                     all_features=args['all_features'],
                                     feature_classes=args['feature_classes'])

        db.at[i, 'features'] = features  
        

    return db  

