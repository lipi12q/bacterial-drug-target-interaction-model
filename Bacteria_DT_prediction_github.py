
import pandas as pd
import numpy as np



#import DeepPurpose
import DeepPurpose.DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *

#load model
path = 'drug_target_model\chembl_cnn_train_fungi_1wan'
net = models.model_pretrained(path_dir = path)
drug_encoding, target_encoding = 'CNN', 'CNN'
 # 1. load test data
df = pd.read_csv('test.csv')

smiles_list = df['Smiles'].tolist()
sequence_list = df['Sequence'].tolist()
original_scores = df['score'].tolist()  # score = 0
X_pred = data_process(smiles_list, sequence_list, original_scores, 
                                drug_encoding, target_encoding, 
                                split_method='no_split')
y_pred = net.predict(X_pred)
# 4. results
result_df = df.copy()
result_df['score'] = y_pred  # replace initial socres

# save
output_csv = "predicted_results.csv"  
result_df.to_csv(output_csv, index=False)
print(f"complete, save results to {output_csv}")

