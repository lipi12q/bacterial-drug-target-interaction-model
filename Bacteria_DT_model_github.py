import DeepPurpose.DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
import pandas as pd
import numpy as np

final_data = pd.read_csv('data/bacteria_final_dataset_traing.csv')
X_drug = final_data['Smiles']
X_target = final_data['Target sequence']
y = final_data['Standard Value']

drug_encoding = 'CNN'
target_encoding = 'CNN'
train, val, test = data_process(list(X_drug), list(X_target), list(y),
                                drug_encoding, target_encoding,
                                split_method='HTS',frac=[0.8,0.1,0.1])

config = generate_config(drug_encoding = drug_encoding,
                         target_encoding = target_encoding,
                         cls_hidden_dims = [1024,1024,512],
                         train_epoch = 10,
                         LR = 0.0001,
                         batch_size = 256,
                         cnn_drug_filters = [32,64,96],
                         cnn_target_filters = [32,64,96],
                         cnn_drug_kernels = [4,6,8],
                         cnn_target_kernels = [4,8,12],
                        )
model = models.model_initialize(**config)

model.train(train, val, test)
model.save_model('drug_target_model/chembl_cnn_train_bacteria_10wan')