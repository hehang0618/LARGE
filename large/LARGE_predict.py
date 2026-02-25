import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle as pk
from scipy.special import softmax
import pkg_resources
import large

script_dir = list(large.__path__)[0]
datas_dir = script_dir + "/data/temp_data_LARGE/"
    
torch.manual_seed(12345)
class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        self.layer_1 = nn.Linear(num_feature, 2048)
        self.layer_2 = nn.Linear(2048, 1024)
        self.layer_3 = nn.Linear(1024, 512)
        self.layer_out = nn.Linear(512, num_class) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(2048)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.batchnorm3 = nn.BatchNorm1d(512)
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)        
        return x

class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def extract_ec_data_from_results_pkls(result_pkl_path, type):
    result_pkl_f = open(result_pkl_path, "rb")
    results = pk.load(result_pkl_f)
    prot_ids = results['all_prot_ids']
    input_embs = results['plm_embs'] 
    input_embs_new=input_embs 
    hidden_embs = results['glm_embs'] 
    hidden_embs_new=hidden_embs
    label_hidden_concat = np.concatenate((input_embs_new[:,:-1], hidden_embs_new), axis=1) #[1950,2560]
    return label_hidden_concat, prot_ids

def extract_ec_data_from_results_pkls_sequential(result_pkl, type):
    embs = []
    prot_ids = []
    embs,prot_ids = extract_ec_data_from_results_pkls(result_pkl, type)
    embs = np.array(embs)
    prot_ids = np.array(prot_ids)
    return  embs, prot_ids

def save_data(results_dir, type, file_name):
    pkl_f='inf_results.pkl'
    result_pkl=os.path.join(results_dir,pkl_f)
    X_test, prot_ids= extract_ec_data_from_results_pkls_sequential(result_pkl, type)
    y_test=np.zeros(len(X_test)).astype(int)
    return  X_test, y_test


def train(data_path, type, file_name, model_path,X_test,y_test,topn=1):
    NUM_FEATURES = 2560
    categories = [
        ('GeneFamily', 422, 'genefamily_model_weights.pth', 'genefamily_code_to_name.csv'), 
        ('Resistance', 27, 'resistance_model_weights.pth', 'resistance_code_to_name.csv'), 
        ('Mechanism', 12, 'mechanism_model_weights.pth', 'mechanism_code_to_name.csv')
    ]
    
    for cate, num_classes, model_weights, code_to_name_file in categories:
        print(f"Generating results for {cate}")
        
        if topn > num_classes:
            topn = num_classes
        
        code_to_name = {}
        # code_to_name_path = pkg_resources.resource_filename('large', f'data/temp_data_LARGE/{code_to_name_file}')
        code_to_name_path = datas_dir + code_to_name_file
        with open(code_to_name_path, 'r') as csvfile:
            df = pd.read_csv(csvfile)
            for _, row in df.iterrows():
                key = int(row[0])
                value = row[1]
                code_to_name[key] = value
        
        # Output file path
        plm_arg_output_file = f'{data_path}/../{file_name}_{cate}_predicted_results.csv'
        
        # Prepare dataset and loader
        file_nameset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
        test_loader = DataLoader(dataset=file_nameset, batch_size=5000)
        
        # Device setup
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Model setup
        model = MulticlassClassification(num_feature=NUM_FEATURES, num_class=num_classes)
        model.load_state_dict(torch.load(f'{model_path}/{model_weights}', map_location=device))
        model.to(device)
        
        # Prediction
        y_pred_list = []
        y_pred_proba = []
        with torch.no_grad():
            model.eval()
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                y_test_pred = model(X_batch)
                _, y_pred_tags = torch.max(y_test_pred, dim=1)
                y_pred_list.append(y_pred_tags.cpu().numpy())
                y_pred_proba.append(y_test_pred.cpu().numpy())
        
        y_pred_list = np.concatenate(y_pred_list, axis=0)
        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        y_pred_proba = np.concatenate(y_pred_proba, axis=0)
        
        # Load original data for seq names
        original_df = pd.read_csv(f'{data_path}/Cate/{file_name}.csv')
        id_to_name = dict(zip(original_df['Seq ID'], original_df['Seq Name']))
        
        y_pred_proba = softmax(y_pred_proba, axis=1)  
        valid_records = []
        a = np.array(list(code_to_name.keys()))
        
        for i in range(len(y_pred_list)):
            seq_name = id_to_name.get(i+1, 'unknown')
            if seq_name == 'unknown':
                continue
            
            topn_indices = np.argsort(y_pred_proba[i])[-topn:][::-1]
            topn_preds = [code_to_name.get(a[idx], 'unknown') for idx in topn_indices]
            topn_probas = y_pred_proba[i][topn_indices]
            
            top1_pred = topn_preds[0]
            top1_proba = topn_probas[0]
            
            record = {
                "seq_id": seq_name,
                "pred": top1_pred,
                "max_proba": top1_proba
            }
            for j in range(topn):
                record[f"pred_{j+1}"] = topn_preds[j]
                record[f"proba_{j+1}"] = topn_probas[j]
            valid_records.append(record)
        pd.set_option('display.float_format', '{:.3f}'.format)
        valid_results_df = pd.DataFrame(valid_records).round({ 
            'max_proba': 3, 
            **{f'proba_{j+1}': 3 for j in range(topn)} 
        })
        
        valid_results_df.to_csv(plm_arg_output_file, index=False)

def predict_all_categories(results_folder, file_name, probability=0.5):
    print("Generating results for Comprehensive results")
    resistance_results = pd.read_csv(f'{results_folder}/../{file_name}_Resistance_predicted_results.csv')
    genefamily_results = pd.read_csv(f'{results_folder}/../{file_name}_GeneFamily_predicted_results.csv')
    mechanism_results = pd.read_csv(f'{results_folder}/../{file_name}_Mechanism_predicted_results.csv')
    
    resistance_results = resistance_results.rename(
        columns={'pred': 'resistance', 'max_proba': 'resistance_prob'}
    )
    genefamily_results = genefamily_results.rename(
        columns={'pred': 'gene_family', 'max_proba': 'gene_family_prob'}
    )
    mechanism_results = mechanism_results.rename(
        columns={'pred': 'mechanism', 'max_proba': 'mechanism_prob'}
    )
    merged = pd.merge(
        resistance_results[['seq_id', 'resistance', 'resistance_prob']], 
        genefamily_results[['seq_id', 'gene_family', 'gene_family_prob']],
        on='seq_id',
        how='left'
    )
    merged = pd.merge(
        merged,
        mechanism_results[['seq_id', 'mechanism', 'mechanism_prob']],
        on='seq_id',
        how='left'
    )
    
    filtered = merged[
        (merged['resistance'] != 'UnARG') & 
        (merged['resistance_prob'] >= probability)
    ].copy()
    
    filtered.loc[
        filtered['gene_family'] == 'UnARG', 
        ['gene_family', 'gene_family_prob']
    ] = ['Unknown', 0]
    
    filtered.loc[
        filtered['mechanism'] == 'UnARG', 
        ['mechanism', 'mechanism_prob']
    ] = ['Unknown', 0]
    
    filtered['gene_family'] = filtered['gene_family'].fillna('Unknown')
    filtered['gene_family_prob'] = filtered['gene_family_prob'].fillna(0)
    filtered['mechanism'] = filtered['mechanism'].fillna('Unknown')
    filtered['mechanism_prob'] = filtered['mechanism_prob'].fillna(0)
    
    comprehensive_output_file = f'{results_folder}/../{file_name}_Comprehensive_predicted_results.csv'
    filtered.to_csv(comprehensive_output_file, index=False)
    
def predict(results_folder, file_name, num_gpus, model_path,topn,probability):
    type = 'plmglm'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(num_gpus)
    X_test,y_test=save_data(results_folder, type, file_name)
    train(results_folder, type, file_name, model_path,X_test,y_test,topn)
    predict_all_categories(results_folder, file_name,probability)
