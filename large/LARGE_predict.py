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


def predict_categories(data_path, type, file_name, model_path, X_test, y_test, topn, categories, probability=0.5):
    """
    对指定的 categories 列表进行预测，每个类别从 model_path 下读取对应的权重和映射文件。
    文件命名规范：{category}_model_weights.pkl 和 {category}_code_to_name.csv
    """
    NUM_FEATURES = 2560

    for category in categories:
        print(f"Generating results for {category}")

        # 1. 读取 code_to_name 映射，并获取类别数量
        code_to_name_path = os.path.join(model_path, f'{category}_code_to_name.csv')
        if not os.path.exists(code_to_name_path):
            print(f"Warning: Mapping file {code_to_name_path} not found. Skipping category {category}.")
            continue
        
        df_map = pd.read_csv(code_to_name_path)
        # 确保列名为 'code' 和 'name'（与训练输出一致）
        if 'code' not in df_map.columns or 'name' not in df_map.columns:
            print(f"Warning: {code_to_name_path} must contain 'code' and 'name' columns. Skipping.")
            continue
        code_to_name = dict(zip(df_map['code'], df_map['name']))
        num_classes = len(code_to_name)

        # 2. 动态调整 topn
        current_topn = min(topn, num_classes)

        # 3. 模型权重文件路径
        model_weights_path = os.path.join(model_path, f'{category}_model_weights.pkl')
        if not os.path.exists(model_weights_path):
            print(f"Warning: Model weights {model_weights_path} not found. Skipping category {category}.")
            continue

        # 4. 准备 DataLoader
        dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
        test_loader = DataLoader(dataset=dataset, batch_size=5000)

        # 5. 设备与模型加载
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = MulticlassClassification(num_feature=NUM_FEATURES, num_class=num_classes)
        state_dict = torch.load(model_weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # 6. 预测
        y_pred_list = []
        y_pred_proba = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                _, pred_tags = torch.max(outputs, dim=1)
                y_pred_list.append(pred_tags.cpu().numpy())
                y_pred_proba.append(outputs.cpu().numpy())

        y_pred_list = np.concatenate(y_pred_list, axis=0)
        y_pred_proba = np.concatenate(y_pred_proba, axis=0)
        y_pred_proba = softmax(y_pred_proba, axis=1)

        # 7. 加载序列 ID 映射（从原 Cate 目录下的 CSV 获取）
        original_csv = os.path.join(data_path, 'Cate', f'{file_name}.csv')
        if not os.path.exists(original_csv):
            print(f"Error: {original_csv} not found. Cannot map sequence IDs.")
            continue
        df_orig = pd.read_csv(original_csv)
        id_to_name = dict(zip(df_orig['Seq ID'], df_orig['Seq Name']))

        # 8. 生成结果记录
        valid_records = []
        for i in range(len(y_pred_list)):
            seq_id_num = i + 1
            seq_name = id_to_name.get(seq_id_num, 'unknown')
            if seq_name == 'unknown':
                continue

            topn_indices = np.argsort(y_pred_proba[i])[-current_topn:][::-1]
            topn_preds = [code_to_name.get(idx, 'unknown') for idx in topn_indices]
            topn_probas = y_pred_proba[i][topn_indices]

            # 根据 probability 阈值决定 top1 预测
            if topn_probas[0] < probability:
                top1_pred = "UnARG"
                top1_proba = topn_probas[0]  # 仍保留原始概率，或可设为0？根据需求可调
                # 如果需要将低于阈值的所有 topn 都标记为 UnARG，可在此处理
            else:
                top1_pred = topn_preds[0]
                top1_proba = topn_probas[0]

            record = {
                "seq_id": seq_name,
                "pred": top1_pred,
                "max_proba": top1_proba
            }
            for j in range(current_topn):
                # 如果置信度低于阈值且当前是 top1，并且你希望 top1 显示 UnARG，则前几行已处理
                # 这里保持 topn_preds 和 topn_probas 不变（或者也可统一过滤）
                record[f"pred_{j+1}"] = topn_preds[j]
                record[f"proba_{j+1}"] = topn_probas[j]

            valid_records.append(record)

        # 保存结果，路径使用 os.path.abspath 美化
        output_file = os.path.join(data_path, '..', f'{file_name}_{category}_predicted_results.csv')
        pd.DataFrame(valid_records).round({
            'max_proba': 3,
            **{f'proba_{j+1}': 3 for j in range(current_topn)}
        }).to_csv(output_file, index=False)
        output_file = os.path.abspath(output_file)
        print(f"Saved predictions to {output_file}")

def predict_all_categories(results_folder, file_name, probability=0.5):
    print("Generating results for Comprehensive results")
    resistance_results = pd.read_csv(f'{results_folder}/../{file_name}_resistance_predicted_results.csv')
    genefamily_results = pd.read_csv(f'{results_folder}/../{file_name}_genefamily_predicted_results.csv')
    mechanism_results = pd.read_csv(f'{results_folder}/../{file_name}_mechanism_predicted_results.csv')
    
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
    
    comprehensive_output_file = f'{results_folder}/../{file_name}_comprehensive_predicted_results.csv'
    filtered.to_csv(comprehensive_output_file, index=False)
    
def predict(results_folder, file_name, num_gpus, model_path, topn, probability, categories=None):
    type = 'plmglm'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(num_gpus)
    X_test, y_test = save_data(results_folder, type, file_name)
    
    # 默认内置类别（与原始一致）
    if categories is None:
        categories = ['genefamily', 'resistance', 'mechanism']
    
    # 调用修改后的预测函数（原 train 函数）
    predict_categories(results_folder, type, file_name, model_path, X_test, y_test, topn, categories,probability)
    
    # 仅当预测了所有三个默认类别时才生成综合结果
    if set(categories) == {'genefamily', 'resistance', 'mechanism'}:
        predict_all_categories(results_folder, file_name, probability)