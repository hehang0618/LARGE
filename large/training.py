import os
import warnings
warnings.filterwarnings('ignore')
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pickle as pk
from collections import Counter
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from large.plm_embed_esm1b_LARGE import run_esm_embedding
from large.batch_data_LARGE import batch_data
from large.glm_embed import glm_embed

torch.manual_seed(12345)


def replace_nonstandard_amino_acids(sequence, replacements={'J': 'L', '*': ''}):
    """替换非标准氨基酸，与 readfasta.py 中的逻辑一致"""
    return ''.join(replacements.get(aa, aa) for aa in sequence)


def process_category(category, csv_path, output_dir, input_file_name):
    """
    生成 _num.csv 和 _mapping.csv
    与原有 training.py 中的逻辑一致，无改动
    """
    df = pd.read_csv(csv_path)
    cate_df = df[['Seq ID', 'Seq Name', category]]
    cate_csv = os.path.join(output_dir, f'{input_file_name}_{category}.csv')
    cate_df.to_csv(cate_csv, index=False)
    
    df = pd.read_csv(cate_csv)
    codes, uniques = pd.factorize(df[category])
    df[f'{category} Code'] = codes
    df.loc[df[category].isnull(), f'{category} Code'] = 0
    
    mapping_df = pd.DataFrame({
        'code': range(len(uniques)),
        'name': uniques
    })
    mapping_csv = os.path.join(output_dir, f'{input_file_name}_{category}_mapping.csv')
    mapping_df.to_csv(mapping_csv, index=False)
    
    encoded_csv = os.path.join(output_dir, f'{input_file_name}_{category}_num.csv')
    df = df.drop(columns=[category])
    df.to_csv(encoded_csv, index=False)


# ====================== 以下为原有的训练模型定义（保持不变） ======================
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
        
    def __len__(self):
        return len(self.X_data)


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc


def extract_ec_data_from_results_pkls(result_pkl_path, type):
    result_pkl_f = open(result_pkl_path, "rb")
    results = pk.load(result_pkl_f)
    prot_ids = results['all_prot_ids']
    input_embs = results['plm_embs'] 
    hidden_embs = results['glm_embs'] 
    label_hidden_concat = np.concatenate((input_embs[:,:-1], hidden_embs), axis=1)
    return label_hidden_concat, prot_ids


def extract_ec_data_from_results_pkls_sequential(result_pkl, type):
    embs, prot_ids = extract_ec_data_from_results_pkls(result_pkl, type)
    return np.array(embs), np.array(prot_ids)


def save_data(results_dir, type, file_name):
    """从 inf_results.pkl 中提取特征矩阵，返回 X 和占位 y（全零）"""
    pkl_f = 'inf_results.pkl'
    result_pkl = os.path.join(results_dir, pkl_f)
    X_test, prot_ids = extract_ec_data_from_results_pkls_sequential(result_pkl, type)
    y_test = np.zeros(len(X_test)).astype(int)
    return X_test, y_test

def training_data(X_test, y_test, results_dir, filename, category):
    df = pd.read_csv(f'{results_dir}/Cate/{filename}_{category}_num.csv')
    key = category + ' Code'
    y_train = df[key].values
    
    X_train = X_test[:len(y_train)]
    
    nfeatures = 2560
    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    
    entire_y = y_train
    class_count = [0] * (max([int(i) for i in entire_y]) + 1)
    y_counter = Counter(entire_y)
    for key, val in y_counter.items():
        class_count[int(key)] = val
    
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    train_weights = class_weights[torch.tensor([int(i) for i in y_train])].numpy()
    weighted_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(y_train), replacement=True)
    
    EPOCHS = 650
    BATCH_SIZE = 5000
    LEARNING_RATE = 0.0001
    NUM_FEATURES = nfeatures
    NUM_CLASSES = len(class_count)
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              sampler=weighted_sampler)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MulticlassClassification(num_feature=NUM_FEATURES, num_class=NUM_CLASSES)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for e in range(1, EPOCHS + 1):
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
    
    model_filename = category + '_' + 'model_weights' + ".pkl"
    model_file_path = f'{results_dir}/../{model_filename}'
    torch.save(model.state_dict(), model_file_path)


# ====================== 主训练入口 ======================
def run_training(args):
    """
    完整训练流程（适配数字重编号，兼容 batch_data_LARGE.py 的 int 转换）
    """
    input_file = args.training_protein
    input_file_name = os.path.basename(input_file)
    category = args.training_category.lower()
    tempdir = os.path.join(args.output_dir, "temp/")
    os.makedirs(tempdir, exist_ok=True)
    
    # ---------- 1. 读取原始 FASTA，保留完整 header ----------
    records = list(SeqIO.parse(input_file, "fasta"))
    seq_infos = []
    for record in records:
        original_header = record.description.strip() if record.description else record.id
        modified_seq = replace_nonstandard_amino_acids(str(record.seq))
        seq_infos.append({
            'original_header': original_header,
            'seq': modified_seq,
            'length': len(modified_seq)
        })
    
    # ---------- 2. 按长度降序排序（模仿 process_fasta）----------
    seq_infos.sort(key=lambda x: x['length'], reverse=True)
    
    # ---------- 3. 建立映射：数字ID → 原始header ----------
    numeric_ids = list(range(1, len(seq_infos) + 1))
    header_to_id = {info['original_header']: idx for idx, info in enumerate(seq_infos, start=1)}
    
    # 保存映射表（可选）
    mapping_df = pd.DataFrame({
        'numeric_id': numeric_ids,
        'original_header': [info['original_header'] for info in seq_infos]
    })
    mapping_df.to_csv(os.path.join(tempdir, 'header_mapping.csv'), index=False)
    
    # ---------- 4. 生成 _rep.fasta（ID为数字，description为原始header）----------
    rep_file = os.path.join(tempdir, f"{input_file_name}_rep.fasta")
    rep_records = []
    for idx, info in enumerate(seq_infos, start=1):
        rep_records.append(SeqRecord(
            Seq(info['seq']),
            id=str(idx),
            description=info['original_header']
        ))
    SeqIO.write(rep_records, rep_file, "fasta")
    
    # ---------- 5. 读取标签CSV，映射为数字ID，按ID顺序构建标签列表 ----------
    df_tn = pd.read_csv(args.training_labels)
    if 'seqname' not in df_tn.columns or 'seqlabel' not in df_tn.columns:
        print("Error: -tn CSV must contain columns 'seqname' and 'seqlabel'.")
        sys.exit(1)
    
    # 检查一一对应关系
    original_headers_set = set(header_to_id.keys())
    label_headers_set = set(df_tn['seqname'].astype(str))
    if original_headers_set != label_headers_set:
        print("Error: -tp and -tn must have exactly the same seqnames (one-to-one correspondence).")
        sys.exit(1)
    
    # 构建数字ID → 标签的字典
    id_to_label = {}
    for _, row in df_tn.iterrows():
        orig = str(row['seqname'])
        label = str(row['seqlabel'])
        numeric_id = header_to_id[orig]
        id_to_label[numeric_id] = label
    
    # 按数字ID顺序（1..N）生成标签列表
    labels_in_order = [id_to_label[i] for i in range(1, len(seq_infos) + 1)]
    
    # ---------- 6. 生成 sequences.csv（用于 process_category）----------
    sequences_csv = os.path.join(tempdir, f"{input_file_name}_sequences.csv")
    df_seq = pd.DataFrame({
        'Seq ID': [str(i) for i in range(1, len(seq_infos) + 1)],
        'Seq Name': [str(i) for i in range(1, len(seq_infos) + 1)],  # 也可用原始header，但此处用数字更简单
        category: labels_in_order
    })
    df_seq.to_csv(sequences_csv, index=False)
    
    # ---------- 7. 调用 process_category 生成 _num.csv ----------
    cate_dir = os.path.join(tempdir, 'Cate')
    os.makedirs(cate_dir, exist_ok=True)
    process_category(category, sequences_csv, cate_dir, input_file_name)
    
    # ---------- 8. 嵌入（与原有流程相同）----------
    print("--------------------------------------------------")
    print("Step 1: Embedding sequences...")
    pklfile = os.path.join(tempdir, f"LARGE_seq_cate_fasta_{input_file_name}.esm.embs.pkl")
    batch_dir = os.path.join(tempdir, f"batched_data_{input_file_name}")
    os.makedirs(batch_dir, exist_ok=True)
    
    regression_path = os.path.join(args.model_dir, "esm2_t33_650M_UR50D-contact-regression.pt")
    model_path_esm = os.path.join(args.model_dir, "esm2_t33_650M_UR50D.pt")
    run_esm_embedding(
        fasta_file=rep_file,
        output_path=pklfile,
        model_path=model_path_esm,
        regression_path=regression_path,
        num_workers=args.cpu,
        num_gpus=args.gpu,
        max_length=args.length,
        model=args.model
    )
    batch_data(pk.load(open(pklfile, "rb")), batch_dir)
    
    print("--------------------------------------------------")
    print("Step 2: Run GLM-META...")
    glm_model_path = os.path.join(args.model_dir, "pytorch_model.bin")
    glm_embed(
        data_path=batch_dir,
        model_path=glm_model_path,
        output_path=tempdir,
        num_gpus=args.gpu,
        num_workers=args.cpu
    )
    
    # ---------- 9. 提取特征矩阵，并加载真实标签 ----------
    print("--------------------------------------------------")
    print("Step 3: Preparing training data...")
    X_test, _ = save_data(tempdir, "plmglm", input_file_name)  # 返回特征，忽略假标签
    
    # 从 _num.csv 中按 Seq ID 顺序提取编码后的标签
    num_csv = os.path.join(cate_dir, f'{input_file_name}_{category}_num.csv')
    df_num = pd.read_csv(num_csv)
    # 确保按 Seq ID 排序（应该是数字，但保险起见转为 int 后排序）
    df_num['Seq ID'] = df_num['Seq ID'].astype(int)
    df_num = df_num.sort_values('Seq ID')
    y_true = df_num[f'{category} Code'].values.astype(int)
    
    # 检查特征与标签数量一致
    if len(X_test) != len(y_true):
        print(f"Error: Feature count {len(X_test)} does not match label count {len(y_true)}.")
        sys.exit(1)
    
    # ---------- 10. 训练 ----------
    print("Step 4: Training the classifier...")
    training_data(X_test, y_true, tempdir, input_file_name, category)
    
    # 输出类别映射文件
    mapping_csv = os.path.join(cate_dir, f'{input_file_name}_{category}_mapping.csv')
    if os.path.exists(mapping_csv):
        mapping_df = pd.read_csv(mapping_csv)
        code_to_name_df = mapping_df[['code', 'name']]
        code_to_name_path = os.path.join(args.output_dir, f'{category}_code_to_name.csv')
        code_to_name_df.to_csv(code_to_name_path, index=False)
        print(f"\n✅ Training completed successfully!")
        print(f"   Category mapping: {code_to_name_path}")
        model_weight_path = os.path.join(args.output_dir, f'{category}_model_weights.pkl')
        print(f"   Model weights: {model_weight_path}")
    else:
        print("Warning: code_to_name.csv was not generated.")