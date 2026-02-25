import os
import warnings
import numpy as np
import pickle as pk
import importlib.util
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
import large

def batch_data(emb_f, output_dir):
    # 步骤1: 收集所有蛋白质ID和嵌入向量
    embs = []
    all_prot_ids = []
    for key, val in emb_f:
        prot_id = int(key.split(" ")[0])  # 直接转换为整数
        all_prot_ids.append(prot_id)
        embs.append(val)
    
    # 转换为numpy数组
    embs = np.array(embs, dtype=np.float16)
    
    # 加载归一化参数
    spec = importlib.util.find_spec('large')
    script_dir = list(large.__path__)[0]
    embs_mean_file = os.path.join(script_dir, "data/temp_data_LARGE/embs_mean.pkl")
    embs_std_file = os.path.join(script_dir, "data/temp_data_LARGE/embs_std.pkl")
    with open(embs_mean_file, "rb") as f:
        embs_mean = pk.load(f)
    with open(embs_std_file, "rb") as f:
        embs_std = pk.load(f)
    
    normalized_embs = (embs - embs_mean) / embs_std
    
    num_samples = len(normalized_embs)
    if num_samples < 100:
        num_zeros_to_add = 100 - num_samples
        zero_embeds = np.zeros((num_zeros_to_add, normalized_embs.shape[1]), dtype=np.float16)
        extended_embs = np.vstack((normalized_embs, zero_embeds))
    else:
        extended_embs = normalized_embs
    
    # 执行PCA降维
    PCA_LABEL = PCA(n_components=99, whiten=True)
    
    # 使用扩展后的嵌入进行PCA拟合
    if num_samples < 100:
        all_labels_extended = PCA_LABEL.fit_transform(extended_embs)
        # 只取原始样本的PCA结果
        all_labels = all_labels_extended[:num_samples]
    else:
        all_labels = PCA_LABEL.fit_transform(extended_embs)  # 这里extended_embs=normalized_embs
    
    # 创建prot_id到标签的映射
    prot_id_to_label = dict(zip(all_prot_ids, all_labels))
    
    # 步骤2: 准备数据
    prot_emb_pairs = []
    for prot_id, emb in zip(all_prot_ids, normalized_embs):  # 注意这里使用原始归一化嵌入
        prot_emb_pairs.append((prot_id, emb, prot_id_to_label[prot_id]))
    
    # 按prot_id排序
    prot_emb_pairs.sort(key=lambda x: x[0])
    
    # 步骤3: 分批处理
    batch_size = 1
    batches = []
    total_prots = len(prot_emb_pairs)
    for start_idx in range(0, total_prots, batch_size):
        end_idx = min(start_idx + batch_size, total_prots)
        batch_data = prot_emb_pairs[start_idx:end_idx]
        actual_size = len(batch_data)
        
        # 初始化数组
        prot_ids = np.zeros(batch_size, dtype=int)
        embeds = np.zeros((batch_size, 1281), dtype=np.float16)  # 1280 + 1 (orientation)
        label_embeds = np.zeros((batch_size, 100), dtype=np.float16)  # 99 + 1 (orientation)
        attention_mask = np.zeros(batch_size, dtype=int)
        #生成对角矩阵让其关注本身
        # attention_mask = np.zeros((batch_size, batch_size), dtype=int)
        # for i in range(actual_size):
            # attention_mask[i, i] = 1        

        for i, (prot_id, emb, label) in enumerate(batch_data):
            prot_ids[i] = prot_id
            embeds[i] = np.append(emb, 0.5)  # 添加方向维度（默认为正链0.5）
            label_embeds[i] = np.append(label, 0.5)  # 添加方向维度
        
        # 对于不足batch_size的部分，prot_ids填充-1
        if actual_size < batch_size:
            prot_ids[actual_size:] = -1
        
        # 添加到batch列表
        batches.append({
            'prot_ids': prot_ids,
            'embeds': embeds,
            'label_embeds': label_embeds,
            'attention_mask': attention_mask
        })
    
    print(f"Total processed: {total_prots} proteins")
    
    # 保存批次数据
    with open(os.path.join(output_dir, "new_seq.pkl"), "wb") as f:
        pk.dump(batches, f)

