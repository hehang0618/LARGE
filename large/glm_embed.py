import os
import torch
from torch import nn
from large.gLM import *
from transformers import RobertaConfig
from tqdm import tqdm
import os
import numpy as np
import argparse
import pathlib
import datetime
import pickle as pk
import pdb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def infer(data_path, model,output_path, device, id_dict,num_workers):
    HIDDEN_SIZE = 1280
    B_SIZE = 1 # batch size
    HALF = True
    EMB_DIM = 1281
    NUM_PC_LABEL = 100
    f_list = os.listdir(data_path)
    test_pkls=[]
    for pkl_f in f_list:
        if pkl_f == "prot_index_dict.pkl": 
            id_dict = pk.load(open(os.path.join(data_path,pkl_f), "rb"))
        else:
            test_pkls.append(str(os.path.join(data_path,pkl_f)))
    
    torch.cuda.empty_cache()
    scaler = torch.cuda.amp.GradScaler()
        
    for pkl_f in test_pkls:
        input_embs = []
        hidden_embs = []
        all_prot_ids = []
        pickle_file =  open(pkl_f, 'rb')
        dataset = pk.load(pickle_file)
        loader = torch.utils.data.DataLoader(dataset, batch_size =B_SIZE, shuffle=False, drop_last=False)
        for batch in tqdm(loader, total=len(loader)):            
            inputs_embeds= batch['embeds'].type(torch.FloatTensor)        
            attention_mask = batch['attention_mask'].type(torch.FloatTensor)
            mask = torch.zeros(attention_mask.shape) #nothing is masked
            masked_tokens = (mask==1) & (attention_mask != 0)
            masked_tokens = torch.unsqueeze(masked_tokens, -1)
            masked_tokens = masked_tokens.to(device)
            inputs_embeds = inputs_embeds.to(device)
            inputs_embeds  = torch.where(masked_tokens, -1.0, inputs_embeds)
            attention_mask = attention_mask.to(device)
            labels = batch['label_embeds'].type(torch.FloatTensor)
            labels = labels.to(device)
            input_embs.append(inputs_embeds.cpu().detach().numpy())
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    # call model
                    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels = labels, masked_tokens = masked_tokens, output_attentions = False)
                    last_hidden_states = outputs.last_hidden_state
                    hidden_embs.append(last_hidden_states.cpu().detach().numpy().astype(np.float16))
                    prot_ids = batch['prot_ids']
                    all_prot_ids.append(prot_ids)
        input_embs = np.concatenate(np.concatenate(input_embs, axis = 0), axis = 0) # remove batch dimension
        hidden_embs =  np.concatenate(np.concatenate(hidden_embs, axis = 0), axis = 0) # remove batch dimension
        all_prot_ids = np.concatenate(np.concatenate(all_prot_ids, axis = 0), axis = 0)
        
        results_filename = output_path+"inf_results.pkl"
        results = {}
        results['plm_embs'] = input_embs
        results['glm_embs'] = hidden_embs
        results['all_prot_ids'] = all_prot_ids
        results_f = open(results_filename, "wb")
        pk.dump(results, results_f)
        results_f.close()

        pickle_file.close()
    return None


def glm_embed(data_path, model_path, output_path,num_gpus,num_workers):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(num_gpus)
    num_pred = 4
    max_seq_length = 1 
    num_attention_heads = 10
    num_hidden_layers= 19
    pos_emb = "relative_key_query"
    pred_probs = True
    HIDDEN_SIZE = 1280
    EMB_DIM = 1281
    NUM_PC_LABEL = 100
    e = datetime.datetime.now()
    results_dir = output_path
    # populate config 
    config = RobertaConfig(
        max_position_embedding = 1,
        hidden_size = HIDDEN_SIZE,
        num_attention_heads = num_attention_heads,
        type_vocab_size = 1,
        tie_word_embeddings = False,
        num_hidden_layers = num_hidden_layers,
        num_pc = NUM_PC_LABEL, 
        num_pred = num_pred,
        predict_probs = pred_probs,
        emb_dim = EMB_DIM,
        output_attentions=True,
        output_hidden_states=True,
        position_embedding_type = pos_emb,
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model =  gLM(config)
    model.load_state_dict(torch.load(model_path, map_location=device),strict=False)
    model.eval()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    id_dict = None
    with torch.no_grad():
        infer(data_path,model,output_path=results_dir,device=device, id_dict=id_dict,num_workers=num_workers)