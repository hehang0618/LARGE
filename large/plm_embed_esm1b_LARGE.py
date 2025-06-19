#We set the max length of protein 5000 here, because seq longer than 5000 may cause memory error.
import os
import warnings
warnings.filterwarnings('ignore')
import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap
import esm
from esm import FastaBatchedDataset
from tqdm import tqdm
import pickle as pk
import socket
from Bio import SeqIO
import numpy as np

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

class TruncatedFastaBatchedDataset(FastaBatchedDataset):
    """扩展FastaBatchedDataset类以支持序列截断"""
    def __init__(self, sequence_labels, sequence_strs, max_length=100000):
        super().__init__(sequence_labels, sequence_strs)
        self.max_length = max_length

    def __getitem__(self, idx):
        label, seq = super().__getitem__(idx)
        if len(seq) > self.max_length:
            return label, seq[:self.max_length]
        return label, seq

    @classmethod
    def from_file(cls, fasta_file, max_length=100000):
        sequence_labels, sequence_strs = [], []

        for record in SeqIO.parse(fasta_file, "fasta"):
            sequence_labels.append(record.id)
            sequence_strs.append(str(record.seq))

        return cls(sequence_labels, sequence_strs, max_length=max_length)

def run_esm_embedding(fasta_file, output_path, model_path, regression_path, num_workers, num_gpus, max_length=5000,model='slow'):
    port = find_free_port()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(num_gpus)
    toks_per_batch = 12290
    # 使用修改后的数据集类，设置最大序列长度为10000
    dataset = TruncatedFastaBatchedDataset.from_file(fasta_file, max_length=max_length)
    
    if model=="slow":
        batches = [[i] for i in range(len(dataset))]
    else:
        batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    
    url = f"tcp://localhost:{port}"
    torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)

    # Load local model files instead of downloading
    model_name = "esm2_t33_650M_UR50D"
    model_data = torch.load(model_path)
    regression_data = torch.load(regression_path) if regression_path else None

    fsdp_params = dict(
        mixed_precision=True,
        flatten_parameters=True,
        state_dict_device=torch.device("cpu"),
        cpu_offload=True,
    )

    with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
        model, vocab = esm.pretrained.load_model_and_alphabet_core(
            model_name, model_data, regression_data
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=vocab.get_batch_converter(), batch_sampler=batches, num_workers=int(num_workers)
        )
        model.eval()
        for name, child in model.named_children():
            if name == "layers":
                for layer_name, layer in child.named_children():
                    wrapped_layer = wrap(layer)
                    setattr(child, layer_name, wrapped_layer)
        model = wrap(model)

    start_memory = torch.cuda.memory_allocated()
    sequence_representations = []

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            toks = toks.cuda()
            # 这里保持12288的截断作为额外的安全措施
            toks = toks[:, :12288]
            results = model(toks, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]

            for i, label in enumerate(labels):
                truncate_len = min(12288, len(strs[i]))
                sequence_representations.append(
                    (label, token_representations[i, 1 : truncate_len + 1].mean(0).detach().cpu().numpy())
                )

            if batch_idx % 5 == 0:
                del results, token_representations
                torch.cuda.empty_cache()
    with open(output_path, "wb") as f:
        pk.dump(sequence_representations, f)
    return sequence_representations
