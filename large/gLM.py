from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import *
import torch
from torch import nn
from torch.nn import MSELoss
from typing import Optional, Tuple, Union
import torch.utils.checkpoint
from transformers.utils import ModelOutput


class MultiPredMaskedLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    masked_lm_loss: Optional[torch.FloatTensor] = None
    probs_loss: Optional[torch.FloatTensor] = None
    most_likely_lm_loss: Optional[torch.FloatTensor] = None
    avg_var: Optional[torch.FloatTensor] = None
    logits_all_preds: torch.FloatTensor = None
    logits_closest: torch.FloatTensor = None
    logits_most_likely: torch.FloatTensor = None
    last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None


class gLMMultiHead(nn.Module):
    """Roberta Head for masked language modeling with num_pred predictions."""

    def __init__(self, config):
        super().__init__()
        self.num_pred = config.num_pred
        self.num_pc = config.num_pc
        self.dense = nn.Linear(config.hidden_size, config.num_pc*self.num_pred)
        self.hidden_size = config.hidden_size
        self.predict_probs = config.predict_probs
        if config.predict_probs:
            self.dense_prob = nn.Linear(config.hidden_size, self.num_pred)


    def forward(self, features, **kwargs):
        #[batch, max_seq_len, hidden_size].
        x = self.dense(features)
        x_shape = list(x.shape)
        # [batch, max_seq_len, num_pred, hidden_size].
        x = x.view(*x_shape[:-1],self.num_pred, self.num_pc)
        if self.predict_probs:
            # [batch, max_seq_len, num_pred].
            probs = self.dense_prob(features)
        else:
            probs = None
        return x, probs

class gLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()        
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.num_pc)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-06)
    def forward(self, features, **kwargs):
        x = self.dense(features)
        return x

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_embeds'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_embeds, and attention_mask for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}

def symmetrize(x):
    """Make layer symmetric in final two dimensions, used for contact prediction."""
    return x + x.transpose(-1, -2)


class ContactPredictionHead(nn.Module):
    """modified fair esm's contact prediction head"""
    """Performs symmetrization and computes a logistic regression on the output features"""

    def __init__(
        self,
        in_features: int,
        bias=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.regression = nn.Linear(in_features, 1, bias)
        self.activation = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, tokens, attentions):
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # features: B x C x T x T
        attentions = attentions.to(
            self.regression.weight.device
        )  # attentions always float32, may need to convert to float16
        attentions = symmetrize(attentions)
        attentions = attentions.permute(0, 2, 3, 1)
        return attentions
class gLM_base(RobertaModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.num_pred = config.num_pred
        self.predict_probs = config.predict_probs
        self.lm_head = gLMMultiHead(config)
        self.contact_head = ContactPredictionHead(config.num_hidden_layers*config.num_attention_heads)
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder
 
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        masked_tokens: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor],  MultiPredMaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        
        return MultiPredMaskedLMOutput(
            last_hidden_state = sequence_output
        )
class gLM(RobertaModel):
    
    def __init__(self, config):
        super().__init__(config) 
        self.roberta = gLM_base(config)
        # linear resizing 
        self.dense = nn.Linear(config.emb_dim, config.hidden_size)
        self.output_attentions = config.output_attentions
        # The LM head weights require special treatment only when they are tied with the word embeddings
        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        masked_tokens: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor],  MultiPredMaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = self.dense(inputs_embeds)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=self.output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels = labels,
            masked_tokens = masked_tokens,
        )
        
        return outputs

