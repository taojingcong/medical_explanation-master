
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from copy import deepcopy
import math
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                                  BertConfig, BertLMHeadModel, BertTokenizer, GPT2LMHeadModel, 
                                  GPT2Model, BertForMaskedLM, BertForMultipleChoice,
                                  BartForConditionalGeneration, BartModel, GPT2PreTrainedModel, 
                                  GPT2DoubleHeadsModel, AlbertPreTrainedModel)
from transformers.modeling_bert import BertOnlyMLMHead, BertModel
from transformers.modeling_albert import AlbertModel
from transformers.modeling_roberta import RobertaModel
from transformers.modeling_utils import SequenceSummary

def calc_mse_loss(mrc_outputs, explanation_outputs, mask=None):
    if mask is not None:
        # mask has False at padding_idx
        sel_mask = mask[:, :, None].expand_as(explanation_outputs).bool()
        s_logits_slct = torch.masked_select(explanation_outputs, sel_mask)
        t_logits_slct = torch.masked_select(mrc_outputs, sel_mask)
    else:
        t_logits_slct = mrc_outputs
        s_logits_slct = explanation_outputs
    return F.mse_loss(s_logits_slct, t_logits_slct)

def calc_kl_div(mrc_outputs, explanation_outputs, temperature=1.0):
    loss_kl = F.kl_div(
            input=F.log_softmax(mrc_outputs / temperature, dim=-1),
            target=F.softmax(explanation_outputs / temperature, dim=-1),
            reduction="batchmean",
    ) * (temperature ** 2)
                
    return loss_kl

class AttentionMerge(nn.Module):
    """
    H (B, L, hidden_size) => h (B, hidden_size)
    """
    def __init__(self, input_size, attention_size, dropout_prob):
        super(AttentionMerge, self).__init__()
        self.attention_size = attention_size
        self.hidden_layer = nn.Linear(input_size, self.attention_size)
        self.query_ = nn.Parameter(torch.Tensor(self.attention_size, 1))
        self.dropout = nn.Dropout(dropout_prob)

        self.query_.data.normal_(mean=0.0, std=0.02)

    def forward(self, values, mask=None):
        """
        (b, l, h) -> (b, h)
        """
        if mask is None:
            mask = torch.zeros_like(values)
            # mask = mask.data.normal_(mean=0.0, std=0.02)
        else:
            mask = (1 - mask.unsqueeze(-1).type(torch.float)) * -1000.

        keys = self.hidden_layer(values)
        keys = torch.tanh(keys)
        query_var = torch.var(self.query_)
        # (b, l, h) + (h, 1) -> (b, l, 1)
        attention_probs = keys @ self.query_ / math.sqrt(self.attention_size * query_var)
        # attention_probs = keys @ self.query_ / math.sqrt(self.attention_size)

        attention_probs = F.softmax(attention_probs * mask, dim=1)
        attention_probs = self.dropout(attention_probs)

        context = torch.sum(attention_probs + values, dim=1)
        return context


class BertLMAddMrcHeadModel(BertModel):
    '''
    Bert Model with a language modeling head on top for CLM fine-tuning. 
    CLM stands for Causal Language Modeling in which a given word 
    is trained based only on the previous words and not using the masking technique.
    This model is a PyTorch torch.nn.Module sub-class. 
    Use it as a regular PyTorch Module and 
    refer to the PyTorch documentation for all matter related to general usage and behavio
    '''
    def __init__(self, config):
        # config.update({'is_decoder': True})
        super().__init__(config)
        # assert config.is_decoder, "If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True`."
        self.num_choices = 5
        config.update({'is_decoder': True})
        self.bert = BertModel(config)
        config_mrc = deepcopy(config)
        config_mrc.update({'is_decoder': False})  
        self.mrc_bert = BertModel(config_mrc)
        self.cls = BertOnlyMLMHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.tg_classifier = nn.Linear(config.hidden_size, self.num_choices)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        input_ids_mrc=None,
        attention_mask_mrc=None,
        token_type_ids_mrc=None,
        position_ids_mrc=None,
        head_mask_mrc=None,
        inputs_embeds_mrc=None,
        labels=None,
        labels_mrc=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        encoder_hidden_states_mrc=None,
        encoder_attention_mask_mrc=None,
        output_attentions_mrc=None,
        output_hidden_states_mrc=None,
        **kwargs
    ):
        '''
        - [CLS] context [SEP] choice_1 [SEP]
        - [CLS] context [SEP] choice_2 [SEP]
        - [CLS] context [SEP] choice_3 [SEP]
        - [CLS] context [SEP] choice_4 [SEP]
        - [CLS] context [SEP] choice_5 [SEP]
        '''
        assert ((labels is not None) and (labels_mrc is not None)) or ((labels is None) and (labels_mrc is None))
        #num_choices = input_ids_mrc.shape[1] if input_ids_mrc is not None else inputs_embeds_mrc.shape[1]
        num_choices = self.num_choices
        outputs = self.bert(
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
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        explanation_classification_scores = self.tg_classifier(outputs[1])
        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        input_ids_mrc = input_ids_mrc.view(-1, input_ids_mrc.size(-1)) if input_ids_mrc is not None else None
        attention_mask_mrc = attention_mask_mrc.view(-1, attention_mask_mrc.size(-1)) if attention_mask_mrc is not None else None
        token_type_ids_mrc = token_type_ids_mrc.view(-1, token_type_ids_mrc.size(-1)) if token_type_ids_mrc is not None else None
        position_ids_mrc = position_ids_mrc.view(-1, position_ids_mrc.size(-1)) if position_ids_mrc is not None else None
        inputs_embeds_mrc = (
            inputs_embeds_mrc.view(-1, inputs_embeds_mrc.size(-2), inputs_embeds_mrc.size(-1))
            if inputs_embeds_mrc is not None
            else None
        )
        mrc_outputs = self.mrc_bert(
            input_ids_mrc,
            attention_mask=attention_mask_mrc,
            token_type_ids=token_type_ids_mrc,
            position_ids=position_ids_mrc,
            head_mask=head_mask_mrc,
            inputs_embeds=inputs_embeds_mrc,
            encoder_hidden_states=encoder_hidden_states_mrc,
            encoder_attention_mask=encoder_attention_mask_mrc,
            output_attentions=output_attentions_mrc,
            output_hidden_states=output_hidden_states_mrc,
        )
        if labels_mrc is not None:
            pooled_output_mrc = mrc_outputs[1]
            pooled_output_mrc = self.dropout(pooled_output_mrc)
            logits_mrc = self.classifier(pooled_output_mrc)
            reshaped_logits_mrc = logits_mrc.view(-1, num_choices)
            
            mse_loss = calc_kl_div(reshaped_logits_mrc, explanation_classification_scores)

            outputs_mrc = (reshaped_logits_mrc,) + mrc_outputs[2:]  # add hidden states and attention if they are here
            
            outputs = outputs + outputs_mrc 

        if labels_mrc is not None:
            loss_fct_mrc = CrossEntropyLoss()
            loss_mrc = loss_fct_mrc(reshaped_logits_mrc, labels_mrc)
            outputs =  (loss_mrc,)  + outputs

            outputs = (mse_loss, ) + outputs
        # mse mrc lm
        return outputs # (loss_mrc, ),  (ltr_lm_loss, ), prediction_scores, (hidden_states), (attentions), reshaped_logits, (hidden_states), (attentions)


class GPT2LMAddMrcHead(GPT2DoubleHeadsModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_choices = 5
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        config.summary_type = "cls_index"
        config.num_labels = 1
        self.multiple_choice_head = SequenceSummary(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        input_ids_mrc=None,
        attention_mask_mrc=None,
        token_type_ids_mrc=None,
        position_ids_mrc=None,
        head_mask_mrc=None,
        inputs_embeds_mrc=None,
        labels=None,
        labels_mrc=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        encoder_hidden_states_mrc=None,
        encoder_attention_mask_mrc=None,
        output_attentions_mrc=None,
        output_hidden_states_mrc=None,
        **kwargs
    ):
        '''
        - [CLS] context [SEP] choice_1 [SEP]
        - [CLS] context [SEP] choice_2 [SEP]
        - [CLS] context [SEP] choice_3 [SEP]
        - [CLS] context [SEP] choice_4 [SEP]
        - [CLS] context [SEP] choice_5 [SEP]
        '''
        assert ((labels is not None) and (labels_mrc is not None)) or ((labels is None) and (labels_mrc is None))
        #num_choices = input_ids_mrc.shape[1] if input_ids_mrc is not None else inputs_embeds_mrc.shape[1]
        num_choices = self.num_choices
        
        # explanation_classification_scores = self.tg_classifier(outputs[1])
        # outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,

        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        # explanation_classification_scores = self.tg_classifier(outputs[1])
        # outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        input_ids_mrc = input_ids_mrc.view(-1, input_ids_mrc.size(-1)) if input_ids_mrc is not None else None
        attention_mask_mrc = attention_mask_mrc.view(-1,) if attention_mask_mrc is not None else None
        token_type_ids_mrc = token_type_ids_mrc.view(-1, token_type_ids_mrc.size(-1)) if token_type_ids_mrc is not None else None
        position_ids_mrc = position_ids_mrc.view(-1, position_ids_mrc.size(-1)) if position_ids_mrc is not None else None
        inputs_embeds_mrc = (
            inputs_embeds_mrc.view(-1, inputs_embeds_mrc.size(-2), inputs_embeds_mrc.size(-1))
            if inputs_embeds_mrc is not None
            else None
        )
        mrc_outputs = self.transformer(
            input_ids_mrc,
            past=None, 
            attention_mask=None,
            token_type_ids=None,
            position_ids=position_ids_mrc,
            head_mask=head_mask_mrc,
            inputs_embeds=inputs_embeds_mrc,
            output_attentions=output_attentions_mrc,
            output_hidden_states=output_hidden_states_mrc,
        )
        hidden_states = mrc_outputs[0]
        if labels_mrc is not None:
            # mc_token_ids
            mc_logits = self.multiple_choice_head(hidden_states, attention_mask_mrc).squeeze(-1)
            # print("#######: ", mc_logits.shape)
            reshaped_logits_mrc = mc_logits.view(-1, num_choices)
            # mse_loss = calc_kl_div(reshaped_logits_mrc, explanation_classification_scores)
            # print("#######: ", reshaped_logits_mrc.shape)
            outputs_mrc = (reshaped_logits_mrc,) + mrc_outputs[2:]  # add hidden states and attention if they are here
            # print("len: ", len(outputs))
            outputs = outputs + outputs_mrc 

        if labels_mrc is not None:
            loss_fct_mrc = CrossEntropyLoss()
            loss_mrc = loss_fct_mrc(reshaped_logits_mrc, labels_mrc)
            outputs =  (loss_mrc,)  + outputs

            outputs = (loss_mrc, ) + outputs
        # mse mrc lm
        # print(outputs[5].shape)
        return outputs # (loss_mrc, ),  (ltr_lm_loss, ), prediction_scores, (hidden_states), (attentions), reshaped_logits, (hidden_states), (attentions)


class AlbertLMAddMrcHeadModel(AlbertPreTrainedModel):
    '''
    Bert Model with a language modeling head on top for CLM fine-tuning. 
    CLM stands for Causal Language Modeling in which a given word 
    is trained based only on the previous words and not using the masking technique.
    This model is a PyTorch torch.nn.Module sub-class. 
    Use it as a regular PyTorch Module and 
    refer to the PyTorch documentation for all matter related to general usage and behavio
    '''
    def __init__(self, config):
        # config.update({'is_decoder': True})
        super().__init__(config)
        # assert config.is_decoder, "If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True`."
        self.num_choices = 5
        # config_lm.update({'is_decoder': True})
        # self.bert = BertModel(config_lm)
        # self.config_lm = config_lm
        self.albert = AlbertModel(config)
        # self.cls = BertOnlyMLMHead(config_lm)
        # config_mrc.hidden_dropout_prob = 0.1
        self.att_merge = AttentionMerge(config.hidden_size, attention_size=1024, dropout_prob=0.1)
        # self.dropout = nn.Dropout(0.1)
        self.scorer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, 1)
        )
        # self.classifier = nn.Linear(config.hidden_size, 1)
        # self.tg_classifier = nn.Linear(config_lm.hidden_size, self.num_choices)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        input_ids_mrc=None,
        attention_mask_mrc=None,
        token_type_ids_mrc=None,
        position_ids_mrc=None,
        head_mask_mrc=None,
        inputs_embeds_mrc=None,
        labels=None,
        labels_mrc=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        encoder_hidden_states_mrc=None,
        encoder_attention_mask_mrc=None,
        output_attentions_mrc=None,
        output_hidden_states_mrc=None,
        **kwargs
    ):
        '''
        - [CLS] context [SEP] choice_1 [SEP]
        - [CLS] context [SEP] choice_2 [SEP]
        - [CLS] context [SEP] choice_3 [SEP]
        - [CLS] context [SEP] choice_4 [SEP]
        - [CLS] context [SEP] choice_5 [SEP]
        '''
        assert ((labels is not None) and (labels_mrc is not None)) or ((labels is None) and (labels_mrc is None))
        #num_choices = input_ids_mrc.shape[1] if input_ids_mrc is not None else inputs_embeds_mrc.shape[1]
        num_choices = self.num_choices
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     position_ids=position_ids,
        #     head_mask=head_mask,
        #     inputs_embeds=inputs_embeds,
        #     encoder_hidden_states=encoder_hidden_states,
        #     encoder_attention_mask=encoder_attention_mask,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        # )

        # sequence_output = outputs[0]
        # prediction_scores = self.cls(sequence_output)
        # explanation_classification_scores = self.tg_classifier(outputs[1])
        # outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        
        # if labels is not None:
        #     # we are doing next-token prediction; shift prediction scores and input ids by one
        #     prediction_scores = prediction_scores[:, :-1, :].contiguous()
        #     labels = labels[:, 1:].contiguous()
        #     loss_fct = CrossEntropyLoss()
        #     ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config_lm.vocab_size), labels.view(-1))
        #     outputs = (ltr_lm_loss,) + outputs

        input_ids_mrc = input_ids_mrc.view(-1, input_ids_mrc.size(-1)) if input_ids_mrc is not None else None
        attention_mask_mrc = attention_mask_mrc.view(-1, attention_mask_mrc.size(-1)) if attention_mask_mrc is not None else None
        token_type_ids_mrc = token_type_ids_mrc.view(-1, token_type_ids_mrc.size(-1)) if token_type_ids_mrc is not None else None
        position_ids_mrc = position_ids_mrc.view(-1, position_ids_mrc.size(-1)) if position_ids_mrc is not None else None
        inputs_embeds_mrc = (
            inputs_embeds_mrc.view(-1, inputs_embeds_mrc.size(-2), inputs_embeds_mrc.size(-1))
            if inputs_embeds_mrc is not None
            else None
        )
        # self,
        # input_ids=None,
        # attention_mask=None,
        # token_type_ids=None,
        # position_ids=None,
        # head_mask=None,
        # inputs_embeds=None,
        # output_attentions=None,
        # output_hidden_states=None,
        mrc_outputs = self.albert(
            input_ids_mrc,
            attention_mask=attention_mask_mrc,
            token_type_ids=token_type_ids_mrc,
            # position_ids=position_ids_mrc,
            # head_mask=head_mask_mrc,
            # inputs_embeds=inputs_embeds_mrc,
            # encoder_hidden_states=encoder_hidden_states_mrc,
            # encoder_attention_mask=encoder_attention_mask_mrc,
            # output_attentions=output_attentions_mrc,
            # output_hidden_states=output_hidden_states_mrc,
        )
        
        if labels_mrc is not None:
            # pooled_output_mrc = mrc_outputs[1]
            # pooled_output_mrc = self.dropout(pooled_output_mrc)
            # logits_mrc = self.classifier(pooled_output_mrc)
            h12 = self.att_merge(mrc_outputs[0], attention_mask_mrc)
            logits_mrc = self.scorer(h12)
            reshaped_logits_mrc = logits_mrc.view(-1, num_choices)
            
            # mse_loss = calc_kl_div(reshaped_logits_mrc, explanation_classification_scores)

            outputs_mrc = (reshaped_logits_mrc,) + mrc_outputs[2:]  # add hidden states and attention if they are here
            
            outputs = outputs_mrc 

        if labels_mrc is not None:
            loss_fct_mrc = CrossEntropyLoss()
            loss_mrc = loss_fct_mrc(reshaped_logits_mrc, labels_mrc)
            outputs =  (loss_mrc,)  + outputs
            outputs =  (loss_mrc,)  + outputs
            outputs = (loss_mrc, ) + outputs
            outputs = (loss_mrc, ) + outputs
        # mse mrc lm
        return outputs # (loss_mrc, ),  (ltr_lm_loss, ), prediction_scores, (hidden_states), (attentions), reshaped_logits, (hidden_states), (attentions)


class RobertaLMAddMrcHeadModel(RobertaModel):
    '''
    Bert Model with a language modeling head on top for CLM fine-tuning. 
    CLM stands for Causal Language Modeling in which a given word 
    is trained based only on the previous words and not using the masking technique.
    This model is a PyTorch torch.nn.Module sub-class. 
    Use it as a regular PyTorch Module and 
    refer to the PyTorch documentation for all matter related to general usage and behavio
    '''
    def __init__(self, config):
        # config.update({'is_decoder': True})
        super().__init__(config)
        # assert config.is_decoder, "If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True`."
        self.num_choices = 5
        config.update({'is_decoder': True})
        self.bert = BertModel(config)
        config_mrc = deepcopy(config)
        config_mrc.update({'is_decoder': False})  
        self.mrc_bert = RobertaModel(config_mrc)
        self.cls = BertOnlyMLMHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.tg_classifier = nn.Linear(config.hidden_size, self.num_choices)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        input_ids_mrc=None,
        attention_mask_mrc=None,
        token_type_ids_mrc=None,
        position_ids_mrc=None,
        head_mask_mrc=None,
        inputs_embeds_mrc=None,
        labels=None,
        labels_mrc=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        encoder_hidden_states_mrc=None,
        encoder_attention_mask_mrc=None,
        output_attentions_mrc=None,
        output_hidden_states_mrc=None,
        **kwargs
    ):
        '''
        - [CLS] context [SEP] choice_1 [SEP]
        - [CLS] context [SEP] choice_2 [SEP]
        - [CLS] context [SEP] choice_3 [SEP]
        - [CLS] context [SEP] choice_4 [SEP]
        - [CLS] context [SEP] choice_5 [SEP]
        '''
        assert ((labels is not None) and (labels_mrc is not None)) or ((labels is None) and (labels_mrc is None))
        #num_choices = input_ids_mrc.shape[1] if input_ids_mrc is not None else inputs_embeds_mrc.shape[1]
        num_choices = self.num_choices
        outputs = self.bert(
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
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        explanation_classification_scores = self.tg_classifier(outputs[1])
        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        input_ids_mrc = input_ids_mrc.view(-1, input_ids_mrc.size(-1)) if input_ids_mrc is not None else None
        attention_mask_mrc = attention_mask_mrc.view(-1, attention_mask_mrc.size(-1)) if attention_mask_mrc is not None else None
        token_type_ids_mrc = token_type_ids_mrc.view(-1, token_type_ids_mrc.size(-1)) if token_type_ids_mrc is not None else None
        position_ids_mrc = position_ids_mrc.view(-1, position_ids_mrc.size(-1)) if position_ids_mrc is not None else None
        inputs_embeds_mrc = (
            inputs_embeds_mrc.view(-1, inputs_embeds_mrc.size(-2), inputs_embeds_mrc.size(-1))
            if inputs_embeds_mrc is not None
            else None
        )
        # self,
        # input_ids=None,
        # attention_mask=None,
        # token_type_ids=None,
        # position_ids=None,
        # head_mask=None,
        # inputs_embeds=None,
        # output_attentions=None,
        # output_hidden_states=None,
        mrc_outputs = self.mrc_bert(
            input_ids_mrc,
            attention_mask=attention_mask_mrc,
            token_type_ids=token_type_ids_mrc,
            position_ids=position_ids_mrc,
            head_mask=head_mask_mrc,
            inputs_embeds=inputs_embeds_mrc,
            # encoder_hidden_states=encoder_hidden_states_mrc,
            # encoder_attention_mask=encoder_attention_mask_mrc,
            output_attentions=output_attentions_mrc,
            output_hidden_states=output_hidden_states_mrc,
        )
        if labels_mrc is not None:
            pooled_output_mrc = mrc_outputs[1]
            pooled_output_mrc = self.dropout(pooled_output_mrc)
            logits_mrc = self.classifier(pooled_output_mrc)
            reshaped_logits_mrc = logits_mrc.view(-1, num_choices)
            
            mse_loss = calc_mse_loss(reshaped_logits_mrc, explanation_classification_scores)

            outputs_mrc = (reshaped_logits_mrc,) + mrc_outputs[2:]  # add hidden states and attention if they are here
            
            outputs = outputs + outputs_mrc 

        if labels_mrc is not None:
            loss_fct_mrc = CrossEntropyLoss()
            loss_mrc = loss_fct_mrc(reshaped_logits_mrc, labels_mrc)
            outputs =  (loss_mrc,)  + outputs

            outputs = (mse_loss, ) + outputs
        # mse mrc lm
        return outputs # (loss_mrc, ),  (ltr_lm_loss, ), prediction_scores, (hidden_states), (attentions), reshaped_logits, (hidden_states), (attentions)


class BartLMAddMrcHeadModel(BartModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_choices = 5
        self.model = BartModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.tg_classifier = nn.Linear(config.hidden_size, self.num_choices)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        input_ids_mrc=None,
        attention_mask_mrc=None,
        token_type_ids_mrc=None,
        position_ids_mrc=None,
        head_mask_mrc=None,
        inputs_embeds_mrc=None,
        labels=None,
        labels_mrc=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        encoder_hidden_states_mrc=None,
        encoder_attention_mask_mrc=None,
        output_attentions_mrc=None,
        output_hidden_states_mrc=None,
        **kwargs
    ):
        '''
        - [CLS] context [SEP] choice_1 [SEP]
        - [CLS] context [SEP] choice_2 [SEP]
        - [CLS] context [SEP] choice_3 [SEP]
        - [CLS] context [SEP] choice_4 [SEP]
        - [CLS] context [SEP] choice_5 [SEP]
        '''
        assert ((labels is not None) and (labels_mrc is not None)) or ((labels is None) and (labels_mrc is None))
        #num_choices = input_ids_mrc.shape[1] if input_ids_mrc is not None else inputs_embeds_mrc.shape[1]
        num_choices = self.num_choices
        outputs = self.bert(
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
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        explanation_classification_scores = self.tg_classifier(outputs[1])
        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        input_ids_mrc = input_ids_mrc.view(-1, input_ids_mrc.size(-1)) if input_ids_mrc is not None else None
        attention_mask_mrc = attention_mask_mrc.view(-1, attention_mask_mrc.size(-1)) if attention_mask_mrc is not None else None
        token_type_ids_mrc = token_type_ids_mrc.view(-1, token_type_ids_mrc.size(-1)) if token_type_ids_mrc is not None else None
        position_ids_mrc = position_ids_mrc.view(-1, position_ids_mrc.size(-1)) if position_ids_mrc is not None else None
        inputs_embeds_mrc = (
            inputs_embeds_mrc.view(-1, inputs_embeds_mrc.size(-2), inputs_embeds_mrc.size(-1))
            if inputs_embeds_mrc is not None
            else None
        )
        mrc_outputs = self.mrc_bert(
            input_ids_mrc,
            attention_mask=attention_mask_mrc,
            token_type_ids=token_type_ids_mrc,
            position_ids=position_ids_mrc,
            head_mask=head_mask_mrc,
            inputs_embeds=inputs_embeds_mrc,
            encoder_hidden_states=encoder_hidden_states_mrc,
            encoder_attention_mask=encoder_attention_mask_mrc,
            output_attentions=output_attentions_mrc,
            output_hidden_states=output_hidden_states_mrc,
        )
        if labels_mrc is not None:
            pooled_output_mrc = mrc_outputs[1]
            pooled_output_mrc = self.dropout(pooled_output_mrc)
            logits_mrc = self.classifier(pooled_output_mrc)
            reshaped_logits_mrc = logits_mrc.view(-1, num_choices)
            
            mse_loss = calc_kl_div(reshaped_logits_mrc, explanation_classification_scores)

            outputs_mrc = (reshaped_logits_mrc,) + mrc_outputs[2:]  # add hidden states and attention if they are here
            
            outputs = outputs + outputs_mrc 

        if labels_mrc is not None:
            loss_fct_mrc = CrossEntropyLoss()
            loss_mrc = loss_fct_mrc(reshaped_logits_mrc, labels_mrc)
            outputs =  (loss_mrc,)  + outputs

            outputs = (mse_loss, ) + outputs
        # mse mrc lm
        return outputs # (loss_mrc, ),  (ltr_lm_loss, ), prediction_scores, (hidden_states), (attentions), reshaped_logits, (hidden_states), (attentions)


class AlbertOneLMAddMrcHeadModel(AlbertPreTrainedModel):
    def __init__(self, config):
        # config.update({'is_decoder': True})
        super().__init__(config)
        # assert config.is_decoder, "If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True`."
        self.num_choices = 5
        config.update({'is_decoder': True})
        self.albert = AlbertModel(config)
        self.cls = BertOnlyMLMHead(config)
        # self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, 1)#in_features=4096
        
        config.summary_type = "cls_index"
        config.summary_proj_to_labels = True
        config.summary_use_proj = True
        config.summary_first_dropout = 0.15
        config.num_labels = 1
        self.config = config
        self.multiple_choice_head = SequenceSummary(config)
        # self.att_merge = AttentionMerge(config.hidden_size, attention_size=1024, dropout_prob=0.1)
        # self.dropout = nn.Dropout(0.1)
        # self.scorer = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Linear(config.hidden_size, 1)
        # )
        config.num_labels = 5
        self.explanation_head = SequenceSummary(config)
        # self.att_merge_2 = AttentionMerge(config.hidden_size, attention_size=1024, dropout_prob=0.1)
        # self.scorer_2 = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Linear(config.hidden_size, self.num_choices)
        # )
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        input_ids_mrc=None,
        attention_mask_mrc=None,
        token_type_ids_mrc=None,
        position_ids_mrc=None,
        head_mask_mrc=None,
        inputs_embeds_mrc=None,
        labels=None,
        labels_mrc=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        encoder_hidden_states_mrc=None,
        encoder_attention_mask_mrc=None,
        output_attentions_mrc=None,
        output_hidden_states_mrc=None,
        **kwargs
    ):
        '''
        - [CLS] context [SEP] choice_1 [SEP]
        - [CLS] context [SEP] choice_2 [SEP]
        - [CLS] context [SEP] choice_3 [SEP]
        - [CLS] context [SEP] choice_4 [SEP]
        - [CLS] context [SEP] choice_5 [SEP]
        '''
        assert ((labels is not None) and (labels_mrc is not None)) or ((labels is None) and (labels_mrc is None))
        #num_choices = input_ids_mrc.shape[1] if input_ids_mrc is not None else inputs_embeds_mrc.shape[1]
        num_choices = self.num_choices
        outputs = self.albert(#输入的只有input_ids和attention_mask
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )#输出为[[16,58,4096],[16,4096]]
        pre = torch.sum(attention_mask, axis=-1) 
        attention_length = pre - torch.ones_like(pre)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)#16*58*30000
        # explanation_classification_scores = self.tg_classifier(outputs[1])

        explanation_classification_scores = self.explanation_head(sequence_output, attention_length).squeeze(-1)
        # h_exp = self.att_merge_2(outputs[0], attention_mask)
        # explanation_classification_scores = self.scorer_2(h_exp)
            
        explanation_classification_scores = explanation_classification_scores.view(-1, num_choices)#16*5
        # print("explanation_classification_scores: ", explanation_classification_scores.shape)
        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        #用prediction_scores和labels来计算损失?
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()#去掉最后一个词,变成[16,57,30000]
            labels = labels[:, 1:].contiguous()#去掉开头的一个单词
            loss_fct = CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs
        #展开
        input_ids_mrc = input_ids_mrc.view(-1, input_ids_mrc.size(-1)) if input_ids_mrc is not None else None
        attention_mask_mrc = attention_mask_mrc.view(-1, attention_mask_mrc.size(-1)) if attention_mask_mrc is not None else None
        token_type_ids_mrc = token_type_ids_mrc.view(-1, token_type_ids_mrc.size(-1)) if token_type_ids_mrc is not None else None
        position_ids_mrc = position_ids_mrc.view(-1, position_ids_mrc.size(-1)) if position_ids_mrc is not None else None
        inputs_embeds_mrc = (#none
            inputs_embeds_mrc.view(-1, inputs_embeds_mrc.size(-2), inputs_embeds_mrc.size(-1))
            if inputs_embeds_mrc is not None
            else None
        )
        # self,
        # input_ids=None,
        # attention_mask=None,
        # token_type_ids=None,
        # position_ids=None,
        # head_mask=None,
        # inputs_embeds=None,
        # output_attentions=None,
        # output_hidden_states=None,
        mrc_outputs = self.albert(
            input_ids_mrc,#options_input的序列[80,41]
            attention_mask=attention_mask_mrc,#[80,41]
            token_type_ids=token_type_ids_mrc,#[80,41]
            position_ids=position_ids_mrc,#none
            head_mask=head_mask_mrc,#none
            inputs_embeds=inputs_embeds_mrc,
            # encoder_hidden_states=encoder_hidden_states_mrc,
            # encoder_attention_mask=encoder_attention_mask_mrc,
            output_attentions=output_attentions_mrc,
            output_hidden_states=output_hidden_states_mrc,
        )#[[80,41,4096],[80,4096]]
        hidden_states = mrc_outputs[0]
        pre = torch.sum(attention_mask_mrc, axis=-1) 
        attention_mrc_length = pre - torch.ones_like(pre)
        # print(attention_mrc_length)
        if labels_mrc is not None:
            # pooled_output_mrc = mrc_outputs[1]
            # pooled_output_mrc = self.dropout(pooled_output_mrc)
            # logits_mrc = self.classifier(pooled_output_mrc)
            # reshaped_logits_mrc = logits_mrc.view(-1, num_choices)
            # print(hidden_states.shape)
            mc_logits = self.multiple_choice_head(hidden_states, attention_mrc_length).squeeze(-1)

            # h12 = self.att_merge(mrc_outputs[0], attention_mask_mrc)
            # mc_logits = self.scorer(h12)
            

            # print("mc_logits: ", mc_logits.shape)
            reshaped_logits_mrc = mc_logits.view(-1, num_choices)
            # print("reshaped_logits_mrc: ", reshaped_logits_mrc.shape)
            outputs_mrc = (reshaped_logits_mrc,) + mrc_outputs[2:]  # add hidden states and attention if they are here
            # print("len: ", len(outputs))
            mse_loss = calc_kl_div(reshaped_logits_mrc, explanation_classification_scores, temperature=2.0)
            
            outputs = outputs + outputs_mrc 

        if labels_mrc is not None:
            loss_fct_mrc = CrossEntropyLoss()
            loss_mrc = loss_fct_mrc(reshaped_logits_mrc, labels_mrc)
            outputs =  (loss_mrc,)  + outputs
            loss_fct_explanation = CrossEntropyLoss()
            mse_loss += loss_fct_explanation(explanation_classification_scores, labels_mrc)
            outputs = (mse_loss, ) + outputs
        # mse mrc lm
        return outputs # (loss_mrc, ),  (ltr_lm_loss, ), prediction_scores, (hidden_states), (attentions), reshaped_logits, (hidden_states), (attentions)
