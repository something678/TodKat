# -*- coding: utf8 -*-


from torch.nn import CrossEntropyLoss
from transformers import AlbertModel
from transformers.models.albert.modeling_albert import AlbertMLMHead
from transformers.models.albert.modeling_albert import AlbertPreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers.utils import logging

from utilities.vaeEncoder import VaeEncoder
from utilities.vaeDecoder import VaeDecoder
from utilities.attention import Attention
from torch.distributions.normal import Normal
import torch
import torch.autograd


logger = logging.get_logger(__name__)



class TopicDrivenMaskedLM(AlbertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, param_encoder_pi_size=512, param_hidden_layer_size=1024, param_topic_embedsize=1024):
        super().__init__(config)

        self.albert = AlbertModel(config, add_pooling_layer=False)
        self.predictions = AlbertMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

        self.albert_hiddensize = self.albert.encoder.albert_layer_groups[-1].albert_layers[-1].ffn_output.out_features

        self.vae_decoder = VaeDecoder(
            param_dim_topic=param_topic_embedsize,
            # param_dim_vocab=param_vocabulary_size,
            param_dim_vocab=self.albert_hiddensize,
            param_dim_wncs=self.albert_hiddensize,
            param_dim_hidden=param_hidden_layer_size)
        # param_dim_encoder: encoder hiddenlayer size (\pi)
        # param_dim_vocab: vocabulary size
        #                  has switched to albert_hiddenlayer_size
        # param_dim_hidden: hidden semantic size
        self.vae_encoder = VaeEncoder(
            param_dim_encoder=param_encoder_pi_size,
            # param_dim_vocab=param_vocabulary_size,
            param_dim_vocab=self.albert_hiddensize,
            param_dim_hidden=param_hidden_layer_size)
        
        self.hidden_layer_size = param_hidden_layer_size
        
        self.att = Attention(self.albert_hiddensize, self.albert_hiddensize)

        # self.standard_normal = Normal(
        #     loc=torch.zeros(param_hidden_layer_size),
        #     scale=1.0)
    
    def sample_an_zs(self, param_mu, param_sigma_log_pow, device):
        '''
        generate one sample of z
        =====================
        params:
        ----------
        param_mu: mu
        param_sigma_log_pow: sigma_log

        return:
        ----------
        z_s: one sample of z
        '''
        # create an empty tensor with specific size
        # eps = norma

        # # ---------- changed in the ALBERT front version
        # if self.on_cuda:
        #     eps = torch.autograd.Variable(
        #         self.standard_normal.sample()).cuda()
        # else:
        #     eps = torch.autograd.Variable(
        #         self.standard_normal.sample())

        # eps = torch.autograd.Variable(self.standard_normal.sample())
        # eps = eps.cuda()
        # # # ---------- changed in the ALBERT front version
        # sigma = torch.sqrt(torch.exp(param_sigma_log_pow))
        # z_s = param_mu + sigma * eps

        # eps = torch.normal(mean=torch.zeros(param_hidden_layer_size, device=device),
        # eps = torch.normal(mean=torch.zeros(self.hidden_layer_size),
        #                    std=1.0,
        #                    device=device)
        eps = torch.normal(mean=torch.zeros(self.hidden_layer_size, device=device),
                           std=1.0)
        sigma = torch.sqrt(torch.exp(param_sigma_log_pow))
        z_s = param_mu + sigma * eps
        return z_s

    def get_output_embeddings(self):
        return self.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.predictions.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.albert.embeddings.word_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        halfway_hidden_states = outputs[-1][:self.config.num_hidden_layers // 2]

        aggd_hidden_semantic_state = self.att(halfway_hidden_states[-1])
        mu, sigma_log_pow = self.vae_encoder(aggd_hidden_semantic_state)
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        z_s = self.sample_an_zs(mu, sigma_log_pow, device=device)
        
        zswc = torch.cat([z_s, halfway_hidden_states[-1].view(-1, self.albert_hiddensize)], dim=0)
        B, SEQ, EMB = halfway_hidden_states[-1].size()
        x_c = self.vae_decoder(zswc).view(B, SEQ + 1, EMB)[:, 0, :]

        hidden_states = halfway_hidden_states[-1]
        all_hidden_states = (hidden_states,)


        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        head_mask = [None] * (self.config.num_hidden_layers // 2) if head_mask is None else head_mask

        hidden_states = torch.cat((halfway_hidden_states[-1], x_c.unsqueeze(1)), dim=1)
        ones = torch.zeros(attention_mask.shape[0], 1, device=device)
        attention_mask_first = torch.cat((attention_mask, ones), dim=1)
        extended_attention_mask_first = attention_mask_first[:, :].unsqueeze(1).unsqueeze(2)
        extended_attention_mask_first = extended_attention_mask_first.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask_first = (1.0 - extended_attention_mask_first) * -10000.0
        layer_group_output = self.albert.encoder.albert_layer_groups[0](
            hidden_states[:, :],
            extended_attention_mask_first,
            head_mask[0 : self.config.num_hidden_layers],
            output_attentions,
            output_hidden_states,
        )
        hidden_states = layer_group_output[0][:, :-1]
        if output_attentions:
            all_attentions = all_attentions + layer_group_output[-1][:, :-1]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        for i in range(self.config.num_hidden_layers // 2, self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))
            # print(group_idx)
            layer_group_output = self.albert.encoder.albert_layer_groups[group_idx](
                hidden_states,
                extended_attention_mask,
                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
                output_attentions,
                output_hidden_states,
            )
            hidden_states = layer_group_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        prediction_scores = self.predictions(all_hidden_states[-1])

        # kl-divergence loss computed here.
        kld_loss = -0.5 * torch.mean(
            (1 + sigma_log_pow - mu**2 -
             torch.exp(sigma_log_pow)),
            dim=1)
        # sum the instance loss
        kld_loss = torch.mean(kld_loss, dim=0)

        # cls_loss = torch.sum(
        # cls reconstruction loss computed here
        
        clss_rec = all_hidden_states[-1][:, 0, :]
        clss_grt = outputs[0][:, 0, :]
        # normaldist = Normal(mu, torch.tensor(1, dtype=torch.float32))
        normaldist = Normal(clss_grt, torch.tensor(1, dtype=torch.float32, device=device))
        clsrec_loss = normaldist.log_prob(clss_rec)
        clsrec_loss = clsrec_loss.mean()

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.reshape(-1))
        else:
            masked_lm_loss = 0

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss + kld_loss + clsrec_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
