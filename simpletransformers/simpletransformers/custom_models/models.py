import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    BertModel,
    BertPreTrainedModel,
    DistilBertModel,
    ElectraForMaskedLM,
    # ElectraForPreTraining,
    FlaubertModel,
    RobertaModel,
    XLMModel,
    XLMPreTrainedModel,
    XLNetModel,
    XLNetPreTrainedModel,
)
from transformers.configuration_distilbert import DistilBertConfig
from transformers.configuration_roberta import RobertaConfig
from transformers.configuration_xlm_roberta import XLMRobertaConfig
from transformers.modeling_albert import AlbertConfig, AlbertModel, AlbertPreTrainedModel
from transformers.modeling_distilbert import DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST
from transformers.modeling_electra import (
    ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
    ElectraConfig,
    ElectraModel,
    ElectraPreTrainedModel,
)
from transformers.modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST, RobertaClassificationHead
from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.modeling_xlm_roberta import XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
from transformers.modeling_roberta import RobertaForQuestionAnswering


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """
    Bert model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.pos_weight = pos_weight

        self.init_weights()

    def forward(
        self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaForMultiLabelSequenceClassification(BertPreTrainedModel):
    """
    Roberta model adapted for multi-label sequence classification
    """

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "roberta"

    def __init__(self, config, pos_weight=None):
        super(RobertaForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class XLNetForMultiLabelSequenceClassification(XLNetPreTrainedModel):
    """
    XLNet model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(XLNetForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.d_model, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
        )
        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        logits = self.logits_proj(output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class XLMForMultiLabelSequenceClassification(XLMPreTrainedModel):
    """
    XLM model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(XLMForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.transformer = XLMModel(config)
        self.sequence_summary = SequenceSummary(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
        )

        output = transformer_outputs[0]
        logits = self.sequence_summary(output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep new_mems and attention/hidden states if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class DistilBertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = DistilBertConfig
    pretrained_model_archive_map = DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST
    load_tf_weights = None
    base_model_prefix = "distilbert"

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class DistilBertForMultiLabelSequenceClassification(DistilBertPreTrainedModel):
    """
    DistilBert model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(DistilBertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(
        self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None,
    ):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class AlbertForMultiLabelSequenceClassification(AlbertPreTrainedModel):
    """
    Alber model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(AlbertForMultiLabelSequenceClassification, self).__init__(config)

        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class FlaubertForMultiLabelSequenceClassification(FlaubertModel):
    """
    Flaubert model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(FlaubertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.transformer = FlaubertModel(config)
        self.sequence_summary = SequenceSummary(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
        )

        output = transformer_outputs[0]
        logits = self.sequence_summary(output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep new_mems and attention/hidden states if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class XLMRobertaForMultiLabelSequenceClassification(RobertaForMultiLabelSequenceClassification):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST


class ElectraPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

############activations.py
import logging
import math

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def swish(x):
    return x * torch.sigmoid(x)


def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


ACT2FN = {
    "relu": F.relu,
    "swish": swish,
    "gelu": gelu,
    "tanh": torch.tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
}


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(activation_string, list(ACT2FN.keys())))
##########################
class ElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze()

        return logits


class ElectraRMD(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, disc_hid, seq_len, output_size):
        super().__init__()

        self.fc1 = nn.Linear(disc_hid, disc_hid)
        self.fc2 = nn.Linear(disc_hid, output_size)

    def forward(self, discriminator_hidden_states):
        discriminator_sequence_output = discriminator_hidden_states[:, 0, :]

        input_ = torch.squeeze(discriminator_sequence_output, dim=1)

        fc1_out = F.relu(self.fc1(input_))
        fc2_out = self.fc2(fc1_out)

        return fc2_out

class ElectraForPreTraining(ElectraPreTrainedModel):
    def __init__(self, config, output_size=6, extra_args=None):
        super().__init__(config)
        self.extra_args = extra_args
        if "vanilla_electra" in extra_args and extra_args["vanilla_electra"] != False:
            self.electra = ElectraModel(config)
            self.discriminator_predictions = ElectraDiscriminatorPredictions(config)
            self.init_weights()
        else:
            disc_hid = config.hidden_size
            seq_len = extra_args["max_seq_length"]

            self.electra = ElectraModel(config)
            self.discriminator_predictions = ElectraDiscriminatorPredictions(config)
            self.rmd_predictions = ElectraRMD(disc_hid, seq_len, output_size)
            self.init_weights()

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
    ):
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`, defaults to :obj:`None`):
            Labels for computing the ELECTRA loss. Input should be a sequence of tokens (see :obj:`input_ids` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates the token is an original token,
            ``1`` indicates the token was replaced.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.ElectraConfig`) and inputs:
        loss (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss of the ELECTRA objective.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`)
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import ElectraTokenizer, ElectraForPreTraining
        import torch
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        model = ElectraForPreTraining.from_pretrained('google/electra-small-discriminator')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, seq_relationship_scores = outputs[:2]
        """

        discriminator_hidden_states = self.electra(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, output_attentions,
        )

        discriminator_sequence_output = discriminator_hidden_states[0]

        representations = discriminator_sequence_output[:, 0, :]

        if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
            rmd_output = self.rmd_predictions(discriminator_sequence_output)

        logits = self.discriminator_predictions(discriminator_sequence_output)
        rtd_output = logits

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.float())
            else:
                loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float())

        if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
            return (rmd_output, loss, representations, rtd_output)  # (loss), scores, (hidden_states), (attentions)

        return loss, representations, rtd_output


class ElectraForLanguageModelingModel(PreTrainedModel):
    def __init__(self, config, output_size=100, extra_args=None, **kwargs):
        super(ElectraForLanguageModelingModel, self).__init__(config, **kwargs)
        self.extra_args = extra_args
        if "generator_config" in kwargs:
            generator_config = kwargs["generator_config"]
        else:
            generator_config = config
        self.generator_model = ElectraForMaskedLM(generator_config)
        if "discriminator_config" in kwargs:
            discriminator_config = kwargs["discriminator_config"]
        else:
            discriminator_config = config
        self.discriminator_model = ElectraForPreTraining(discriminator_config, output_size=output_size, extra_args=self.extra_args)
        self.vocab_size = generator_config.vocab_size
        if kwargs.get("tie_generator_and_discriminator_embeddings", True):
            self.tie_generator_and_discriminator_embeddings()
        if "random_generator" in kwargs:
            self.random_generator = kwargs['random_generator']
            print(f'IN MODEL: RANDOM GENERATOR: {self.random_generator}')

    def tie_generator_and_discriminator_embeddings(self):
        gen_embeddings = self.generator_model.electra.embeddings
        disc_embeddings = self.discriminator_model.electra.embeddings

        # tie word, position and token_type embeddings
        gen_embeddings.word_embeddings.weight = disc_embeddings.word_embeddings.weight
        gen_embeddings.position_embeddings.weight = disc_embeddings.position_embeddings.weight
        gen_embeddings.token_type_embeddings.weight = disc_embeddings.token_type_embeddings.weight

    def forward(self, inputs, masked_lm_labels, attention_mask=None, token_type_ids=None, replace_tokens=True):
        d_inputs = inputs.clone()

        if replace_tokens:
            g_out = self.generator_model(
                inputs, masked_lm_labels=masked_lm_labels, attention_mask=attention_mask, token_type_ids=token_type_ids
            )

            sample_probs = torch.softmax(g_out[1], dim=-1, dtype=torch.float32)
            sample_probs = sample_probs.view(-1, self.vocab_size)

            sampled_tokens = torch.multinomial(sample_probs, 1).view(-1)
            sampled_tokens = sampled_tokens.view(d_inputs.shape[0], -1)

            if self.random_generator:
                sampled_tokens = torch.randint(5, self.vocab_size-1, sampled_tokens.shape)
                sampled_tokens = sampled_tokens.cuda()

            # labels have a -100 value to mask out loss from unchanged tokens.
            mask = masked_lm_labels.ne(-100)

            # replace the masked out tokens of the input with the generator predictions.
            d_inputs[mask] = sampled_tokens[mask]

            # turn mask into new target labels.  1 (True) for corrupted, 0 otherwise.
            # if the prediction was correct, mark it as uncorrupted.
            correct_preds = sampled_tokens == masked_lm_labels
            d_labels = mask.long()
            d_labels[correct_preds] = 0
        else:
            # no element is corrupted
            d_labels = torch.zeros_like(masked_lm_labels)

        # run token classification, predict whether each token was corrupted.
        output = self.discriminator_model(
            d_inputs, labels=d_labels, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
            d_out, d_loss, representations, rtd_out = output
            if replace_tokens:
                g_loss = g_out[0]
                g_scores = g_out[1]
            else:
                g_loss = None
                g_scores = None
            d_scores = d_out #useless?

            return g_loss, g_scores, d_out, d_inputs, d_loss, d_scores, d_labels, representations, rtd_out
        else:
            d_loss, representations, rtd_out = output
            g_loss = g_out[0]
            # d_loss = output_electra[0]
            g_scores = g_out[1]

            return g_loss, g_scores, d_inputs, d_loss, d_labels, representations, rtd_out



class ElectraForSequenceClassification(ElectraPreTrainedModel):
    r"""
    Mostly the ssame as BertForSequenceClassification. A notable difference is that this class contains a pooler while
    BertForSequenceClassification doesn't. This is because pooling happens internally in a BertModel but not in an
    ElectraModel.
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    """  # noqa
    config_class = ElectraConfig
    pretrained_model_archive_map = ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "electra"

    def __init__(self, config, weight=None):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)
        self.pooler = ElectraPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.weight = weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds)
        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(weight=self.weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class ElectraForMultiLabelSequenceClassification(ElectraPreTrainedModel):
    """
    ElectraForSequenceClassification model adapted for multi-label sequence classification
    """

    config_class = ElectraConfig
    pretrained_model_archive_map = ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "electra"

    def __init__(self, config, pos_weight=None):
        super(ElectraForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.electra = ElectraModel(config)
        self.pooler = ElectraPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds)
        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class ElectraForQuestionAnswering(ElectraPreTrainedModel):
    """
    Identical to BertForQuestionAnswering other than using an ElectraModel
    """

    config_class = ElectraConfig
    pretrained_model_archive_map = ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "electra"

    def __init__(self, config, weight=None):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):

        outputs = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class XLMRobertaForQuestionAnswering(RobertaForQuestionAnswering):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
