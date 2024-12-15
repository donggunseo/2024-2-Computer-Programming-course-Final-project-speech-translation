from transformers import Wav2Vec2Model, MBartModel, MBartForConditionalGeneration
import torch
import torch.nn as nn
from transformers.generation.utils import GenerationMixin

def expand_attention_mask(attention_mask, tgt_len):
    # Attention mask를 4D로 확장
    return attention_mask[:, None, None, :].expand(-1, 1, tgt_len, -1)

class JointSpeechEncoder(nn.Module):
    def __init__(self, wav2vec_model="facebook/wav2vec2-large-960h-lv60"):
        super(JointSpeechEncoder, self).__init__()
        # Load Wav2Vec2 and mBART encoder
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec_model)
        self.wav2vec2.config.attention_dropout = 0.3
        # Wav2Vec2 configuration
        self.wav2vec_hidden_size = self.wav2vec2.config.hidden_size

        # Use only the first 12 layers of Wav2Vec2
        self.wav2vec_layers = nn.ModuleList(self.wav2vec2.encoder.layers[:12])

        # Length adaptor (Down-sampling)
        self.length_adaptor = nn.Sequential(
            nn.Conv1d(self.wav2vec_hidden_size, self.wav2vec_hidden_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.wav2vec_hidden_size, self.wav2vec_hidden_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.wav2vec_hidden_size, self.wav2vec_hidden_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, input_values):
        # Wav2Vec2 base processing
        hidden_states = self.wav2vec2.feature_extractor(input_values)
        hidden_states = hidden_states.transpose(1,2)
        hidden_states, _ = self.wav2vec2.feature_projection(hidden_states)
        hidden_states = self.wav2vec2._mask_hidden_states(
            hidden_states, mask_time_indices=None, attention_mask=None
        )
        position_embeddings = self.wav2vec2.encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.wav2vec2.encoder.layer_norm(hidden_states)
        hidden_states = self.wav2vec2.encoder.dropout(hidden_states)


        # Pass through the first 12 layers of Wav2Vec2
        for layer in self.wav2vec_layers:
            hidden_states = layer(hidden_states)[0]


        # Down-sampling using length adaptor
        hidden_states = hidden_states.transpose(1, 2)  # Convert to (Batch, Hidden, Seq) for Conv1D
        hidden_states = self.length_adaptor(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)  # Convert back to (Batch, Seq, Hidden)

        batch_size, sequence_length, hidden_size = hidden_states.size()
        if hidden_size != self.wav2vec_hidden_size:
            raise ValueError(f"Hidden size mismatch. Expected {self.wav2vec_hidden_size}, got {hidden_size}")

        # Return downsampled output
        return hidden_states

class JointSpeechTranslationModel(nn.Module, GenerationMixin):
    def __init__(self, wav2vec_model="facebook/wav2vec2-large-960h-lv60", mbart_model="facebook/mbart-large-50-one-to-many-mmt"):
        super(JointSpeechTranslationModel, self).__init__()
        # Initialize JointSpeechEncoder
        self.speech_encoder = JointSpeechEncoder(wav2vec_model)
        self.speech_encoder.wav2vec2.freeze_feature_extractor()

        # Load mBART model
        self.mbart_model = MBartForConditionalGeneration.from_pretrained(mbart_model)
        self.mbart_model.config.attention_dropout=0.3
        

        self.generation_config = self.mbart_model.generation_config
        self.config = self.mbart_model.config 

    def forward(self, input_values, labels=None, attention_mask=None):
        # Pass speech input through the speech encoder
        if labels is not None:
            labels = labels.contiguous()
        encoder_outputs = self.speech_encoder(input_values)
        del input_values
        # Pass encoder outputs and decoder inputs to the mBART model
        outputs = self.mbart_model(
            inputs_embeds=encoder_outputs,  # Pass encoder outputs as embeddings
            labels=labels,  # Labels for loss computation
            attention_mask=None # Optional: Attention mask
        )
        

        return outputs
    
    def generate(self, input_values, attention_mask=None, **generate_kwargs):
        # Speech encoder outputs
        encoder_outputs = self.speech_encoder(input_values)
        del input_values
        # Use mBART generate method
        return self.mbart_model.generate(
            inputs_embeds=encoder_outputs,
            attention_mask=None,
            **generate_kwargs
        )
