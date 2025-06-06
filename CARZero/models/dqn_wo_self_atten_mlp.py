import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
import ipdb

from transformers import AutoModel, BertConfig, AutoTokenizer

from ..models.transformer_decoder import *

# from transformer_decoder import *


class TQN_Model(nn.Module):
    def __init__(
        self,
        # embed_dim: int = 768,
        # class_num: int = 2,
        cfg=None,
    ):
        super().__init__()
        embed_dim = cfg.model.fusion.d_model
        class_num = cfg.model.fusion.class_num
        decoder_number_layer = cfg.model.fusion.decoder_number_layer

        # embed_dim = 768
        # class_num = 1
        # decoder_number_layer = 4
        self.d_model = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        decoder_layer = TransformerDecoderWoSelfAttenLayer(
            self.d_model, 4, 1024, 0.1, "relu", normalize_before=True
        )
        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            decoder_number_layer,
            self.decoder_norm,
            return_intermediate=False,
        )
        self.dropout_feas = nn.Dropout(0.1)
        # v1
        # self.mlp_head = nn.Sequential( # nn.LayerNorm(768),
        #     nn.Linear(embed_dim, embed_dim // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(embed_dim // 2, embed_dim // 4),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(embed_dim // 4, class_num)
        # )

        # v2
        # self.mlp_head = nn.Sequential( # nn.LayerNorm(768),
        #     nn.Linear(embed_dim, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(1024, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, class_num)
        # )

        # # v3
        self.mlp_head = nn.Sequential(  # nn.LayerNorm(768),
            nn.Linear(embed_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, class_num),
        )

        # v4
        # self.mlp_head = nn.Sequential( # nn.LayerNorm(768),
        #     nn.Linear(embed_dim, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(512, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(128, class_num)
        # )

        # v5
        # self.mlp_head = nn.Sequential( # nn.LayerNorm(768),
        #     nn.Linear(embed_dim, 2048),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(512, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, class_num)
        # )

        # v6
        # self.mlp_head = nn.Sequential( # nn.LayerNorm(768),
        #     nn.Linear(embed_dim, 2048),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(512, class_num)
        # )

        # v7
        # self.mlp_head = nn.Sequential( # nn.LayerNorm(768),
        #     nn.Linear(embed_dim, 768),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(768, 384),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(384, 192),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(192, class_num)
        # )

        # v8
        # self.mlp_head = nn.Sequential( # nn.LayerNorm(768),
        #     nn.Linear(embed_dim, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(1024, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(64, class_num)
        # )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # @Note: Demo flow -- fusion
    def forward(
        self,
        k_v_features,  # Memory input (e.g., from encoder) → shape: [batch_size, seq_len, dim]
        q_features,  # Query input (e.g., decoder input) → shape: [query_size: batch_size (in_training) flex_size (inferences), dim]
        pos=None,
        return_atten=False,
        inside_repeat=True,
    ):

        batch_size = k_v_features.shape[0]

        # Transpose memory features to match TransformerDecoder input shape
        # Resulting shape: [seq_len, batch_size, dim]
        k_v_features = k_v_features.transpose(0, 1)

        # Expand q_features
        # Original q_features: [query_size, dim]
        # After unsqueeze + repeat: [query_size, batch_size, dim]
        if inside_repeat:
            q_features = q_features.unsqueeze(1).repeat(1, batch_size, 1)

        # Apply normalization to both query and memory features
        # Helps with training stability
        k_v_features = self.decoder_norm(k_v_features)
        q_features = self.decoder_norm(q_features)

        # Pass query and key/value features through the decoder
        # Features shape: [query_size, batch_size, d_model]
        # Attention map shape: [batch_size, query_size, key_len]
        features, atten_map = self.decoder(
            q_features, # [query_size, batch_size, dim]
            k_v_features, # [seq_len, batch_size, dim]
            memory_key_padding_mask=None,
            pos=pos,
            query_pos=None,
        )

        # Apply dropout and transpose output to shape [batch_size, query_size, dim]
        features = self.dropout_feas(features).transpose(0, 1)

        # Project features to final output space
        # Resulting shape: [batch_size, query_size]
        out = self.mlp_head(features)

        # Step 8: Optionally return attention map along with output
        if return_atten:
            return out, atten_map
        else:
            return out


# if __name__ == "__main__":
#     x_global = torch.ones(64, 768).cuda()
#     x_local = torch.ones(64, 64, 768).cuda()

#     model = TQN_Model().cuda()
#     with torch.no_grad():
#         model.eval()
#         out = model(x_local, x_global)
#         ipdb.set_trace()
