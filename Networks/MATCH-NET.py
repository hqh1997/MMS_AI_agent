import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/')  
from Networks.blocks import AttentionNetGated, CombinedPreGatedMultiheadAttention
from Networks.fusion import BilinearFusion, ConcatFusion, GatedConcatFusion
from typing import List

class MATCH_Net(nn.Module):
    def __init__(self, mic_sizes: List[int], model_size: str = 'medium', n_classes: int = 4, dropout: float = 0.25, fusion: str = 'concat', device: str = 'cpu'):
        super(MATCH_Net, self).__init__()
        self.n_classes = n_classes
        if model_size == 'small':
            self.model_sizes = [128, 128]
        elif model_size == 'medium':
            self.model_sizes = [256, 256]
        elif model_size == 'big':
            self.model_sizes = [512, 512]
        # H
        fc = nn.Sequential(
            nn.Linear(1024, self.model_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.H = fc
        # G
        mic_encoders = []
        for mic_size in mic_sizes:
            fc = nn.Sequential(
                nn.Sequential(
                    nn.Linear(mic_size, self.model_sizes[0]),
                    nn.ELU(),
                    nn.AlphaDropout(p=dropout, inplace=False)),
                nn.Sequential(
                    nn.Linear(self.model_sizes[0], self.model_sizes[1]),
                    nn.ELU(),
                    nn.AlphaDropout(p=dropout, inplace=False))
            )
            mic_encoders.append(fc)

        self.G = nn.ModuleList(mic_encoders)

        self.co_attention_M = CombinedPreGatedMultiheadAttention(
            dim1=self.model_sizes[1], 
            dim2=self.model_sizes[1], 
            dk=self.model_sizes[1], 
            output_dim=self.model_sizes[1],  
            num_heads=1
        )
        self.co_attention_R = CombinedPreGatedMultiheadAttention(
            dim1=self.model_sizes[1],
            dim2=self.model_sizes[1],
            dk=self.model_sizes[1],
            output_dim=self.model_sizes[1],
            num_heads=1
        )
        self.co_attention_T = CombinedPreGatedMultiheadAttention(
            dim1=self.model_sizes[1],
            dim2=self.model_sizes[1],
            dk=self.model_sizes[1],
            output_dim=self.model_sizes[1],
            num_heads=1
        )

        # Path Transformer (T_H)
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_sizes[1], nhead=8, dim_feedforward=512, dropout=dropout,
                                                        activation='relu')
        self.path_transformer_M = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        self.path_transformer_R = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        self.path_transformer_T = nn.TransformerEncoder(path_encoder_layer, num_layers=2)

        # WSI Global Attention Pooling (rho_H_M)
        self.path_attention_head_M = AttentionNetGated(n_classes=1, input_dim=self.model_sizes[1], hidden_dim=self.model_sizes[1])
        self.path_rho_M = nn.Sequential(*[nn.Linear(self.model_sizes[1], self.model_sizes[1]), nn.ReLU(), nn.Dropout(dropout)])

        # WSI Global Attention Pooling (rho_H_R)   
        self.path_attention_head_R = AttentionNetGated(n_classes=1, input_dim=self.model_sizes[1], hidden_dim=self.model_sizes[1])
        self.path_rho_R = nn.Sequential(*[nn.Linear(self.model_sizes[1], self.model_sizes[1]), nn.ReLU(), nn.Dropout(dropout)])

        # WSI Global Attention Pooling (rho_H_T)
        self.path_attention_head_T = AttentionNetGated(n_classes=1, input_dim=self.model_sizes[1], hidden_dim=self.model_sizes[1])
        self.path_rho_T = nn.Sequential(*[nn.Linear(self.model_sizes[1], self.model_sizes[1]), nn.ReLU(), nn.Dropout(dropout)])

        # macro Transformer (T_G)
        macro_encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_sizes[1], nhead=8, dim_feedforward=512, dropout=dropout,
                                                        activation='relu')
        self.macro_transformer = nn.TransformerEncoder(macro_encoder_layer, num_layers=2)
        # radio Transformer (T_G)
        radio_encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_sizes[1], nhead=8, dim_feedforward=512, dropout=dropout,
                                                        activation='relu')
        self.radio_transformer = nn.TransformerEncoder(radio_encoder_layer, num_layers=2)
        # text Transformer (T_G)
        text_encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_sizes[1], nhead=8, dim_feedforward=512, dropout=dropout,
                                                        activation='relu')
        self.text_transformer = nn.TransformerEncoder(text_encoder_layer, num_layers=2)

        # macro Global Attention Pooling 
        self.macro_attention_head = AttentionNetGated(n_classes=1, input_dim=self.model_sizes[1], hidden_dim=self.model_sizes[1])
        self.macro_rho = nn.Sequential(*[nn.Linear(self.model_sizes[1], self.model_sizes[1]), nn.ReLU(), nn.Dropout(dropout)])
        
        # radio Global Attention Pooling 
        self.radio_attention_head = AttentionNetGated(n_classes=1, input_dim=self.model_sizes[1], hidden_dim=self.model_sizes[1])
        self.radio_rho = nn.Sequential(*[nn.Linear(self.model_sizes[1], self.model_sizes[1]), nn.ReLU(), nn.Dropout(dropout)])

        # text Global Attention Pooling 
        self.text_attention_head = AttentionNetGated(n_classes=1, input_dim=self.model_sizes[1], hidden_dim=self.model_sizes[1])
        self.text_rho = nn.Sequential(*[nn.Linear(self.model_sizes[1], self.model_sizes[1]), nn.ReLU(), nn.Dropout(dropout)])

        # Fusion Layer
        self.fusion = fusion
        if self.fusion == 'concat':
            self.fusion_layer = ConcatFusion(dims=[self.model_sizes[1], self.model_sizes[1], self.model_sizes[1], self.model_sizes[1], self.model_sizes[1], self.model_sizes[1]],
                                             hidden_size=self.model_sizes[1], output_size=self.model_sizes[1]).to(device=device)
        elif self.fusion == 'bilinear':
            self.fusion_layer = BilinearFusion(dim1=self.model_sizes[1], dim2=self.model_sizes[1], output_size=self.model_sizes[1])
        elif self.fusion == 'gated_concat':
            self.fusion_layer = GatedConcatFusion(dims=[self.model_sizes[1], self.model_sizes[1], self.model_sizes[1], self.model_sizes[1], self.model_sizes[1], self.model_sizes[1]],
                                                  hidden_size=self.model_sizes[1], output_size=self.model_sizes[1]).to(device=device)
        else:
            raise RuntimeError(f'Fusion mechanism {self.fusion} not implemented')
        # Classifier
        self.classifier = nn.Linear(self.model_sizes[1], n_classes)


    def forward(self, wsi, macros, radios, texts):

        H_bag = self.H(wsi).squeeze(0)
        M_bag = self.G[0](macros.type(torch.float32)).squeeze(0)
        R_bag = self.G[1](radios.type(torch.float32)).squeeze(0)
        T_bag = self.G[2](texts.type(torch.float32)).squeeze(0)

        H_coattn_M, M_coattn = self.co_attention_M(H_bag, M_bag)
        path_trans_M = self.path_transformer_M(H_coattn_M)
        macro_trans = self.macro_transformer(M_bag)

        # Global Attention Pooling
        A_path_M, h_path_M = self.path_attention_head_M(path_trans_M.squeeze(1))
        A_path_M = torch.transpose(A_path_M, 1, 0)
        h_path_M = torch.mm(F.softmax(A_path_M, dim=1), h_path_M)
        h_path_M = self.path_rho_M(h_path_M).squeeze()

        A_macro, h_macro = self.macro_attention_head(macro_trans.squeeze(1))
        A_macro = torch.transpose(A_macro, 1, 0)
        h_macro = torch.mm(F.softmax(A_macro, dim=1), h_macro)
        h_macro = self.macro_rho(h_macro).squeeze()

        H_coattn_R, R_coattn = self.co_attention_R(H_bag, R_bag)
        path_trans_R = self.path_transformer_R(H_coattn_R)
        radio_trans = self.radio_transformer(R_bag)

        A_path_R, h_path_R = self.path_attention_head_R(path_trans_R.squeeze(1))
        A_path_R = torch.transpose(A_path_R, 1, 0)
        h_path_R = torch.mm(F.softmax(A_path_R, dim=1), h_path_R)
        h_path_R = self.path_rho_R(h_path_R).squeeze()

        A_radio, h_radio = self.radio_attention_head(radio_trans.squeeze(1))
        A_radio = torch.transpose(A_radio, 1, 0)
        h_radio = torch.mm(F.softmax(A_radio, dim=1), h_radio)
        h_radio = self.radio_rho(h_radio).squeeze()

        H_coattn_T, T_coattn = self.co_attention_T(H_bag, T_bag)
        path_trans_T = self.path_transformer_T(H_coattn_T)
        text_trans = self.text_transformer(T_bag)

        A_path_T, h_path_T = self.path_attention_head_T(path_trans_T.squeeze(1))
        A_path_T = torch.transpose(A_path_T, 1, 0)
        h_path_T = torch.mm(F.softmax(A_path_T, dim=1), h_path_T)
        h_path_T = self.path_rho_T(h_path_T).squeeze()

        A_text, h_text = self.text_attention_head(text_trans.squeeze(1))
        A_text = torch.transpose(A_text, 1, 0)
        h_text = torch.mm(F.softmax(A_text, dim=1), h_text)
        h_text = self.text_rho(h_text).squeeze()

        # print(self.model_sizes[1])
        # print( self.fusion_layer)
        h = self.fusion_layer(h_path_M, h_macro, h_path_R, h_radio,h_path_T, h_text)
        # print(h.shape)
        # logits: classifier output
        # size   --> (1, 4)
        # domain --> R
        logits = self.classifier(h).unsqueeze(0)
        # hazards: probability of patient death in interval j
        # size   --> (1, 4)
        # domain --> [0, 1]
        hazards = torch.sigmoid(logits)
        # survs: probability of patient survival after time t
        # size   --> (1, 4)
        # domain --> [0, 1]
        survs = torch.cumprod(1 - hazards, dim=1)
        # Y: predicted probability distribution
        # size   --> (1, 4)
        # domain --> [0, 1] (probability distribution)
        Y = F.softmax(logits, dim=1)

        attention_scores = {'M_coattn': M_coattn, 'R_coattn': R_coattn, 'T_coattn': T_coattn, 'radio':A_radio, 'macro':A_macro, 'text':A_text,
                             'A_path_M':A_path_M, 'A_path_R':A_path_R, 'A_path_T':A_path_T}  

        return hazards, survs, Y, attention_scores


    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
