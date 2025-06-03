import random
import torch
from tqdm import tqdm
import CARZero
import pandas as pd
import json
import numpy as np
from CARZero.models.CARZero_model_dqn_wo_self_atten_gl_mlp import CARZeroDQNWOSAGLMLP
from utils import *
import os
from torch.nn.functional import sigmoid


def prep_training_data_example(model: CARZeroDQNWOSAGLMLP, device):
    path = "/home/lpk/CARZero/MIMIC-CXR"
    csv_head = ["id", "text"]
    text_csv = pd.read_csv(f"{path}/text.csv", sep=";", names=csv_head)
    imgs = []
    caption_ids = []
    attention_mask = []
    token_type_ids = []

    for _, row in text_csv.iterrows():
        img_path = f'{path}/{row["id"]}.jpg'
        processed_imgs = model.process_single_img(img_path)
        txts = model.process_text(row["text"], device)
        imgs.append(processed_imgs)
        caption_ids.append(txts["caption_ids"])
        attention_mask.append(txts["attention_mask"])
        token_type_ids.append(txts["token_type_ids"])

    imgs = torch.stack(imgs, dim=0).to(device)
    caption_ids = torch.stack(caption_ids, dim=0).squeeze(1).to(device)
    attention_mask = torch.stack(attention_mask, dim=0).squeeze(1).to(device)
    token_type_ids = torch.stack(token_type_ids, dim=0).squeeze(1).to(device)

    return {
        "imgs": imgs,
        "caption_ids": caption_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=device)
    x = prep_training_data_example(CARZero_model, device)
    
    img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, i2t_cls, t2i_cls = (
        CARZero_model(x)
    )
    
    loss = CARZero_model.calc_loss(
        img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, i2t_cls, t2i_cls
    )
    
    print(loss)
