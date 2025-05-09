import random
import torch
from tqdm import tqdm
import CARZero
import pandas as pd
import json
import numpy as np
from utils import *
import os
from torch.nn.functional import sigmoid

def obtain_simr_demo(image_path, text_path, demo_size=1000):
    df = pd.read_csv(image_path)
    with open(text_path, "r") as f:
        cls_prompts = json.load(f)

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=device)

    lst = df["Path"].tolist()
    random.shuffle(lst)
    lst = lst[:demo_size]
    bs = 32
    image_list = split_list(lst, bs)

    processed_txt = CARZero_model.process_class_prompts(cls_prompts, device)

    for i, img in tqdm(
        enumerate(image_list), total=len(image_list), desc="Processing images"
    ):
        processed_imgs = CARZero_model.process_img(img, device)
        similarities = CARZero.dqn_shot_classification(
            CARZero_model, processed_imgs, processed_txt
        )

        if i == 0:
            similar = similarities
        else:
            similar = pd.concat([similar, similarities], axis=0)

    return lst, similar


def check_result_chestxray_14(predict, image_list):
    csv_head = [
        "path",
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Lung Mass",
        "Lung Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural Thickening",
        "Hernia",
    ]
    df_test = pd.read_csv(
        "./Dataset/ChestXray14/test_list.txt", sep=" ", names=csv_head
    )

    key = csv_head[1:]

    df_test = df_test[df_test["path"].isin(image_list)]

    label = df_test[key].values

    # predict_probs = sigmoid(torch.tensor(predict.values, dtype=torch.float32)).numpy()

    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict.values, label)
    print(f"Total AUC: {macro_auc}")
    # micro_prc, macro_prc = calculate_micro_macro_auprc(label, predict)
    # print("Micro AUPRC: {:.4f}, Macro AUPRC: {:.4f}".format(micro_prc, macro_prc))
    # for disease, auc in zip(key, per_auc):
    #     print(f"{disease}: {auc}")


if __name__ == "__main__":
    image_list, similarities = obtain_simr_demo(
        "./Dataset/ChestXray14/chestxray14_test_image.csv",
        "./Dataset/ChestXray14/chestxray14_test_text.json",
    )

    image_list = [os.path.join(*path.split("/")[-2:]) for path in image_list]

    check_result_chestxray_14(similarities, image_list)
