import torch
from tqdm import tqdm
import CARZero
import pandas as pd
import json
import numpy as np
from utils import *
import os
from sklearn.preprocessing import MultiLabelBinarizer

def obtain_simr(image_path, text_path):
    df = pd.read_csv(image_path)
    with open(text_path, "r") as f:
        cls_prompts = json.load(f)

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CARZero_model = CARZero.load_CARZero(name="CARZero_vit_b_16", device=device)

    # process input images and class prompts
    ## batchsize
    bs = 32
    image_list = split_list(df["Path"].tolist(), bs)
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

    return similar


def triple_Chexpert14_result(predict_csv, label_file_path):
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
    df_test = pd.read_csv(label_file_path, sep=" ", names=csv_head)

    key = csv_head[1:]

    predict = pd.read_csv(predict_csv).values
    label = df_test[key].values
    pre = np.zeros((predict.shape[0], predict.shape[1]))
    for i in range(predict.shape[0]):
        logit = predict[i]
        ind = np.argmax(logit)
        pre[i, ind] = 1

    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)
    print(f"Total AUC: {macro_auc}")
    micro_prc, macro_prc = calculate_micro_macro_auprc(label, predict)
    print("Micro AUPRC: {:.4f}, Macro AUPRC: {:.4f}".format(micro_prc, macro_prc))
    for disease, auc in zip(key, per_auc):
        print(f"{disease}: {auc}")

    save_macro_auprc_plot(label, predict, predict_csv.replace(".csv", ".png"))
    print(f"Save {predict_csv.replace('.csv', '.png')}")


def tripple_openi_rusult_merge(predict_csv, label_file_path):
    pathologies = [
        # NIH
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        ## "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
        # ---------
        "Fracture",
        "Opacity",
        "Lesion",
        # ---------
        "Calcified Granuloma",
        "Granuloma",
        # ---------
        "No_Finding",
    ]

    mapping = dict()
    mapping["Pleural_Thickening"] = ["pleural thickening"]
    mapping["Infiltration"] = ["Infiltrate"]
    mapping["Atelectasis"] = ["Atelectases"]

    # Load data
    csv = pd.read_csv(label_file_path)
    csv = csv.replace(np.nan, "-1")

    gt = []
    for pathology in pathologies:
        mask = csv["labels_automatic"].str.contains(pathology.lower())
        if pathology in mapping:
            for syn in mapping[pathology]:
                # print("mapping", syn)
                mask |= csv["labels_automatic"].str.contains(syn.lower())
        gt.append(mask.values)

    gt = np.asarray(gt).T
    gt = gt.astype(np.float32)

    # Rename pathologies
    pathologies = np.char.replace(pathologies, "Opacity", "Lung Opacity")
    pathologies = np.char.replace(pathologies, "Lesion", "Lung Lesion")

    ## Rename by myself
    pathologies = np.char.replace(
        pathologies, "Pleural_Thickening", "pleural thickening"
    )
    pathologies = np.char.replace(pathologies, "Infiltration", "Infiltrate")
    pathologies = np.char.replace(pathologies, "Atelectasis", "Atelectases")
    gt[np.where(np.sum(gt, axis=1) == 0), -1] = 1

    label = gt[:, :-1]

    predict = pd.read_csv(predict_csv).values

    head, medium, tail = obtaion_LT_multi_label_distribution(label)

    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(
        predict[:, head], label[:, head]
    )
    print(f"Head AUC: {macro_auc}")
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(
        predict[:, medium], label[:, medium]
    )
    print(f"Medium AUC: {macro_auc}")
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(
        predict[:, tail], label[:, tail]
    )
    print(f"Tail AUC: {macro_auc}")
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(predict, label)
    print(f"Total AUC: {macro_auc}")
    micro_prc, macro_prc = calculate_micro_macro_auprc(label, predict)
    print("Micro AUPRC: {:.4f}, Macro AUPRC: {:.4f}".format(micro_prc, macro_prc))

    for i, k in enumerate(pathologies[:-1]):
        print(f"{k}: {per_auc[i]}")


if __name__ == "__main__":

    images = [
        "./Dataset/OpenI/openi_multi_label_image.csv",
        "./Dataset/ChestXray14/chestxray14_test_image.csv",
    ]

    texts = [
        "./Dataset/OpenI/openi_multi_label_text.json",
        "./Dataset/ChestXray14/chestxray14_test_text.json",
    ]

    result_file_name = "test"

    os.makedirs("./Performance/" + result_file_name, exist_ok=True),

    save_csvs = [
        "./Performance/" + result_file_name + "/Openi.csv",
        "./Performance/" + result_file_name + "/ChestXray14.csv",
    ]

    # Skip running again
    # for i, (img, txt, savecsv) in enumerate(zip(images, texts, save_csvs)):
    #     start = time.time()
    #     similarities = obtain_simr(img, txt)
    #     similarities.to_csv(savecsv, index=False)

    print("Openi")
    tripple_openi_rusult_merge(save_csvs[0], "./Dataset/OpenI/custom.csv")
    print("ChestXray14")
    triple_Chexpert14_result(save_csvs[1], "./Dataset/ChestXray14/test_list.txt")
