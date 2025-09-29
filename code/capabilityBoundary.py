import os
import json
import torch
from typing import List
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import joblib 
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
import argparse

def load_test_data(model_name, 
                   datasetNames=["aime24","aime25","amc23",
                # "hmmt_feb_2025",
                # "reliablemath",
                # "gsm8k","hle",
                ]):
    hiddenDict={} 
    hiddenDict[model_name]={}
    resDict=hiddenDict[model_name]
    detailsDict={}
    for datasetName in datasetNames:
        dataDir = "./data/inference"+datasetName
        for root, dirs, files in os.walk(dataDir):
            if model_name in dirs:
                resDict[datasetName]={'correct':[],'wrong':[]}
                detailsDict[datasetName]={'correct':[],'wrong':[]}
                dataSaveDir = os.path.join(root, model_name)
                hidden=torch.load(os.path.join(dataSaveDir,"hiddenStates.pt"))

                with open(os.path.join(dataSaveDir,"evaluation_results.jsonl"), "r", encoding="utf-8") as f:
                    for line in f:
                        evaluation_results=json.loads(line)
                        correctness = evaluation_results['correctness'] 
                        details = evaluation_results['details'] 
                        assert len(details)==len(correctness)

                fullDetails=[]
                with open(os.path.join(dataSaveDir,"all_experiments.jsonl"), "r", encoding="utf-8") as f:
                    for line in f:
                        fullDetails.append(json.loads(line))

                if not len(fullDetails)==len(correctness):
                    print(f"Warning: In {datasetName}, fullDetails length {len(fullDetails)} != correctness length {len(correctness)}")
                    min_len = min(len(fullDetails), len(correctness))
                    fullDetails = fullDetails[:min_len]
                    correctness = correctness[:min_len]
                    hidden = hidden[:min_len]
                


                for i in range(len(hidden)):
                    hiddenStates = hidden[i]['ffn'][-1].squeeze()[-1]
                    
                    if correctness[i]:
                        resDict[datasetName]['correct'].append(hiddenStates)
                        detailsDict[datasetName]['correct'].append(fullDetails[i])
                    else:
                        resDict[datasetName]['wrong'].append(hiddenStates)
                        detailsDict[datasetName]['wrong'].append(fullDetails[i])

    correct_states = []
    wrong_states = []
    for k,v in hiddenDict[model_name].items():
        correct_states+=v['correct']
        wrong_states+=v['wrong']

    correct_details = []
    wrong_details = []
    for k,v in detailsDict.items():
        correct_details+=v['correct']
        wrong_details+=v['wrong']

    correct_np = torch.stack([t.to(torch.float32) for t in correct_states]).cpu().numpy()
    wrong_np = torch.stack([t.to(torch.float32) for t in wrong_states]).cpu().numpy()

    print(f"correct samples: {correct_np.shape}")
    print(f"wrong samples: {wrong_np.shape}")

    return correct_np, wrong_np, correct_details, wrong_details

def extract_last_number(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    pattern = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?"
    matches = re.findall(pattern, text)
    return matches[-1] if matches else ""

def evaluate(res: List, dataset = "gsm8k"):
    total = 0
    correct = 0
    minorList = []

    for data in res:
        pred_raw = data.get("answer", "")
        gold_raw = data.get("gt", "")

        if dataset == "gsm8k":
            gold_ans = extract_last_number(gold_raw) 
        elif dataset == "hle":
            gold_ans = gold_raw

        if gold_ans in pred_raw[-100:]: 
            correct += 1
            if dataset == "hle":
                # print(total)
                minorList.append(total)
        else:
            if dataset == "gsm8k":
                # print(total)
                minorList.append(total)
        total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"Total: {total}, Correct: {correct}, Accuracy: {acc:.2%}")
    return minorList

def capabilityBoundary(X,y,X_test=None, y_test=None, max_iter=500, C = 0.3, visualResSaveDir="/data/qiu_workspace/zqj/thinkingPattern/inferenceResults"):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    joblib.dump(scaler, os.path.join(visualResSaveDir,"scaler.pkl"))

    lda = LDA(solver="lsqr", shrinkage="auto")
    lda.fit(Xs, y)
    joblib.dump(lda, os.path.join(visualResSaveDir,"lda.pkl"))

    w = lda.coef_[0]             
    b = lda.intercept_[0]
    w_norm = np.linalg.norm(w)
    w_hat = w / (w_norm + 1e-12)   

    lr = LogisticRegression(
        solver="lbfgs",           
        penalty="l2",
        C=C,                    
        max_iter=max_iter,
        random_state=42
    )    
    lr.fit(Xs, y)
    joblib.dump(lr, os.path.join(visualResSaveDir,"lr.pkl"))

    proj_on_w = (Xs @ w_hat)[:, None] * w_hat[None, :]
    Xs_res = Xs - proj_on_w
    pca_res = PCA(n_components=1, random_state=0).fit(Xs_res)
    v2 = pca_res.components_[0]
    v2 = v2 - (v2 @ w_hat) * w_hat
    v2 /= (np.linalg.norm(v2) + 1e-12)

    W2 = np.stack([w_hat, v2], axis=1)     
    Z = Xs @ W2                             # (n, 2)
    z1, z2 = Z[:, 0], Z[:, 1]

    if X_test is not None and y_test is not None:
        X_test_scaled  = scaler.transform(X_test) 
        Z_test = X_test_scaled @ W2                             # (n, 2)
        z1_test, z2_test = Z_test[:, 0], Z_test[:, 1]

        y_pred_lda = lda.predict(X_test_scaled)
        y_pred_lr = lr.predict(X_test_scaled)

        acc_lda = np.mean(y_pred_lda == y_test)
        acc_lr = np.mean(y_pred_lr == y_test)
        print(f"LDA overall accuracy: {acc_lda:.2%}")
        print(f"LR overall accuracy: {acc_lr:.2%}")

        for label in [1, 0]:
            mask = (y_test == label)
            acc_lda_label = np.mean(y_pred_lda[mask] == y_test[mask])
            acc_lr_label = np.mean(y_pred_lr[mask] == y_test[mask])
            print(f"LDA class{label}accuracy: {acc_lda_label:.2%}")
            print(f"LR  class{label}accuracy: {acc_lr_label:.2%}")

    pad = 0.5
    z1_min, z1_max = z1.min() - pad, z1.max() + pad
    z2_min, z2_max = z2.min() - pad, z2.max() + pad

    if X_test is not None and y_test is not None:
        z1_min = min(z1_min, z1_test.min() - pad)
        z1_max = max(z1_max, z1_test.max() + pad)
        z2_min = min(z2_min, z2_test.min() - pad)
        z2_max = max(z2_max, z2_test.max() + pad)

    gx, gy = np.meshgrid(np.linspace(z1_min, z1_max, 400),
                            np.linspace(z2_min, z2_max, 400))
    grid_points = gx.ravel()[:, None] * w_hat[None, :] + gy.ravel()[:, None] * v2[None, :]

    lda_scores = lda.decision_function(grid_points)   # shape (G,)
    lda_pred = (lda_scores > 0).astype(int)

    lr_scores = lr.decision_function(grid_points)
    lr_pred = (lr_scores > 0).astype(int)

    Z_lda = lda_pred.reshape(gx.shape)
    Z_lr  = lr_pred.reshape(gx.shape)

    z1_boundary = -b / (w_norm + 1e-12)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    # --- 背景区域（按 LDA 预测标签分色） ---
    ax.contourf(gy, -gx, Z_lda, levels=[-1, 0, 1], colors=["mistyrose","honeydew"], alpha=0.5)

    # --- 数据点 ---
    ax.scatter(z2[y==0], -z1[y==0], label="Unsolvable Question (train)", c="red", alpha=0.7, s=18)
    ax.scatter(z2[y==1], -z1[y==1], label="Solvable Question (train)", c="green", alpha=0.7, s=18)
    if X_test is not None and y_test is not None:
        ax.scatter(z2_test[y_test==0], -z1_test[y_test==0], label="Unsolvable Question (test)", c="orange", alpha=0.7, s=18)
        ax.scatter(z2_test[y_test==1], -z1_test[y_test==1], label="Solvable Question (test)", c="lightgreen", alpha=0.7, s=18)

    # --- 分界线（竖线 → 横线） ---
    ax.axhline(-z1_boundary, linewidth=2.5, color="#579BD3", linestyle="-", label="Capability Boundary")

    # 去掉坐标刻度
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(visualResSaveDir, "visualization")
    plt.savefig(out_path, dpi=160)
    plt.show()

    return scaler, lda, lr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, default="gpt-oss-20b")
    args = parser.parse_args()

    model_name = args.modelname

    print(f"Test model: {model_name}")

    gsm8kRes = []
    file_path="Enter your inference results directory of gsm8k"+model_name+".jsonl"
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            gsm8kRes.append(json.loads(line))

    hleRes = []
    file_path="Enter your inference results directory of hle"+model_name+".jsonl"
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            hleRes.append(json.loads(line))

    gsm8k_states_path = "Enter your hidden states directory of gsm8k"+model_name+"_hiddenStates.pt" 
    if not os.path.exists(gsm8k_states_path):
        gsm8k_states_path = "Enter your hidden states directory of gsm8k"+model_name+"_hiddenStates.pt"

    hle_states_path = "Enter your hidden states directory of hle"+model_name+"_hiddenStates.pt" 
    if not os.path.exists(hle_states_path):
        hle_states_path = "Enter your hidden states directory of hle"+model_name+"_hiddenStates.pt"

    print("Loading hidden states...")
    gsm8k_states = torch.load(gsm8k_states_path)
    hle_states = torch.load(hle_states_path)

    gsm8k_states_ = []
    hle_states_ = []
    for i in range(len(gsm8k_states)):
        gsm8k_states_.append(gsm8k_states[i]['ffn'][-1].squeeze()[-1])
    for i in range(len(hle_states)):
        hle_states_.append(hle_states[i]['ffn'][-1].squeeze()[-1])

    gsm8k_np = torch.stack([t.to(torch.float32) for t in gsm8k_states_]).cpu().numpy()
    hle_np = torch.stack([t.to(torch.float32) for t in hle_states_]).cpu().numpy()

    print(f"gsm8k samples: {gsm8k_np.shape}")
    print(f"hle samples: {hle_np.shape}")

    correct_np, wrong_np, correct_details, wrong_details = load_test_data(args.modelname)

    print("Inference results on gsm8k:")
    gsm8kWrongIndexList=evaluate(gsm8kRes)
    all_indices = set(range(len(gsm8k_np)))
    minor_set = set(gsm8kWrongIndexList)
    gsm8kCorrectIndexList = sorted(list(all_indices - minor_set))

    print("Inference results on hle:")
    hleCorrectIndexList=evaluate(hleRes, dataset="hle")
    all_indices = set(range(len(hle_np)))
    minor_set = set(hleCorrectIndexList)
    hleWrongIndexList = sorted(list(all_indices - minor_set))


    in_diverse = np.concatenate([gsm8k_np[gsm8kCorrectIndexList], correct_np], axis=0) 
    out_diverse = np.concatenate([hle_np[hleWrongIndexList], wrong_np], axis=0)

    X_diverse_all = np.concatenate([in_diverse, out_diverse], axis=0)
    y_diverse_all = np.array([1]*len(in_diverse) + [0]*len(out_diverse))

    X_diverse_all_train, X_diverse_all_val, y_diverse_all_train, y_diverse_all_val = train_test_split(
        X_diverse_all, y_diverse_all, test_size=0.2, random_state=42, stratify=y_diverse_all
    )
    print(X_diverse_all_train.shape, X_diverse_all_val.shape)
    print(y_diverse_all_train.shape, y_diverse_all_val.shape)

    visualResSaveDir="./data/capabilityBoundaries"+model_name
    if not os.path.exists(visualResSaveDir):
        os.makedirs(visualResSaveDir)

    for C in [0.1, 1]:
        print(f"C for lr: {C}")
        scaler, lda, lr = capabilityBoundary(X_diverse_all_train,y_diverse_all_train,X_diverse_all_val,y_diverse_all_val, C=C, visualResSaveDir=visualResSaveDir)    

        predict_proba = lr.predict_proba(scaler.fit_transform(correct_np))

        proba_diff = np.abs(predict_proba[:, 0] - predict_proba[:, 1])
        for threshold in np.arange(0.05, 0.5, 0.05):
            uncertain_indices = np.where(proba_diff < threshold)[0]
            if len(uncertain_indices) > 0:
                print(f"Threshold: {threshold}")
                print("samples close to capability boundary:", uncertain_indices)
                break

        onCBlength = np.average([correct_details[i]['token_stats']['total_tokens'] for i in uncertain_indices])
        averageLength = np.average([item['token_stats']['total_tokens'] for item in correct_details])

        print(f"Average output length of samples close to capability boundary: {onCBlength}, Average output length of all samples: {averageLength}")

if __name__ == "__main__":
    main()