import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from xgboost import XGBRFClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 12

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()


def BH(calib_scores, calib_scores_hat, test_scores_hat, delta, c_min):

    ntest = len(test_scores_hat)
    ncalib = len(calib_scores)
    pvals = np.zeros(ntest)
    idx = np.arange(ntest)

    ''''
        Compute the corresponding p-value
    '''
    for j in range(ntest):
        pvals[j] = (1 + np.sum((calib_scores < c_min) & (calib_scores_hat >= test_scores_hat[j]))) / (ncalib + 1)

    sorted_indices = np.argsort(pvals)
    sorted_pvals = pvals[sorted_indices]
    thresholds = (np.arange(1, ntest + 1) / ntest) * delta

    selected = sorted_pvals <= thresholds
    if np.any(selected):
        k_max = np.max(np.where(selected))
        edge_indices = sorted_indices[: k_max + 1]
    else:
        edge_indices = []
    cloud_indices = np.setdiff1d(idx, edge_indices)
    return edge_indices, cloud_indices


def CA(edge_test_probs, cloud_test_probs, cloud_prediction_set, training_data, val_data, test_data, edge_set_training, edge_set_val, edge_set_test, alpha, delta):

    edge_confidence, _ = edge_test_probs.max(dim=1)

    #######################  Train the alignment score predictor on the training dataset  #######################
    ground_truth_alignment_score_binary_training = (torch.mul(cloud_test_probs[training_data], edge_set_training).sum(dim=1) >= 1 - alpha).to(torch.int).float().cpu().numpy()
    clf = XGBRFClassifier(n_estimators=100, subsample=0.9, colsample_bynode=0.2).fit(edge_confidence[training_data], ground_truth_alignment_score_binary_training)
    # clf = RandomForestClassifier(max_depth=30, random_state=2024).fit(edge_confidence[training_data], ground_truth_alignment_score_binary_training)
    # clf = LogisticRegression(class_weight='balanced').fit(edge_confidence[training_data], ground_truth_alignment_score_binary_training)

    #######################  Predict the alignment score for val and test dataset  #######################
    edge_confidence = edge_confidence.cpu().numpy().reshape(-1, 1)
    predicted_alignment_score_test = clf.predict_proba(edge_confidence[test_data])[:, 1]
    predicted_alignment_score_val = clf.predict_proba(edge_confidence[val_data])[:, 1]

    #######################  Evaluate the alignement score for val and test dataset  #######################
    ground_truth_alignment_score_val = torch.mul(cloud_test_probs[val_data], edge_set_val).sum(dim=1).cpu().numpy()
    ground_truth_alignment_score_binary_val = (torch.mul(cloud_test_probs[val_data], edge_set_val).sum(dim=1) >= 1 - alpha).to(torch.int).float().cpu().numpy()

    ground_truth_alignment_score_test = torch.mul(cloud_test_probs[test_data], edge_set_test).sum(dim=1).cpu().numpy()
    ground_truth_alignment_score_binary_test = (torch.mul(cloud_test_probs[test_data], edge_set_test).sum(dim=1) >= 1 - alpha).to(torch.int).float().cpu().numpy()

    edge_processed_inputs, cloud_processed_inputs = BH(ground_truth_alignment_score_val, predicted_alignment_score_val, predicted_alignment_score_test, delta, 1 - alpha)

    if len(edge_processed_inputs) == 0:
        avg_sat_rate = 1
    else:
        avg_sat_rate = np.sum(ground_truth_alignment_score_test[edge_processed_inputs] >= 1 - alpha) / len(edge_processed_inputs)

    deferral_rate = len(cloud_processed_inputs) / len(test_data)
    edge_normalized_size = torch.mul(edge_set_test.sum(dim=1), 1 / cloud_prediction_set[test_data].sum(dim=1))
    ineff = (edge_normalized_size[edge_processed_inputs].sum() + len(cloud_processed_inputs)) / len(test_data)

    results = {
        "deferral_ratio": deferral_rate.item() if torch.is_tensor(deferral_rate) else deferral_rate,
        "normalized inefficiency": ineff.item() if torch.is_tensor(ineff) else ineff,
        "avg_sat_rate": avg_sat_rate.item() if torch.is_tensor(avg_sat_rate) else avg_sat_rate,
        'alpha': alpha,
        'delta': delta
    }

    return results

def model_cascading(edge_test_probs, cloud_test_probs, edge_prediction_set, cloud_prediction_set, gamma, alpha, delta):

    edge_process_inputs = (edge_test_probs.max(dim=1).values >= gamma).to(torch.int).float()
    cloud_process_inputs = (edge_test_probs.max(dim=1).values < gamma).to(torch.int).float()

    conventional_cas_deferral_rate = 1 - edge_process_inputs.sum() / edge_test_probs.shape[0]

    edge_conventional_normalized_size = torch.mul(edge_prediction_set.sum(dim=1), 1 / cloud_prediction_set.sum(dim=1))
    conventional_cas_ineff = (torch.mul(edge_process_inputs, edge_conventional_normalized_size).sum() + cloud_process_inputs.sum()) / edge_test_probs.shape[0]

    a = (torch.mul(torch.mul(cloud_test_probs, edge_prediction_set).sum(dim=1), edge_process_inputs) >= 1 - alpha).to(torch.int).float().sum()
    conv_avg_sat_rate = a / edge_process_inputs.sum()

    results = {
        "deferral_rate": conventional_cas_deferral_rate.item() if torch.is_tensor(conventional_cas_deferral_rate) else conventional_cas_deferral_rate,
        "normalized inefficiency": conventional_cas_ineff.item() if torch.is_tensor(conventional_cas_ineff) else conventional_cas_ineff,
        "avg_sat_rate": conv_avg_sat_rate.item() if torch.is_tensor(conv_avg_sat_rate) else conv_avg_sat_rate,
        'alpha': alpha,
        'delta': delta
    }

    return results

def minimal_prediction_set(
    probs: torch.Tensor,
    alpha: float,
    dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert 0 <= alpha < 1

    vals, idx = probs.sort(dim=dim, descending=True)
    csum = vals.cumsum(dim=dim)

    target = 1.0 - alpha
    ge_mask = csum >= target

    first_true = ge_mask.to(torch.int).argmax(dim=dim)
    set_sizes = first_true + 1

    B, C = probs.shape
    ranks = torch.arange(C, device=probs.device).unsqueeze(0)
    take_rank = ranks <= first_true.unsqueeze(1)

    pred_set_mask = torch.zeros_like(ge_mask)
    pred_set_mask.scatter_(dim, idx, take_rank)

    return pred_set_mask, set_sizes


num_bins = 15
COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'

def _bin_initializer(bin_dict, num_bins=15):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0

def _populate_bins(confs, preds, labels, batch, num_bins=15):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    if batch is False:
        for i in range(0, num_test_samples):
            confidence = confs[i]
            prediction = preds[i]
            label = labels[i]
            binn = int(math.ceil(((num_bins * confidence) - 1)))
            if binn == num_bins:
                binn = binn - 1
            if binn == -1:
                binn = 0

            bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
            bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
            bin_dict[binn][ACC] = bin_dict[binn][ACC] + (1 if (label == prediction) else 0)
    else:
        for i in range(0, num_test_samples):
            confidence = confs[i]
            index = i // len(preds[0])
            index_c = i % len(preds[0])
            # print(index)
            prediction = preds[index][index_c]
            label = labels[i]
            binn = int(math.ceil(((num_bins * confidence) - 1)))
            if binn == num_bins:
                binn = binn - 1
            bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
            bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
            bin_dict[binn][ACC] = bin_dict[binn][ACC] + (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / float(bin_dict[binn][COUNT])
    return bin_dict

def expected_calibration_error(confs, preds, labels, batch=False, num_bins=15):
    bin_dict = _populate_bins(confs, preds, labels, batch, num_bins)
    num_samples = len(labels)
    ece = 0
    acc = 0
    ce = []
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * abs(bin_accuracy - bin_confidence)
        ce.append(abs(bin_accuracy - bin_confidence))
        acc += (bin_accuracy * bin_count) / num_samples
    # return ece, acc, max(ce).item()
    return ece, acc, bin_dict

def reliability_diagram_plot(confs, preds, labels, network_type, batch=False, num_bins=15):

    ece, acc, bin_dict = expected_calibration_error(confs, preds, labels, batch, num_bins)

    bar_width = 1 / num_bins
    bbins = np.linspace(bar_width / 2, 1 - bar_width / 2, num_bins)
    x = np.linspace(bar_width / 2, 1 - bar_width / 2, num_bins)
    bar_width = 1 / num_bins

    ECE = torch.round(ece*100, decimals=3).cpu()
    ece = round(ECE.numpy().tolist(), 3)
    accuracy = round(acc*100, 3)

    plt.figure()
    left, width = 0.1, 0.8
    bottom, height = 0.1, 0.1
    line1 = [left, bottom, width, 0.23]
    line2 = [left, 0.4, width, 0.5]
    ax1 = plt.axes(line2) #upper
    ax2 = plt.axes(line1) #below

    ax1.grid(True, linestyle='dashed', alpha=0.5)
    ax1.set_xlim(0.4, 1)
    bin_conf, bin_acc, bin_count = [0] * num_bins, [0] * num_bins, [0] * num_bins
    for i in range(num_bins):
        bin_conf[i], bin_acc[i], bin_count[i] = bin_dict[i][BIN_CONF], bin_dict[i][BIN_ACC], bin_dict[i][COUNT]
        bin_count[i] = bin_count[i] / len(labels)
    converted_bin_conf = []
    for item in bin_conf:
        if isinstance(item, torch.Tensor):
            converted_bin_conf.append(item.item())  # Convert the tensor to a Python float
        else:
            converted_bin_conf.append(item)

    ax1.set_ylabel('Test Accuracy / Test Confidence')
    ax1.bar(x, converted_bin_conf, bar_width, align='center', facecolor='r', edgecolor='black', label='Gap', hatch='/', alpha=0.3)
    ax1.bar(x, bin_acc, bar_width, align='center', facecolor='b', edgecolor='black', label='Outputs', alpha=0.75)
    ax1.text(0.7, 0.25, r'ECE={}'.format(ece), fontsize=16, bbox=dict(facecolor='lightskyblue', alpha=0.9))
    ax1.text(0.7, 0.1, r'Acc={}'.format(accuracy), fontsize=16, bbox=dict(facecolor='lightskyblue', alpha=0.9))
    ax1.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=3)
    ax1.legend()

    ax2.set_xlabel('Confidence')
    ax2.bar(bbins, bin_count, bar_width, align='center', facecolor='blue', edgecolor='black', label='Gap', alpha=0.7)
    ax2.grid(True, linestyle='dashed', alpha=0.5)
    ax2.set_xlim(0.4, 1)
    ax2.set_ylabel('Sampling frequency')
    ax2.set_ylim(0, 1)
    plt.savefig(f'./{network_type}.jpeg', dpi=1000)
    plt.show()

def sample_groups(N: int, sizes):
    sizes = list(map(int, sizes))
    need = sum(sizes)
    if need > N:
        raise ValueError(f"sizes sum to {need} > N={N}; impossible without replacement.")

    perm = torch.randperm(N)
    groups = list(torch.split(perm[:need], sizes))
    leftover = perm[need:]

    return groups, leftover

def gaussian_kernel(x1: torch.Tensor, x2: torch.Tensor, bandwidth) -> torch.Tensor:

    if len(x1.shape) == 1:
        x1 = x1.unsqueeze(0)
    if len(x2.shape) == 1:
        x2 = x2.unsqueeze(0)

    _, d = x1.shape
    dist_sq = torch.cdist(x1, x2, p=2).pow(2)

    return torch.exp(-dist_sq / (2 * bandwidth ** 2))


# def sample_gaussian_kernel(x, h, clamp=(-4, 4)): # for qa dataset
def sample_gaussian_kernel(x, h, clamp=(-3, 3)):  # for image classification

    x_tilde = x + h * torch.randn_like(x)

    return x_tilde.clamp(*clamp) if clamp is not None else x_tilde

def sample_box_kernel_l2ball(x, h, clamp=(0., 1.)):

    B = x.shape[0] if x.dim() > 1 else 1
    flat = x.view(B, -1)
    d = flat.size(1)
    g = torch.randn_like(flat)
    g = g / (torch.linalg.norm(g, dim=1, keepdim=True) + 1e-12)
    r = h * torch.rand(B, 1, device=x.device, dtype=x.dtype).pow(1.0 / d)
    x_tilde = (flat + g * r).view_as(x)
    return x_tilde.clamp(*clamp) if clamp is not None else x_tilde

def box_kernel(X1: torch.Tensor, X2: torch.Tensor, bandwidth) -> torch.Tensor:
    distances = torch.cdist(X1, X2, p=2)

    return (distances <= bandwidth).float()


def localized_conformal_prediction(x_cal: torch.Tensor,
                                   y_cal: torch.Tensor,
                                   x_test: torch.Tensor,
                                   cal_scores,
                                   edge_test_probs,
                                   alpha: float,
                                   bandwidth: float,
                                   randomness: bool = True,
                                   kernel: str = 'gaussian') -> torch.Tensor:

    n_cal = len(y_cal)
    n_test = len(x_test)
    num_classes = edge_test_probs.shape[1]
    cal_scores = cal_scores
    prediction_sets = torch.zeros(n_test, num_classes, dtype=torch.bool)

    for i, x in enumerate(x_test):

        if randomness:
            if kernel == 'gaussian':
                x_single = sample_gaussian_kernel(x.unsqueeze(0), bandwidth)
            elif kernel == 'box':
                x_single = sample_box_kernel_l2ball(x.unsqueeze(0), bandwidth)
        else:
            x_single = x.unsqueeze(0)

        if kernel == "gaussian":
            kernel_cal = gaussian_kernel(x_cal, x_single, bandwidth).squeeze()
            kernel_self = gaussian_kernel(x.unsqueeze(0), x_single, bandwidth).item()
        elif kernel == "box":
            kernel_cal = box_kernel(x_cal, x_single, bandwidth).squeeze()
            kernel_self = box_kernel(x.unsqueeze(0), x_single, bandwidth).item()

        denominator = kernel_cal.sum() + kernel_self
        w_xi = kernel_cal / denominator
        w_x = kernel_self / denominator

        all_scores = torch.cat([cal_scores, torch.tensor([float('inf')], device=cal_scores.device)])
        all_weights = torch.cat([w_xi, torch.tensor([w_x], device=w_xi.device)])

        sorted_indices = torch.argsort(all_scores)
        sorted_scores = all_scores[sorted_indices]
        sorted_weights = all_weights[sorted_indices]

        cumsum_weights = torch.cumsum(sorted_weights, dim=0)
        quantile_level = 1 - alpha
        idx = torch.searchsorted(cumsum_weights, quantile_level)
        idx = torch.clamp(idx, 0, len(sorted_scores) - 1)
        q_local = sorted_scores[idx]

        for y in range(num_classes):
            y_candidate = torch.tensor([y], dtype=torch.long, device=x_test.device)
            score = -torch.log(edge_test_probs[i, y_candidate]).item()
            prediction_sets[i, y] = score <= q_local

    return prediction_sets




