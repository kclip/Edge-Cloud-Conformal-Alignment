import argparse
import torch
import sys
import os
import matplotlib
matplotlib.use("TkAgg")
import utils as pf

sys.path.append(os.path.dirname(sys.path[0]))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='qa', help='dataset')
parser.add_argument('--alpha', type=float, default=0.2, help='target level for conditional coverage')
parser.add_argument('--sigma', type=float, default=20, help='kernel bandwidth for LCP')
parser.add_argument('--delta', type=float, default=0.05, help='target level for average satisfaction rate')




def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Current deploy device is {device}')

    """
    ============================================================
    Loading the inference data
    ============================================================
    """
    if 'cifar' in args.dataset:
        inference_results = torch.load(f'./cifar100/inference_results/cifar100_inference_results.pt')
        inputs, outputs = inference_results['inputs'].to(device), inference_results['outputs'].to(device)
    elif args.dataset == 'qa':
        inference_results = torch.load(f'./qa/4options/qa_inference_results.pt', weights_only=False)
        embedding_results = torch.load(f'./qa/4options/qa_embeddings_results.pt', weights_only=False)
        inputs, outputs = embedding_results.to(device), inference_results['outputs'].to(device)

    cloud_test_probs, edge_test_probs = inference_results['large_test_probs'].to(device), inference_results['small_test_probs'].to(device)

    cloud_pred_list, edge_conventional_pred_list, edge_CP_pred_list, edge_LCP_pred_list = [], [], [], []
    cas_conv_list, cas_CP_list, cas_LCP_list, CA_conv_list, CA_CP_list, CA_LCP_list = [], [], [], [], [], []

    alphas = [0.3, 0.25, 0.2, 0.15, 0.1]
    deltas = [0.2]
    for alpha in alphas:
        for delta in deltas:
            print(f'current alpha {alpha}, delta {delta}')
            for _ in range(200):
                
                #######################  Partition the dataset for different purposes  #######################
                if 'cifar' in args.dataset:
                    groups, rest = pf.sample_groups(10000, [500, 200, 500, 100])
                elif args.dataset == 'qa':
                    groups, rest = pf.sample_groups(1720, [500, 200, 500, 100])
                cal_data, training_data, val_data, test_data = groups

                """
                ============================================================
                Measure the calibration performance of the edge model by treating cloud as reference
                ============================================================
                """

                #######################  Generate the label by treating cloud as reference  #######################
                ground_truth_labels = torch.multinomial(cloud_test_probs, num_samples=1)

                #######################  Evaluate the edge model  #######################
                confidence, pred_label = torch.max(edge_test_probs.data, dim=1)
                ECE, accuracy, _ = pf.expected_calibration_error(confidence, pred_label, ground_truth_labels, num_bins=15)
                pf.reliability_diagram_plot(confidence, pred_label, ground_truth_labels, 'qa')
                print(f'edge model on test data set: ECE is {ECE * 100}%, accuracy is {accuracy * 100}%')

                """
                ============================================================
                Construct all prediction set
                ============================================================
                """

                #######################  Construct the cloud HMS #######################
                cloud_mask, cloud_sizes = pf.minimal_prediction_set(cloud_test_probs, alpha=alpha)
                cloud_mask = cloud_mask.to(torch.int).float()
                cloud_coverage = cloud_mask[torch.arange(len(ground_truth_labels)), ground_truth_labels].mean()
                cloud_prediction_size = cloud_sizes.float().mean()

                cloud_prediction = {
                    "coverage": cloud_coverage.item() if torch.is_tensor(cloud_coverage) else cloud_coverage,
                    "normalized inefficiency": cloud_prediction_size.item() if torch.is_tensor(cloud_prediction_size) else cloud_prediction_size,
                    'alpha': alpha,
                    'delta': delta
                }
                cloud_pred_list.append(cloud_prediction)

                #######################  Construct the edge HMS  #######################
                edge_conventional_mask, edge_conventional_sizes = pf.minimal_prediction_set(edge_test_probs, alpha=alpha)
                edge_conventional_mask = edge_conventional_mask.to(torch.int).float()
                edge_conventional_coverage = edge_conventional_mask[torch.arange(len(ground_truth_labels)), ground_truth_labels][test_data].mean()

                cond_prob = torch.mul(edge_conventional_mask, cloud_test_probs).sum(dim=1)
                avg_sat_rate = (cond_prob >= 1 - alpha)[test_data].to(torch.int).float().mean()

                edge_conventional_prediction_size = torch.mul(edge_conventional_sizes[test_data].float(), 1/cloud_sizes[test_data].float()).mean()

                edge_conventional_prediction = {
                    "coverage": edge_conventional_coverage.item() if torch.is_tensor(edge_conventional_coverage) else edge_conventional_coverage,
                    "normalized inefficiency": edge_conventional_prediction_size.item() if torch.is_tensor(edge_conventional_prediction_size) else edge_conventional_prediction_size,
                    'avg_sat_rate': avg_sat_rate.item() if torch.is_tensor(avg_sat_rate) else avg_sat_rate,
                    'alpha': alpha,
                    'delta': delta
                }
                edge_conventional_pred_list.append(edge_conventional_prediction)

                #######################  Construct the edge conformal prediction set  #######################
                cal_labels = ground_truth_labels[cal_data]
                NC_score_set = -torch.log(edge_test_probs[cal_data.reshape(1, -1), cal_labels.reshape(1, -1)])
                k = int(torch.ceil(torch.tensor((len(cal_data) + 1) * (1 - alpha))).item())
                k = min(max(k, 1), len(cal_data))
                edge_CP_threshold = NC_score_set.kthvalue(k).values.item()

                edge_CP_mask_test = (-torch.log(edge_test_probs[test_data]) <= edge_CP_threshold).to(torch.int).float()
                edge_CP_mask_val = (-torch.log(edge_test_probs[val_data]) <= edge_CP_threshold).to(torch.int).float()
                edge_CP_mask_training = (-torch.log(edge_test_probs[training_data]) <= edge_CP_threshold).to(torch.int).float()

                edge_CP_sizes = torch.mul(edge_CP_mask_test.sum(dim=1), 1/cloud_sizes[test_data].float()).mean()
                edge_CP_coverage = edge_CP_mask_test[torch.arange(len(test_data)), ground_truth_labels[test_data].reshape(1, -1)].mean()

                cond_prob = torch.mul(edge_CP_mask_test, cloud_test_probs[test_data]).sum(dim=1)
                avg_sat_rate = (cond_prob >= 1 - alpha).to(torch.int).float().mean()

                edge_CP_prediction = {
                    "coverage": edge_CP_coverage.item() if torch.is_tensor(edge_CP_coverage) else edge_CP_coverage,
                    "normalized inefficiency": edge_CP_sizes.item() if torch.is_tensor(edge_CP_sizes) else edge_CP_sizes,
                    'avg_sat_rate': avg_sat_rate.item() if torch.is_tensor(avg_sat_rate) else avg_sat_rate,
                    'alpha': alpha,
                    'delta': delta
                }
                edge_CP_pred_list.append(edge_CP_prediction)

                #######################  Construct the edge LCP set  #######################
                test_inputs_features, cal_inputs_features = inputs[test_data].reshape(len(test_data), -1), inputs[cal_data].reshape(len(cal_data), -1)
                val_inputs_features, training_inputs_features = inputs[val_data].reshape(len(val_data), -1), inputs[training_data].reshape(len(training_data), -1)

                sigma = args.sigma
                edge_LCP_mask_test = pf.localized_conformal_prediction(cal_inputs_features, ground_truth_labels[cal_data], test_inputs_features, NC_score_set.squeeze(), edge_test_probs[test_data], alpha, sigma, randomness=True, kernel='gaussian').to(torch.int).float().to('cuda')
                edge_LCP_mask_val = pf.localized_conformal_prediction(cal_inputs_features, ground_truth_labels[cal_data], val_inputs_features, NC_score_set.squeeze(), edge_test_probs[val_data], alpha, sigma, randomness=True, kernel='gaussian').to(torch.int).float().to('cuda')
                edge_LCP_mask_training = pf.localized_conformal_prediction(cal_inputs_features, ground_truth_labels[cal_data], training_inputs_features, NC_score_set.squeeze(), edge_test_probs[training_data], alpha, sigma, randomness=True, kernel='gaussian').to(torch.int).float().to('cuda')

                edge_LCP_size = torch.mul(edge_LCP_mask_test.sum(dim=1), 1/cloud_sizes[test_data].float()).mean()
                edge_LCP_coverage = edge_LCP_mask_test[torch.arange(len(test_data)), ground_truth_labels[test_data].reshape(1, -1)].mean()

                cond_prob = torch.mul(edge_LCP_mask_test, cloud_test_probs[test_data]).sum(dim=1)
                avg_sat_rate = (cond_prob >= 1 - alpha).to(torch.int).float().mean()

                edge_LCP_prediction = {
                    "coverage": edge_LCP_coverage.item() if torch.is_tensor(edge_LCP_coverage) else edge_LCP_coverage,
                    "normalized inefficiency": edge_LCP_size.item() if torch.is_tensor(edge_LCP_size) else edge_LCP_size,
                    'avg_sat_rate': avg_sat_rate.item() if torch.is_tensor(avg_sat_rate) else avg_sat_rate,
                    'alpha': alpha,
                    'delta': delta
                }
                edge_LCP_pred_list.append(edge_LCP_prediction)


                """
                ============================================================
                Adopt the conventional model cascading mechanism, i.e., confidence-based deferral (CbD)
                ============================================================
                """
                gamma = 1 - delta

                #######################  edge-HMS model cascading  #######################
                conventional_cas_results = pf.model_cascading(edge_test_probs[test_data], cloud_test_probs[test_data], edge_conventional_mask[test_data], cloud_mask[test_data], gamma=gamma, alpha=alpha, delta=delta)
                cas_conv_list.append(conventional_cas_results)

                #######################  CP-Based model cascading  #######################
                CP_cas_results = pf.model_cascading(edge_test_probs[test_data], cloud_test_probs[test_data], edge_CP_mask_test, cloud_mask[test_data], gamma=gamma, alpha=alpha, delta=delta)
                cas_CP_list.append(CP_cas_results)

                #######################  LCP-Based model cascading  #######################
                LCP_cas_results = pf.model_cascading(edge_test_probs[test_data], cloud_test_probs[test_data], edge_LCP_mask_test, cloud_mask[test_data], gamma=gamma, alpha=alpha, delta=delta)
                cas_LCP_list.append(LCP_cas_results)

                """
                ============================================================
                Adopt the conformal alignment-based (CAb) model cascading mechanism
                ============================================================
                """
                #######################  CA only  #######################
                CA_results = pf.CA(edge_test_probs, cloud_test_probs, cloud_mask, training_data, val_data, test_data, edge_conventional_mask[training_data], edge_conventional_mask[val_data], edge_conventional_mask[test_data], alpha, delta)
                CA_conv_list.append(CA_results)

                #######################  CA+CP  #######################
                CA_CP_results = pf.CA(edge_test_probs, cloud_test_probs, cloud_mask, training_data, val_data, test_data, edge_CP_mask_training, edge_CP_mask_val, edge_CP_mask_test, alpha, delta)
                CA_CP_list.append(CA_CP_results)

                #######################  CA+LCP  #######################
                CA_LCP_results = pf.CA(edge_test_probs, cloud_test_probs, cloud_mask, training_data, val_data, test_data, edge_LCP_mask_training, edge_LCP_mask_val, edge_LCP_mask_test, alpha, delta)
                CA_LCP_list.append(CA_LCP_results)

    final_results = {
        'cloud_pred_list': cloud_pred_list,
        'edge_conventional_pred_list': edge_conventional_pred_list,
        'edge_CP_pred_list': edge_CP_pred_list,
        'edge_LCP_pred_list': edge_LCP_pred_list,
        'cas_conv_list': cas_conv_list,
        'cas_CP_list': cas_CP_list,
        'cas_LCP_list': cas_LCP_list,
        'CA_conv_list': CA_conv_list,
        'CA_CP_list': CA_CP_list,
        'CA_LCP_list': CA_LCP_list
    }

    torch.save(final_results, './results.pt')








if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
