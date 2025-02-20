import os
import csv
import numpy as np
import pandas as pd
from nibabel.freesurfer.io import read_annot
from scipy.optimize import linear_sum_assignment
import concurrent.futures

def get_cluster_atlas_percentage(subject_id, subjects_dir, assignments_dir, hemi="lh"):
    assignment_file = os.path.join(assignments_dir, f"{subject_id}_cluster_assignments_{hemi}.npy")
    excel_file = os.path.join(subjects_dir, subject_id, f"GyralNet_files/{subject_id}_info_{hemi}.xlsx")
    annot_file = os.path.join(subjects_dir, subject_id, f"recon_all_result/label/{hemi}.aparc.a2009s.annot")
    
    assignment_matrix = np.load(assignment_file)
    cluster_ids = assignment_matrix[:, 1].astype(int)
    
    df = pd.read_excel(excel_file, sheet_name="GyralNet_Graph")
    point_ids = df["center 3hinge ID"].values.astype(int)
    
    labels, ctab, names = read_annot(annot_file)
    valid_names = names[1:]  # Remove atlas 0 ("unknown")
    
    point_info = {}
    for i, point_id in enumerate(point_ids):
        cluster_id = cluster_ids[i]
        original_atlas_id = labels[point_ids[i]]
        if original_atlas_id == 0:
            continue
        new_atlas_id = original_atlas_id - 1
        point_info[point_id] = [cluster_id, new_atlas_id]
    
    num_atlas_regions = len(valid_names)
    unique_clusters = np.unique(cluster_ids)
    cluster_atlas_percentage = {}
    for cluster in unique_clusters:
        cluster_points = [info for info in point_info.values() if info[0] == cluster]
        if len(cluster_points) == 0:
            continue
        cluster_atlas_ids = [info[1] for info in cluster_points]
        total_points = len(cluster_atlas_ids)
        percentage_vector = np.zeros(num_atlas_regions)
        for atlas_idx in range(num_atlas_regions):
            count = np.sum(np.array(cluster_atlas_ids) == atlas_idx)
            percentage_vector[atlas_idx] = count / total_points if total_points > 0 else 0
        cluster_atlas_percentage[cluster] = percentage_vector
    return cluster_atlas_percentage, valid_names

def process_subject(subject_id, subjects_dir, assignments_dir, hemi="lh"):
    try:
        cluster_perc, valid_names = get_cluster_atlas_percentage(subject_id, subjects_dir, assignments_dir, hemi)
        return subject_id, cluster_perc, valid_names
    except Exception as e:
        print(f"Skipping subject {subject_id} ({hemi}) due to error: {e}")
        return None

def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8)

def compute_cost(u, v, metric="l2"):
    if metric == "l2":
        return np.linalg.norm(u - v)
    elif metric == "l1":
        return np.linalg.norm(u - v, ord=1)
    elif metric == "linf":
        return np.linalg.norm(u - v, ord=np.inf)
    elif metric == "cosine":
        return 1 - cosine_similarity(u, v)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def get_subject_cluster_mapping(metric, ref_subj, all_subjects_cluster_percentages):
    ref_mapping = all_subjects_cluster_percentages[ref_subj]
    ref_clusters = list(ref_mapping.keys())
    subject_cluster_mapping = {ref_subj: {k: k for k in ref_clusters}}
    for subj_id, cluster_mapping in all_subjects_cluster_percentages.items():
        if subj_id == ref_subj:
            continue
        curr_clusters = list(cluster_mapping.keys())
        cost_matrix = np.zeros((len(ref_clusters), len(curr_clusters)))
        for i, ref_cl in enumerate(ref_clusters):
            for j, curr_cl in enumerate(curr_clusters):
                cost_matrix[i, j] = compute_cost(ref_mapping[ref_cl], cluster_mapping[curr_cl], metric)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        mapping = {}
        for r, c in zip(row_ind, col_ind):
            mapping[curr_clusters[c]] = ref_clusters[r]
        subject_cluster_mapping[subj_id] = mapping
    return subject_cluster_mapping

def relabel_features(mapping, features_dict):
    new_features = {}
    for subj, map_dict in mapping.items():
        if subj not in features_dict:
            continue
        orig_features = features_dict[subj]
        new_feat = {}
        for orig, new in map_dict.items():
            orig_int = int(orig) if isinstance(orig, str) else orig
            new_int = int(new) if isinstance(new, str) else new
            if orig_int in orig_features:
                new_feat[new_int] = orig_features[orig_int]
        new_features[subj] = new_feat
    return new_features

def evaluate_metric(args):
    metric, ref_subj, hemi, subjects_data= args
    mapping_output_file = f"../../graph_embedding/dmon/tracemap_only_cluster_mapping_50/{ref_subj}_temp_subject_cluster_mapping_{metric}_{hemi}.csv"
    
    mapping = get_subject_cluster_mapping(metric, ref_subj, subjects_data)
    with open(mapping_output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["subject_id", "subject_cluster", "ref_cluster"])
        for subj, mapping_dict in mapping.items():
            for subj_cluster, ref_cluster in mapping_dict.items():
                writer.writerow([subj, subj_cluster, ref_cluster])
    print(f"Mapping results for metric '{metric}' (hemi={hemi}) saved to {mapping_output_file}")

    return (metric, ref_subj, hemi)

if __name__ == "__main__":
    subjects_dir = "/8T_ZY/HCP_new"       # Update as needed.
    assignments_dir = "../../graph_embedding/tracemap_only_results/clusters_50"  # Update as needed.
    
    subject_ids = [d for d in os.listdir(subjects_dir)
                   if os.path.isdir(os.path.join(subjects_dir, d))]
    
    with open("ref_subjects.txt", "r") as f:
        ref_subjs = [line.strip() for line in f if line.strip()]
    
    hemis = ["lh", "rh"]
    
    all_subjects_cluster_percentages = {}
    for hemi in hemis:
        print(f"Processing hemisphere: {hemi}")
        hemi_cluster_percentages = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_subject, subject_id, subjects_dir, assignments_dir, hemi): subject_id
                       for subject_id in subject_ids}
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result is None:
                        continue
                    subject_id, cluster_perc, subj_valid_names = result
                    hemi_cluster_percentages[subject_id] = cluster_perc
                except Exception as exc:
                    print(f"Exception while processing subject: {exc}")
        if not hemi_cluster_percentages:
            print(f"No subjects processed successfully for hemisphere {hemi}.")
            continue
        all_subjects_cluster_percentages[hemi] = hemi_cluster_percentages
    
    eval_tasks = []
    for hemi in hemis:
        if hemi not in all_subjects_cluster_percentages:
            continue
        subjects_data = all_subjects_cluster_percentages[hemi]
        for ref_subj in ref_subjs:
            if ref_subj not in subjects_data:
                print(f"Reference subject {ref_subj} not found for hemisphere {hemi}. Skipping.")
                continue
            for metric in ["l2", "l1", "linf", "cosine"]:
                eval_tasks.append((metric, ref_subj, hemi, subjects_data))
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
        futures = [executor.submit(evaluate_metric, task) for task in eval_tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception as exc:
                print(f"Exception during evaluation: {exc}")