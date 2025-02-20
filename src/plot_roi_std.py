import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def load_labels(label_file):
    """Load ROI labels from labels.txt"""
    with open(label_file, "r") as file:
        labels = [line.strip() for line in file]
    return labels

def load_cluster_data(cluster_file):
    """Load cluster assignment data from .npy file"""
    data = np.load(cluster_file)
    return data[:, 1].astype(int)  # Extract only cluster assignments

def load_roi_info(excel_file):
    """Load ROI information from the given Excel file"""
    df = pd.read_excel(excel_file)
    return df["center 3hinge label"].tolist()

def load_cluster_mapping(mapping_file):
    """Load subject-specific cluster mapping from CSV.
    CSV is expected to have three columns: subject_id, subject_cluster, ref_cluster.
    """
    df = pd.read_csv(mapping_file)
    mapping_dict = {(str(row["subject_id"]), int(row["subject_cluster"])): int(row["ref_cluster"]) for _, row in df.iterrows()}
    # print(mapping_dict)
    return mapping_dict

def remap_clusters(subject, clusters, mapping_dict):
    """Remap subject-specific cluster IDs to reference cluster IDs"""
    remapped_clusters = np.array([mapping_dict.get((subject, c), -1) for c in clusters])
    if -1 in remapped_clusters:
        print(f"Warning: Some clusters in subject {subject} were not found in the mapping file.")
    return remapped_clusters

def process_subject(subject, cluster_dir, dataset_dir, labels, mapping_lh, mapping_rh):
    """Process a single subject and compute cluster-wise ROI statistics"""
    # Construct file paths
    cluster_file_lh = os.path.join(cluster_dir, f"{subject}_cluster_assignments_lh.npy")
    cluster_file_rh = os.path.join(cluster_dir, f"{subject}_cluster_assignments_rh.npy")
    roi_file_lh = os.path.join(dataset_dir, subject, "GyralNet_files", f"{subject}_info_lh.xlsx")
    roi_file_rh = os.path.join(dataset_dir, subject, "GyralNet_files", f"{subject}_info_rh.xlsx")

    # Load data
    if not os.path.exists(cluster_file_lh) or not os.path.exists(cluster_file_rh):
        print(f"Missing cluster files for subject: {subject}")
        return None

    clusters_lh = load_cluster_data(cluster_file_lh)
    clusters_rh = load_cluster_data(cluster_file_rh)
    rois_lh = load_roi_info(roi_file_lh)
    rois_rh = load_roi_info(roi_file_rh)

    # Ensure ROI data and cluster assignments have the same length
    if len(clusters_lh) != len(rois_lh) or len(clusters_rh) != len(rois_rh):
        print(f"Data mismatch in subject {subject}: cluster and ROI count do not match")
        return None

    # Remap clusters using mapping files
    clusters_lh = remap_clusters(subject, clusters_lh, mapping_lh)
    clusters_rh = remap_clusters(subject, clusters_rh, mapping_rh)

    # Initialize cluster-wise ROI statistics (for each cluster, count occurrence of each ROI)
    num_clusters = 8
    # For left hemisphere, use first 75 labels; for right, use remaining 75 labels
    roi_counts_lh = {i: {roi: 0 for roi in labels[:75]} for i in range(num_clusters)}
    roi_counts_rh = {i: {roi: 0 for roi in labels[75:]} for i in range(num_clusters)}

    # Count occurrences of ROIs in each cluster for left hemisphere
    for node_idx, cluster_id in enumerate(clusters_lh):
        if cluster_id in roi_counts_lh:
            roi = rois_lh[node_idx]
            if roi in roi_counts_lh[cluster_id]:
                roi_counts_lh[cluster_id][roi] += 1

    # Count occurrences of ROIs in each cluster for right hemisphere
    for node_idx, cluster_id in enumerate(clusters_rh):
        if cluster_id in roi_counts_rh:
            roi = rois_rh[node_idx]
            if roi in roi_counts_rh[cluster_id]:
                roi_counts_rh[cluster_id][roi] += 1

    return roi_counts_lh, roi_counts_rh

def compute_statistics(all_subjects_stats, num_clusters, num_rois):
    """Compute mean and standard deviation across subjects for each cluster and ROI.
    Returns:
        mean_matrix: num_clusters x num_rois matrix of means.
        std_matrix: num_clusters x num_rois matrix of standard deviations.
        roi_list: list of ROI names.
    """
    roi_list = list(next(iter(all_subjects_stats.values()))[0].keys())  # Get ROI names

    # Build an array: shape = (num_subjects, num_clusters, num_rois)
    subject_data = np.zeros((len(all_subjects_stats), num_clusters, num_rois))

    for s_idx, (_, cluster_data) in enumerate(all_subjects_stats.items()):
        for c in range(num_clusters):
            subject_data[s_idx, c, :] = [cluster_data[c][roi] for roi in roi_list]

    # Compute statistics across subjects (axis=0)
    mean_matrix = np.mean(subject_data, axis=0)
    std_matrix = np.std(subject_data, axis=0)

    return mean_matrix, std_matrix, roi_list

# def plot_boxplots(std_matrix, hemisphere):
#     """Plot a boxplot for each cluster based on std values for the given hemisphere.
#     std_matrix shape is (8, 75): 8 clusters, 75 ROI std values per cluster.
#     """
#     # Prepare data: list of arrays, each array is std values for one cluster
#     data = [std_matrix[i, :] for i in range(std_matrix.shape[0])]
    
#     plt.figure(figsize=(10, 6))
#     plt.boxplot(data, notch=True)
#     plt.xlabel("Cluster ID")
#     plt.ylabel("Standard Deviation")
#     plt.title(f"{hemisphere} Hemisphere ROI Std Dev across 75 ROIs for each Cluster")
#     plt.xticks(range(1, 9), [str(i) for i in range(8)])
#     plt.show()
    
def plot_boxplots(std_matrix, hemisphere):
    """
    Plot a boxplot for each cluster based on std values for the given hemisphere.
    std_matrix shape is (8, 75): 8 clusters, 75 ROI std values per cluster.
    """

    # Prepare the data in a long-form DataFrame where each record has "Cluster" and "Std"
    data = []
    for cluster_id in range(std_matrix.shape[0]):
        for val in std_matrix[cluster_id]:
            data.append({"Cluster": f"Cluster {cluster_id}", "Std": val})

    df = pd.DataFrame(data)

    # Set a Seaborn theme
    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 6))

    # Configure outlier (flier) properties, e.g., diamond-shaped markers
    flierprops = dict(marker='D', markerfacecolor='black', markersize=15, linestyle='none')

    # Create the boxplot
    sns.boxplot(
        x="Cluster",
        y="Std",
        data=df,
        palette="Set3",         # Color palette
        notch=False,             # Use notched boxes
        width=0.6,              # Box width
        fliersize=100,            # Outlier marker size
        linewidth=2,            # Box outline width
        flierprops=flierprops,   # Outlier properties
        whis=[0.01, 99.99]
    )

    # plt.title(f"{hemisphere} Hemisphere ROI Std Dev across 75 ROIs for each Cluster", fontsize=40)
    # plt.xlabel("Cluster ID", fontsize=20)
    plt.ylabel("Standard Deviation", fontsize=52)
    plt.xlabel("")
    # plt.ylabel("")
    # Rotate x-axis labels if needed:
    # plt.xticks(rotation=45)
    # plt.xticks(fontsize=36)
    plt.xticks([])
    plt.yticks(fontsize=42)
    # Adjust the layout so everything fits
    plt.tight_layout()
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()


def main():
    """ Main function to parse arguments and process all subjects """

    parser = argparse.ArgumentParser(description="Cluster-wise ROI statistics")
    parser.add_argument("--cluster_dir", type=str, default="cluster_results/clusters_8", help="Path to the clustering results root directory")
    parser.add_argument("--dataset_dir", type=str, default="Your own dataset root dir", help="Path to the dataset root directory")
    parser.add_argument("--label_file", type=str, default="./labels.txt",required=False, help="Path to the labels.txt file")

    parser.add_argument("--mapping_lh", type=str, default="../mapping/8/599671_temp_subject_cluster_mapping_cosine_lh.csv", help="CSV file mapping left hemisphere cluster IDs")
    parser.add_argument("--mapping_rh", type=str, default="../mapping/8/599671_temp_subject_cluster_mapping_cosine_rh.csv", help="CSV file mapping right hemisphere cluster IDs")

    args = parser.parse_args()

    # Load ROI labels
    labels = load_labels(args.label_file)

    # Load cluster ID mappings
    mapping_lh = load_cluster_mapping(args.mapping_lh)
    mapping_rh = load_cluster_mapping(args.mapping_rh)

    # Get all subject names from the dataset directory
    subjects = [subj for subj in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, subj))]

    # Store statistics for all subjects
    all_subjects_stats_lh = {}
    all_subjects_stats_rh = {}

    for subject in subjects:
        result = process_subject(subject, args.cluster_dir, args.dataset_dir, labels, mapping_lh, mapping_rh)
        if result:
            all_subjects_stats_lh[subject], all_subjects_stats_rh[subject] = result

    # Compute statistics for left and right hemisphere
    mean_lh, std_lh, roi_lh = compute_statistics(all_subjects_stats_lh, 8, 75)
    mean_rh, std_rh, roi_rh = compute_statistics(all_subjects_stats_rh, 8, 75)

    # Output computed mean and standard deviation matrices
    print("Left Hemisphere Mean:\n", mean_lh)
    print("Left Hemisphere Std Dev:\n", std_lh)
    print("\nRight Hemisphere Mean:\n", mean_rh)
    print("Right Hemisphere Std Dev:\n", std_rh)
    
    # Plot boxplots for standard deviation matrices
    plot_boxplots(std_lh, "Left")
    plot_boxplots(std_rh, "Right")

if __name__ == "__main__":
    main()
