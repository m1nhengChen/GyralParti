import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import vtk_to_numpy
 
# --------- Set up subject id and mapping file ---------
subject_id = "110411"  # Set your subject id here

# Load the mapping file (make sure the file exists and adjust the path as needed)
# The mapping file is assumed to have columns: 'subject', 'ref_cluster', 'real_cluster'
mapping_df = pd.read_csv("../mapping/4/599671_temp_subject_cluster_mapping_cosine_lh.csv")
 
# Filter the mapping for the given subject and build a dictionary mapping:
# key = real cluster (from the subject's assignment file)
# value = corresponding ref cluster (which determines the color)
subject_mapping = mapping_df[mapping_df['subject_id'] == int(subject_id)]
real_to_ref = dict(zip(subject_mapping['subject_cluster'], subject_mapping['ref_cluster']))
 
# --------- Load Assignment Matrix and Excel File ---------
assignment_file_path = f"./cluster_results/clusters_4/{subject_id}_cluster_assignments_lh.npy"
assignment_matrix = np.load(assignment_file_path)
point_indices = assignment_matrix[:, 0].astype(int)
print("Point indices from assignment file:", point_indices)
cluster_labels = assignment_matrix[:, 1]  # these are the subject's (real) cluster numbers
 
excel_file_path = f"/HCP/{subject_id}/GyralNet_files/{subject_id}_info_lh.xlsx"
df = pd.read_excel(excel_file_path, sheet_name="GyralNet_Graph")
point_ids = df["center 3hinge ID"].values.astype(int)
print("Point IDs from Excel:", point_ids)
 
# --------- Load Brain Surface VTK File ---------
vtk_file_path = f"HCP/{subject_id}/lh.white.dsi_fa.vtk"
reader = vtk.vtkPolyDataReader()
reader.SetFileName(vtk_file_path)
reader.Update()
polydata = reader.GetOutput()
 
# Extract point coordinates
points = vtk_to_numpy(polydata.GetPoints().GetData())
 
# --------- Define the Reference Cluster Colors ---------
# These colors are keyed by the reference cluster numbers (0–7)
# cluster_colors = {
#     0: (255, 0, 0),       # red
#     1: (0, 255, 0),       # green
#     2: (0, 0, 255),       # blue
#     3: (255, 255, 0),     # yellow
#     4: (255, 0, 255),     # magenta
#     5: (0, 255, 255),     # cyan
#     6: (255, 165, 0),     # orange
#     7: (128, 0, 128)      # purple
# }
 
cluster_colors = {
    0: (255, 0, 0),       # red
    1: (0, 255, 0),       # green
    2: (0, 0, 255),       # blue
    3: (128, 0, 128)      # purple
}
 
# cluster_colors = {
#     0: (255, 0, 255),     # magenta
#     1: (0, 255, 255),     # cyan
#     2: (255, 165, 0),     # orange
#     3: (255, 255, 0),     # yellow
# }
 
# --------- Set up VTK Filters and Writers for Each Cluster ---------
# Here we loop over the unique real cluster numbers (from the subject's assignment)
unique_real_clusters = np.unique(cluster_labels)
cluster_append_filters = {}
cluster_writers = {}
for real_cluster in unique_real_clusters:
    # Look up the corresponding reference cluster
    ref_cluster = real_to_ref.get(real_cluster, None)
    # Use the ref cluster number for naming (if not found, fall back to the real cluster number)
    cluster_identifier = ref_cluster if ref_cluster is not None else real_cluster
    cluster_append_filters[real_cluster] = vtk.vtkAppendPolyData()
    cluster_writers[real_cluster] = vtk.vtkPolyDataWriter()
    cluster_writers[real_cluster].SetFileName(f"visual_4_lh/{subject_id}_4_cluster_{cluster_identifier}_lh.vtk")
 
# Create mapping from point IDs (from the Excel file) to real cluster labels (from the assignment)
point_index_map = dict(zip(point_ids, cluster_labels))
print("Point ID to cluster mapping:", point_index_map)
 
# --------- Create Spheres for Each Point with the Correct Color ---------
for point_id in point_ids:
    if point_id in point_index_map:
        real_cluster = point_index_map[point_id]
        # Look up the corresponding reference cluster to determine the color
        ref_cluster = real_to_ref.get(real_cluster, None)
        if ref_cluster is not None:
            rgb = cluster_colors.get(ref_cluster, (200, 200, 200))
        else:
            rgb = (200, 200, 200)  # fallback color if no mapping exists
 
        # Create a sphere at the specified point
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(points[point_id])
        sphere.SetRadius(2.0)  # Adjust sphere size as needed
        sphere.Update()
 
        # Create a color array for the sphere cells
        color_array = vtk.vtkUnsignedCharArray()
        color_array.SetNumberOfComponents(3)
        color_array.SetName("Colors")
 
        # Assign the color to each cell in the sphere
        n_cells = sphere.GetOutput().GetNumberOfCells()
        for _ in range(n_cells):
            color_array.InsertNextTuple(rgb)
 
        # Attach the color array to the sphere's cell data
        sphere.GetOutput().GetCellData().AddArray(color_array)
        sphere.GetOutput().GetCellData().SetScalars(color_array)
 
        # Add the colored sphere to the corresponding cluster append filter (keyed by real cluster)
        cluster_append_filters[real_cluster].AddInputData(sphere.GetOutput())
 
# --------- Finalize and Write the VTK Files for Each Cluster ---------
for real_cluster, append_filter in cluster_append_filters.items():
    append_filter.Update()
    polydata_output = vtk.vtkPolyData()
    polydata_output.ShallowCopy(append_filter.GetOutput())
    
    writer = cluster_writers[real_cluster]
    writer.SetInputData(polydata_output)
    writer.Write()
