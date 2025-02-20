# GyralParti
Official implementation of paper "GyralNet Subnetwork Partitioning via Differentiable Spectral Modularity Optimization"

<img src="/fig/overall.png" width="600px" style="display: block; margin: 0 auto;"/>

1. **Clone the repository:**
```bash
   git clone [https://github.com/username/subnetwork-partitioning.git](https://github.com/m1nhengChen/GyralParti.git)
   cd src
   ```
2. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
3. **Run our code:**
  ```bash
  python subnet_partition.py
  ```
 **Note:**  The Human Connectome Project (HCP) dataset is **restricted-access** due to its licensing and data usage agreements. Therefore, we cannot include or distribute it within this repository. 

If you wish to use the HCP dataset for your research or analyses:
1. Please **request access** through the [official HCP website](https://www.humanconnectome.org/study/hcp-young-adult).
2. Once access is granted, follow their guidelines to **download** the dataset.
3. Place the downloaded data into the appropriate directory (e.g., `data/HCP/`) to integrate it with the scripts and notebooks in this repository.

For more information about data usage terms and conditions, refer to the [HCP Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms). If you have any questions about the usage or the integration of the HCP dataset, feel free to open an issue or contact us directly.
