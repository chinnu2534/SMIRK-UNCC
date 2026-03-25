# SMIRK-UNCC (SMIRK V2.0)

This repository holds the customized setup, infrastructure, and evaluation extensions for **SMIRK** (3D Facial Expressions through Analysis-by-Neural-Synthesis). It includes the core SMIRK implementation, required FLAME dataset dependencies, custom Python scripts for FLAME integration, and evaluation notebooks (developed as Milestone 2).

## Project Structure

- **`/smirk_repo/`**: The core SMIRK repository. Contains the model architecture, pretrained models, training configurations, and original SMIRK source code. 
- **`/milestone 2/`**: Jupyter notebooks for running inference and evaluation on the SMIRK models across different environments.
  - `SMIRK_Evaluation_PhD.ipynb`
  - `SMIRK_Inference_Colab.ipynb`
  - `SMIRK_Inference_Lightning.ipynb`
- **`/space_files/`**: Contains large dependencies such as `FLAME2020.zip`, which are tracked via Git LFS.
- **`build_eval_notebook.py`**: Helper script to construct or format the evaluation notebooks.
- **`fix_numpy.py`**: Utility script to fix NumPy versioning issues or array generation bugs in the environments.
- **`test_flame.py` / `upload_flame.py`**: Scripts used to configure, test, and upload the FLAME models (e.g., securely pushing them to Hugging Face datasets).

## How to Run

### 1. Environment Setup
To run the models, you must first set up the dependencies for the main SMIRK implementation. This typically requires Python 3.9 and PyTorch with CUDA support.

Navigate to the core repository:
```bash
cd smirk_repo
conda create -n smirk python=3.9
conda activate smirk
pip install -r requirements.txt

# Install PyTorch3D (Adjust for your CUDA version)
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt201/download.html

# Download required SMIRK pre-trained models
bash quick_install.sh
```

### 2. Handling Large FLAME Files (Git LFS)
The archive `FLAME2020.zip` is stored via Git LFS in the `space_files/` directory. Make sure you have [Git LFS](https://git-lfs.com/) installed and run:
```bash
git lfs pull
```
*Note: The `upload_flame.py` script can be used to deploy the FLAME assets to HF datasets if running in cloud environments where LFS pull is restricted natively.*

### 3. Inference and Evaluation
We provide several environments for running the evaluation found in the `/milestone 2` directory:
- **Google Colab:** Upload and run `SMIRK_Inference_Colab.ipynb`.
- **Local / Lightning:** Run `SMIRK_Inference_Lightning.ipynb`.
- **PhD Evaluation:** Run `SMIRK_Evaluation_PhD.ipynb`.

Alternatively, you can run the standard SMIRK demos directly from `/smirk_repo/`:
```bash
cd smirk_repo
python demo.py --input_path samples/test_image2.png --out_path results/ --checkpoint pretrained_models/SMIRK_em1.pt --crop
```
