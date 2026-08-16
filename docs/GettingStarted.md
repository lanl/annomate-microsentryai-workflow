# Getting Started

This guide covers the technical installation, environment setup, and the basic workflow to get your first project up and running.

## 1. Prerequisites
Because this suite relies on heavy machine learning libraries (PyTorch, Anomalib, SAM 2), we highly recommend running it inside an isolated environment.
* **OS:** macOS, Linux, or Windows.
* **Python:** Version **3.10** is highly recommended.
* **Environment Manager:** Anaconda or Miniconda.

## 2. Installation

First, clone the repository and navigate into it:

```bash
git clone https://github.com/lanl/annomate-microsentryai-workflow.git
cd annomate-microsentryai-workflow
```

Create the Conda environment for your hardware. We have provided specific environment files based on your system's capabilities — pick **one** of the following. Each file already declares its own environment name, so `conda env create` sets everything up in a single step; you don't need to create a base environment first.

**For Windows/Linux with NVIDIA GPUs (CUDA):**
```bash
conda env create --file environment-cuda.yml
conda activate annomate-cuda
```

**For macOS (Apple Silicon):**
```bash
conda env create --file environment-mac.yml
conda activate annomate-mac
```

**For CPU Only (No hardware acceleration):**
```bash
conda env create --file environment-cpu.yml
conda activate annomate-cpu
```

*Note for developers: If you plan on contributing, please run pre-commit install to enable automated linting.*


## Running the Application

Ensure your conda environment is activated, then run the main Python script from the root directory:

```bash
python src/main.py
```

The first time the app is launched, a guided welcome tour walks through the interface automatically. You can replay it anytime from **Help > Show Welcome Tour**.

## Building a Standalone Executable (PyInstaller)

If you want to distribute the application to users who do not have Python or Conda installed, you can compile AnnoMate & MicroSentryAI into a standalone executable using PyInstaller. *(Note: PyInstaller is already included in each of the environment-*.yml dependency files).*

From the root directory of the project, run the following command.

**For Windows:**
```bash 
pyinstaller --name "AnnoMate" --windowed --add-data "logos;logos" src/main.py
```

**For MacOS/Linux:**
```bash 
pyinstaller --name "AnnoMate" --windowed --add-data "logos:logos" src/main.py
```

**Important Build Notes:**
* **File Size**: Because the application bundles PySide6, PyTorch, and Anomalib, the resulting build folder (dist/AnnoMate/) will be quite large (often several gigabytes).

* **Console Flag**: The --windowed (or --noconsole) flag hides the terminal window in the final build. If your compiled app crashes on startup, try removing the --windowed flag and rebuilding; this will allow you to see the terminal output and identify any missing hidden imports required by PyTorch or Anomalib.

* **SAM 2 Weights**: The SAM 2 model downloads its checkpoint weights to a local sam_weights/ folder upon first use. You may need to manually copy this folder into the final dist/AnnoMate/ directory if you want it pre-packaged for offline users.