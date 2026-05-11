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
git clone https://github.com/annomate-mircrosentryai-workflow
cd annomate-microsentryai-workflow
```
Create and activate a clean Conda environment:

```bash
conda create -n annomate python=3.10
conda activate annomate
```

Install the required dependencies. We have provided specific environment files based on your system's hardware capabilities. Run **one** of the following commands:

**For Windows/Linux with NVIDIA GPUs (CUDA):**
```bash
conda env update --file environment-cuda.yml --prune
```

**For macOS (Apple Silicon):**
```bash
conda env update --file environment-mac.yml --prune
```

**For CPU Only (No hardware acceleration):**
```bash
conda env update --file environment-cpu.yml --prune
```

*Note for developers: If you plan on contributing, please run pre-commit install to enable automated linting.*


## Running the Application

Ensure your conda environment is activated, then run the main Python script from the root directory:

```bash
python src/main.py
```

## Building a Standalone Executable (PyInstaller)

If you want to distribute the application to users who do not have Python or Conda installed, you can compile AnnoMate & MicroSentryAI into a standalone executable using PyInstaller. *(Note: PyInstaller is already included in your environment.yml dependencies).*

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