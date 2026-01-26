# Getting Started

This guide covers the installation and setup required to run the AnnoMate & MicroSentryAI suite.

## Prerequisites
* **OS:** macOS, Linux, or Windows.
* **Python:** Version 3.10+ is recommended.
* **Conda:** Anaconda or Miniconda is highly recommended for managing ML dependencies.

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/cjgeo22/AnnoMate-and-MicroSentryAI.git
    cd AnnoMate-and-MicroSentryAI
    ```

2.  **Create a Virtual Environment**
    It is best to isolate these heavy dependencies.
    ```bash
    conda create -n annomate python=3.10
    conda activate annomate
    ```

3.  **Install Dependencies**
    Install the required libraries. Note that `anomalib` handles the heavy lifting for the AI backend.
    ```bash
    pip install -r requirements.txt
    ```
    or
    ```bash
    conda env update --file environment.yml --prune
    ```
    *Note: If you are on Apple Silicon (M1/M2/M3), ensure you have the MPS-enabled version of PyTorch installed.*

## Running the Application

To launch the main dashboard, run the `main.py` entry point from the source directory:

```bash
python src/main.py
```

## Building a Standalone Executable (Nuitka)

Alternatively, you can compile the application into a standalone `.exe` (Windows) or binary (macOS/Linux) using Nuitka. This bundles Python and all libraries into a single folder.

### Why Nuitka?
We use Nuitka because it compiles Python code into C++ for better performance and handles complex dependencies (like PyTorch and Qt plugins) more reliably than PyInstaller.

### Build Command
Run the following command from the project root.

**For Windows:**
```bash
python -m nuitka --standalone --enable-plugin=pyqt5 --enable-plugin=numpy --include-data-dir=src/logos=logos --include-package=anomalib --include-package=lightning --output-dir=build src/main.py
```
**For Mac/Linux:**
```bash
python -m nuitka --standalone --macos-create-app-bundle --enable-plugin=pyqt5 --include-data-dir=src/logos=logos --include-package=anomalib --include-package=lightning --output-dir=build src/main.py
```