"""
Registry of available public-dataset import adapters.

Adding a new format is: write an adapter under adapters/ implementing
PublicDatasetAdapter, then add one line here. Nothing else in the import
pipeline (build.py, the import dialog, menu wiring) needs to change.
"""

from core.public_datasets.adapters.mvtec_ad import MVTecADAdapter

PUBLIC_DATASET_ADAPTERS = {
    "MVTec AD": MVTecADAdapter(),
}
