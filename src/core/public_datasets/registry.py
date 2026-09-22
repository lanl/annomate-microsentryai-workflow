"""
Registry of available public-dataset import adapters.

Adding a new format is: write an adapter under adapters/ implementing
PublicDatasetAdapter, then add one line here. Nothing else in the import
pipeline (build.py, the import dialog, menu wiring) needs to change.
"""

from core.public_datasets.adapters.kolektor_sdd import KolektorSDDAdapter
from core.public_datasets.adapters.mvtec_ad import MVTecADAdapter
from core.public_datasets.adapters.mvtec_ad2 import MVTecAD2Adapter
from core.public_datasets.adapters.visa import VisAAdapter

PUBLIC_DATASET_ADAPTERS = {
    "MVTec AD": MVTecADAdapter(),
    "MVTec AD 2": MVTecAD2Adapter(),
    "KolektorSDD": KolektorSDDAdapter(),
    "VisA": VisAAdapter(),
}
