""" Module for PLL algorithms """

from partial_label_learning.methods.cavl_2021 import Cavl
from partial_label_learning.methods.cel_2025 import Cel
from partial_label_learning.methods.crosel_2024 import CroSel
from partial_label_learning.methods.pico_2022 import PiCO
from partial_label_learning.methods.pl_ecoc_2017 import PlEcoc
from partial_label_learning.methods.pl_knn_2005 import PlKnn
from partial_label_learning.methods.pop_2023 import Pop
from partial_label_learning.methods.proden_2020 import Proden
from partial_label_learning.methods.valen_2021 import Valen
from partial_label_learning.methods.vi_ablation import ViAblation
from partial_label_learning.methods.vi_pll import ViPll

__all__ = [
    "Cavl", "Cel", "CroSel", "PiCO", "PlEcoc", "PlKnn",
    "Pop", "Proden", "Valen", "ViPll", "ViAblation",
]
