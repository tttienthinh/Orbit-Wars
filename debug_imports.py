"""Test which import hangs at 123MB."""
import sys
import os

def pr(msg):
    print(msg, flush=True)
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

pr("START")
import json; pr("json ok")
import math; pr("math ok")
import random; pr("random ok")
import shutil; pr("shutil ok")
import time; pr("time ok")
from pathlib import Path; pr("pathlib ok")

pr("importing numpy...")
import numpy as np; pr("numpy ok")

pr("importing polars...")
import polars as pl; pr("polars ok")

pr("importing torch...")
import torch; pr(f"torch ok: {torch.__version__}")

pr("importing torch.nn...")
import torch.nn as nn; pr("torch.nn ok")

pr("importing torch.nn.functional...")
import torch.nn.functional as F; pr("torch.nn.functional ok")

pr("importing sklearn...")
from sklearn.metrics import roc_auc_score; pr("sklearn ok")

pr("importing torch_geometric.data...")
from torch_geometric.data import HeteroData; pr("HeteroData ok")

pr("importing torch_geometric.nn...")
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv; pr("GATConv/HeteroConv/SAGEConv ok")

pr("importing torch_geometric.transforms...")
from torch_geometric.transforms import ToUndirected; pr("ToUndirected ok")

pr("ALL IMPORTS DONE")
