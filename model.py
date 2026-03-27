"""
Module for the RPG-AE model, which is a graph-based autoencoder for molecular generation. 
This module defines the architecture of the encoder and decoder, as well as the training loop and loss functions.   
RPG-AE: Neuro-Symbolic Graph Autoencoders with Rare Pattern Mining for Provenance-Based
Anomaly Detection.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import scipy.sparse as sp
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings("ignore")


#   FEATURE EXTRACTION

class FeatureExtractor:
    """The paper's Algorithm 1 simply states:
    'Convert provenance logs into process-level feature matrix X ∈ ℝⁿˣᵈ'
    It describes the concept of feature extraction but doesn't specify which features to use, 
    because that depends entirely on the dataset the researcher is working with. The paper leaves feature engineering as a domain-specific concern.
    So when implementing FeatureExtractor, some defaults are needed to make the class immediately runnable. I chose these six:"""

    DEFAULT_NUMERIC_COLS = [
        "file_reads", "file_writes", "net_connections",
        "child_processes", "registry_accesses", "duration_s",
    ]

    def __init__(self, numeric_cols: Optional[List[str]] = None):
        self.numeric_cols = numeric_cols or self.DEFAULT_NUMERIC_COLS

    def from_dataframe(self, df: pd.DataFrame, process_col: str = "process_id") -> Tuple[np.ndarray, List]:
        """Converts a DataFrame of provenance logs into a feature matrix X and a list of process IDs.
        
        Args:
            df: DataFrame containing provenance logs, with one row per process.
            process_col: Name of the column containing unique process identifiers.
        
        Returns:
            Tuple of the feature matrix X and the list of process IDs.
        """
        agg = {c: "sum" for c in self.numeric_cols if c in df.columns}
        grouped = df.groupby(process_col).agg(agg).reset_index()
        process_ids = grouped[process_col].tolist()
        X = grouped[list(agg.keys())].values.astype(np.float32)
        return X, process_ids

    @staticmethod
    def from_matrix(X: np.ndarray,
                    process_ids: Optional[List] = None) -> Tuple[np.ndarray, List]:
        """Pass through a pre-built feature matrix."""
        ids = process_ids if process_ids is not None else list(range(len(X)))
        return X.astype(np.float32), ids



#   k-NN SIMILARITY GRAPH

class KNNGraphBuilder:
    """
    Builds a process-similarity graph G^knn = (V, E^knn).

    Steps
    -----
    * Standardise X with zero mean / unit variance.
    * Compute pairwise cosine distances (= 1 - cosine_similarity).
    * Connect each process to its k nearest neighbours (mutual or one-way).
    """

    def __init__(self, k: int = 10, mutual: bool = False):
        self.k = k
        self.mutual = mutual
        self.scaler = StandardScaler()

    def build(self, X: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Returns
        -------
        edge_index : LongTensor [2, E]   – COO edge list
        X_scaled   : float32 ndarray     – standardised feature matrix
        """
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)

        # Cosine distance via NearestNeighbors on L2-normalised vectors
        norms = np.linalg.norm(X_scaled, axis=1, keepdims=True) + 1e-8
        X_norm = X_scaled / norms

        nbrs = NearestNeighbors(n_neighbors=self.k + 1,
                                metric="euclidean",  # equiv. to cosine on normed
                                algorithm="ball_tree").fit(X_norm)
        distances, indices = nbrs.kneighbors(X_norm)

        # Build edge list (exclude self-loop at index 0)
        rows, cols = [], []
        for i, nbr_row in enumerate(indices):
            for j in nbr_row[1:]:           # skip self
                rows.append(i)
                cols.append(j)
                if not self.mutual:         # directed -> also add reverse
                    rows.append(j)
                    cols.append(i)

        if self.mutual:
            # Keep only edges that appear in both directions
            edge_set = set(zip(rows, cols))
            rows_m, cols_m = [], []
            for r, c in edge_set:
                if (c, r) in edge_set:
                    rows_m.append(r)
                    cols_m.append(c)
            rows, cols = rows_m, cols_m

        # Deduplicate
        edges = list(set(zip(rows, cols)))
        if edges:
            rows, cols = zip(*edges)
        else:
            rows, cols = [], []

        edge_index = torch.tensor([list(rows), list(cols)], dtype=torch.long)
        return edge_index, X_scaled



#   RARE PATTERN MINING (Apriori)


class RarePatternMiner:
    """
    Discovers rare behavioural co-occurrence patterns using an Apriori-style
    procedure.

    A pattern is *rare* if its support falls in [min_support, max_support).
    The feature matrix X is binarised: X_bin[i,f] = 1 if X[i,f] > 0.

    Parameters
    ----------
    min_support : float   – lower support threshold (patterns must appear at
                            least this fraction of the time)
    max_support : float   – upper support threshold (patterns must be *rare*,
                            i.e., appear at most this fraction)
    """

    def __init__(self, min_support: float = 0.01, max_support: float = 0.10):
        self.min_support = min_support
        self.max_support = max_support
        self.rare_patterns_: List[frozenset] = []

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """Mine rare patterns from the binarised feature matrix."""
        n, d = X.shape
        names = feature_names if feature_names else [str(i) for i in range(d)]

        # Binarise: feature is present if its value is above the median
        medians = np.median(X, axis=0)
        X_bin = (X > medians).astype(bool)

        # Build transaction list
        transactions = []
        for i in range(n):
            t = [names[f] for f in range(d) if X_bin[i, f]]
            transactions.append(t)

        if not any(transactions):
            self.rare_patterns_ = []
            return self

        te = TransactionEncoder()
        te_array = te.fit_transform(transactions)
        df_bin = pd.DataFrame(te_array, columns=te.columns_)

        # Run Apriori with the low threshold to capture rare itemsets
        freq = apriori(df_bin,
                       min_support=self.min_support,
                       use_colnames=True,
                       max_len=None)

        # Filter to keep only patterns below max_support (rare)
        rare = freq[freq["support"] < self.max_support]
        self.rare_patterns_ = [frozenset(p) for p in rare["itemsets"]]
        return self

    def transform(self, X: np.ndarray,
                  feature_names: Optional[List[str]] = None) -> List[List[frozenset]]:
        """Return the list of rare patterns each process participates in."""
        n, d = X.shape
        names = feature_names if feature_names else [str(i) for i in range(d)]
        medians = np.median(X, axis=0)
        X_bin = (X > medians).astype(bool)

        process_patterns = []
        for i in range(n):
            present = frozenset(names[f] for f in range(d) if X_bin[i, f])
            matched = [p for p in self.rare_patterns_ if p.issubset(present)]
            process_patterns.append(matched)
        return process_patterns



#   RARE-PATTERN GRAPH


class PatternGraphBuilder:
    """
    Builds G^pat = (V, E^pat).

    Two processes are connected if they share at least one rare pattern.
    The degree d(i) in this graph becomes the raw boosting signal b(i).
    """

    def build(self, process_patterns: List[List[frozenset]],
              n: int) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Returns
        -------
        edge_index_pat : LongTensor [2, E]
        degrees        : float32 ndarray [n]  – normalised degrees b(i)
        """
        # Build an inverted index: pattern -> list of process indices
        pattern_to_procs: Dict[frozenset, List[int]] = {}
        for i, patterns in enumerate(process_patterns):
            for p in patterns:
                pattern_to_procs.setdefault(p, []).append(i)

        # Connect processes that share a pattern
        edge_set = set()
        for procs in pattern_to_procs.values():
            for a in procs:
                for b in procs:
                    if a != b:
                        edge_set.add((a, b))

        if edge_set:
            rows, cols = zip(*edge_set)
        else:
            rows, cols = [], []

        edge_index_pat = torch.tensor([list(rows), list(cols)], dtype=torch.long)

        # Degree vector
        degrees = np.zeros(n, dtype=np.float32)
        for r in rows:
            degrees[r] += 1
        d_max = degrees.max() if degrees.max() > 0 else 1.0
        degrees /= d_max          # b(i) = d(i) / d_max  ∈ [0, 1]

        return edge_index_pat, degrees



#   GRAPH AUTOENCODER (GCN Encoder + Inner-Product Decoder)


class GCNEncoder(nn.Module):
    """Two-layer GCN encoder: X, A  ->  Z ∈ R^nemb"""
 
    def __init__(self, in_dim: int, hidden_dim: int = 64, emb_dim: int = 32):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, emb_dim)
        self.dropout = nn.Dropout(0.3)
 
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.dropout(F.relu(self.conv1(x, edge_index))), edge_index)


class InnerProductDecoder(nn.Module):
    """Reconstructs adjacency: A_hat = alpha(Z ZTranspose)"""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z @ z.t())


class GraphAutoencoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, emb_dim: int = 32):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, emb_dim)
        self.decoder = InnerProductDecoder()

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_hat = self.decode(z)
        return adj_hat, z



#   GAE TRAINER


class GAETrainer:
    """Trains the Graph Autoencoder on G^knn."""

    def __init__(self,
                 hidden_dim: int = 64,
                 emb_dim: int = 32,
                 lr: float = 1e-3,
                 epochs: int = 200,
                 device: str = "cpu"):
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device)
        self.model: Optional[GraphAutoencoder] = None

    def _build_adj_target(self, edge_index: torch.Tensor, n: int) -> torch.Tensor:
        """Dense adjacency matrix as reconstruction target."""
        A = torch.zeros(n, n)
        if edge_index.numel() > 0:
            A[edge_index[0], edge_index[1]] = 1.0
        return A.to(self.device)

    def fit(self, X: np.ndarray, edge_index: torch.Tensor,
            verbose: bool = True) -> "GAETrainer":
        n, d = X.shape
        x = torch.tensor(X, dtype=torch.float32).to(self.device)
        ei = edge_index.to(self.device)
        A_target = self._build_adj_target(edge_index, n)

        self.model = GraphAutoencoder(d, self.hidden_dim, self.emb_dim).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Positive-edge weight to handle sparse graphs
        pos_weight = (n * n - A_target.sum()) / (A_target.sum() + 1e-8)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            opt.zero_grad()
            z = self.model.encode(x, ei)
            # Raw logits for BCEWithLogits
            logits = z @ z.t()
            loss = criterion(logits, A_target)
            loss.backward()
            opt.step()

            if verbose and epoch % 50 == 0:
                print(f"  [GAE] Epoch {epoch:>4}/{self.epochs}  loss={loss.item():.4f}")

        return self

    @torch.no_grad()
    def get_embeddings_and_reconstruction( self, X: np.ndarray, edge_index: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        x = torch.tensor(X, dtype=torch.float32).to(self.device)
        ei = edge_index.to(self.device)
        adj_hat, z = self.model(x, ei)
        return z.cpu().numpy(), adj_hat.cpu().numpy()



# ANOMALY SCORING


class AnomalyScorer:
    """
    Computes per-process anomaly scores.

    Baseline score  s_base(i):
        Row-wise reconstruction error between the true adjacency row A[i,:]
        and the predicted row Â[i,:].  We use binary cross-entropy so that
        both missing and spurious edges are penalised.

    Boosted score   s_boosted(i) = s_base(i) + alpha * b(i)
        where b(i) = d_pat(i) / d_max  is the normalised degree in G^pat.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def baseline_score(self, A_true: np.ndarray,
                       A_hat: np.ndarray) -> np.ndarray:
        """Row-wise BCE reconstruction error."""
        eps = 1e-8
        A_hat = np.clip(A_hat, eps, 1 - eps)
        bce_rows = -(A_true * np.log(A_hat) +
                     (1 - A_true) * np.log(1 - A_hat))
        return bce_rows.mean(axis=1).astype(np.float32)

    def boosted_score(self, s_base: np.ndarray,
                      b: np.ndarray) -> np.ndarray:
        return s_base + self.alpha * b



#   FULL PIPELINE (Algo 1)


class RPGAE:
    """
    RPG-AE: Rare-Pattern Graph Autoencoder.

    Parameters
    ----------
    k              : int   - number of neighbours in G^knn
    min_support    : float - lower support for Apriori
    max_support    : float - upper support for Apriori (rare = below this)
    alpha          : float - boosting weight
    hidden_dim     : int   - GCN hidden dimension
    emb_dim        : int   - GCN embedding dimension
    lr             : float - Adam learning rate
    epochs         : int   - training epochs
    device         : str   - 'cpu' or 'cuda'
    feature_names  : list  - optional list of feature column names
    """

    def __init__(self,
                 k: int = 10,
                 min_support: float = 0.01,
                 max_support: float = 0.10,
                 alpha: float = 0.5,
                 hidden_dim: int = 64,
                 emb_dim: int = 32,
                 lr: float = 1e-3,
                 epochs: int = 200,
                 device: str = "cpu",
                 feature_names: Optional[List[str]] = None):

        self.k = k
        self.min_support = min_support
        self.max_support = max_support
        self.alpha = alpha
        self.feature_names = feature_names

        self.knn_builder = KNNGraphBuilder(k=k)
        self.pattern_miner = RarePatternMiner(min_support, max_support)
        self.pattern_graph = PatternGraphBuilder()
        self.gae_trainer = GAETrainer(hidden_dim=hidden_dim,
                                      emb_dim=emb_dim,
                                      lr=lr,
                                      epochs=epochs,
                                      device=device)
        self.scorer = AnomalyScorer(alpha=alpha)

        # Artifacts stored after fit()
        self.edge_index_knn_: Optional[torch.Tensor] = None
        self.edge_index_pat_: Optional[torch.Tensor] = None
        self.X_scaled_: Optional[np.ndarray] = None
        self.embeddings_: Optional[np.ndarray] = None
        self.adj_hat_: Optional[np.ndarray] = None
        self.degrees_pat_: Optional[np.ndarray] = None
        self.rare_patterns_: Optional[List[frozenset]] = None
        self.s_base_: Optional[np.ndarray] = None
        self.s_boosted_: Optional[np.ndarray] = None

    # -- helpers ---------------------------

    def _adj_from_edge_index(self, edge_index: torch.Tensor, n: int) -> np.ndarray:
        A = np.zeros((n, n), dtype=np.float32)
        if edge_index.numel() > 0:
            ei = edge_index.numpy()
            A[ei[0], ei[1]] = 1.0
        return A

    # -- main API ---------------------------

    def fit(self, X: np.ndarray, verbose: bool = True) -> "RPGAE":
        """
        Run the full Algorithm-1 pipeline on feature matrix X.

        Parameters
        ----------
        X       : ndarray [n, d]  – process-level feature matrix
        verbose : bool            – print training progress
        """
        n, d = X.shape
        print(f"RPG-AE  |  n={n} processes,  d={d} features")

        # ---- Step 1: k-NN graph ---------------------------
        print("\n[1/5] Building k-NN similarity graph …")
        self.edge_index_knn_, self.X_scaled_ = self.knn_builder.build(X)
        print(f"      G^knn  |  edges: {self.edge_index_knn_.shape[1]}")

        # ---- Step 2: Rare pattern mining ------------------------------
        print("[2/5] Mining rare patterns (Apriori) …")
        self.pattern_miner.fit(X, self.feature_names)
        self.rare_patterns_ = self.pattern_miner.rare_patterns_
        print(f"      Found {len(self.rare_patterns_)} rare patterns")

        # ---- Step 3: Pattern graph ----------------------------------
        print("[3/5] Building pattern graph G^pat …")
        process_patterns = self.pattern_miner.transform(X, self.feature_names)
        self.edge_index_pat_, self.degrees_pat_ = self.pattern_graph.build(
            process_patterns, n)
        print(f"      G^pat  |  edges: {self.edge_index_pat_.shape[1]}")

        # ---- Step 4: Train GAE on G^knn -------------------------------
        print("[4/5] Training Graph Autoencoder …")
        self.gae_trainer.fit(self.X_scaled_, self.edge_index_knn_, verbose=verbose)

        # ---- Step 5: Score ---------------------
        print("[5/5] Computing anomaly scores …")
        self.embeddings_, self.adj_hat_ = \
            self.gae_trainer.get_embeddings_and_reconstruction(
                self.X_scaled_, self.edge_index_knn_)

        A_true = self._adj_from_edge_index(self.edge_index_knn_, n)
        self.s_base_ = self.scorer.baseline_score(A_true, self.adj_hat_)
        self.s_boosted_ = self.scorer.boosted_score(self.s_base_,
                                                     self.degrees_pat_)
        print("Done ✓")
        return self

    def anomaly_ranking(self, process_ids: Optional[List] = None,
                        top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Returning a ranked DataFrame of processes by boosted anomaly score.

        Parameters
        ----------
        process_ids : list  - optional labels for each process
        top_k       : int   - if set, return only the top-k anomalies
        """
        if self.s_boosted_ is None:
            raise RuntimeError("Call fit() first.")

        n = len(self.s_boosted_)
        ids = process_ids if process_ids is not None else list(range(n))

        df = pd.DataFrame({
            "process_id":    ids,
            "s_base":        self.s_base_,
            "boost":         self.alpha * self.degrees_pat_,
            "s_boosted":     self.s_boosted_,
            "pattern_degree": (self.degrees_pat_ *
                                (self.degrees_pat_.max() or 1)).round().astype(int),
        }).sort_values("s_boosted", ascending=False).reset_index(drop=True)

        if top_k is not None:
            df = df.head(top_k)
        return df

    def score(self, X_new: np.ndarray) -> np.ndarray:
        """
        Score unseen processes at inference time (no re-training).
        Uses only the baseline reconstruction score (no pattern boosting
        for out-of-sample nodes since G^pat is training-time only).
        """
        if self.gae_trainer.model is None:
            raise RuntimeError("Call fit() first.")
        X_sc = self.knn_builder.scaler.transform(X_new).astype(np.float32)
        # Build a temporary joint graph (train + test) and re-encode
        n_train = self.X_scaled_.shape[0]
        X_joint = np.vstack([self.X_scaled_, X_sc])
        ei_joint, _ = KNNGraphBuilder(k=self.k).build(X_joint)
        _, adj_hat_joint = self.gae_trainer.get_embeddings_and_reconstruction(
            X_joint, ei_joint)
        n_joint = len(X_joint)
        A_knn_joint = np.zeros((n_joint, n_joint), dtype=np.float32)
        if ei_joint.numel() > 0:
            e = ei_joint.numpy()
            A_knn_joint[e[0], e[1]] = 1.0
        scores_joint = self.scorer.baseline_score(A_knn_joint, adj_hat_joint)
        return scores_joint[n_train:]   # return only new-process scores