"""
Affect subspace analysis: projection operators and embedding decomposition.

Implements the geometric decomposition of text embeddings into "affect" and
"non-affect" components based on the linear probe weights W_affect.

Semantic Entanglement Caveat:
    Natural language often entangles topic and emotion (e.g., "funeral" is
    colinear with "sadness"). Therefore, h_non_affect may incidentally lose
    some semantic topic information. We interpret "non-affect" as "content
    orthogonal to emotion directions," which may differ from "pure topic."
"""

import numpy as np
from typing import Tuple, Optional, Dict

# =============================================================================
# Projection Matrix Construction
# =============================================================================

def compute_projection_matrix(
    W_affect: np.ndarray, 
    orthonormalize: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute orthogonal projection matrix onto the affect subspace.
    
    Given W_affect of shape (K, d), the affect subspace is spanned by the
    transpose B = W_affect^T (shape d x K).
    
    If orthonormalize=True:
        1. Compute QR decomposition: B = QR
        2. Use Q (orthonormal columns) as the basis
        3. Projection: P_affect = Q @ Q^T
    
    If orthonormalize=False:
        P_affect = B @ (B^T @ B)^{-1} @ B^T
    
    Args:
        W_affect: Probe weight matrix of shape (K, 384) where K is affect dim
        orthonormalize: Whether to use QR orthonormalization (default: True)
        
    Returns:
        Tuple of:
            - Projection matrix P_affect of shape (384, 384)
            - Orthonormalized basis Q of shape (384, K) if orthonormalize=True, else None
    """
    # B: basis matrix with affect directions as columns
    # Shape: (384, K) where K=15
    B = W_affect.T.astype(np.float64)  # Use float64 for numerical stability
    
    if orthonormalize:
        # QR decomposition for orthonormal basis
        Q, R = np.linalg.qr(B)
        
        # P = Q @ Q^T is the orthogonal projector
        P_affect = Q @ Q.T
        
        return P_affect.astype(np.float32), Q.astype(np.float32)
    else:
        # Original formulation: P = B @ (B^T B)^{-1} @ B^T
        G = B.T @ B  # Gram matrix
        
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            G_inv = np.linalg.pinv(G)
        
        P_affect = B @ G_inv @ B.T
        
        return P_affect.astype(np.float32), None


def verify_projection(P: np.ndarray, tol: float = 1e-5) -> dict:
    """
    Verify that P is a valid orthogonal projection matrix.
    
    An orthogonal projection satisfies:
        1. P^2 = P (idempotent)
        2. P^T = P (symmetric)
    
    Args:
        P: Matrix to verify
        tol: Numerical tolerance for equality checks
        
    Returns:
        Dict with verification results
    """
    # Check idempotence: P^2 = P
    P_squared = P @ P
    idempotent_error = np.max(np.abs(P_squared - P))
    
    # Check symmetry: P^T = P
    symmetry_error = np.max(np.abs(P.T - P))
    
    return {
        "is_idempotent": idempotent_error < tol,
        "is_symmetric": symmetry_error < tol,
        "idempotent_error": float(idempotent_error),
        "symmetry_error": float(symmetry_error),
        "rank": int(np.linalg.matrix_rank(P))
    }


# =============================================================================
# Affect Subspace Class
# =============================================================================

class AffectSubspace:
    """
    Manages the affect subspace and provides projection operations.
    
    Initialized with W_affect from a fitted AffectProbe, this class:
        - Computes the projection matrix P_affect (with optional QR orthonormalization)
        - Projects embeddings onto affect / non-affect components
        - Extracts low-dimensional affect coordinates
    """
    
    def __init__(
        self, 
        W_affect: np.ndarray, 
        orthonormalize: bool = True,
        valid_mask: Optional[np.ndarray] = None
    ):
        """
        Initialize the subspace from probe weights.
        
        Args:
            W_affect: Weight matrix from AffectProbe, shape (K, 384)
            orthonormalize: If True (default), use QR decomposition for stable projection
            valid_mask: Optional boolean mask from probe's linearity check
                        If provided, only valid dimensions are used in the subspace
        """
        # Optionally filter to valid dimensions only
        if valid_mask is not None:
            W_affect = W_affect[valid_mask]
        
        self.W_affect = W_affect.astype(np.float32)
        self.K = W_affect.shape[0]  # Affect dimension (possibly reduced)
        self.d = W_affect.shape[1]  # Embedding dimension (384)
        self.orthonormalized = orthonormalize
        
        # Compute projection matrix and (optionally) orthonormal basis
        self.P_affect, self.Q = compute_projection_matrix(W_affect, orthonormalize)
        
        # Basis matrix for coordinate extraction
        if orthonormalize and self.Q is not None:
            # Use orthonormalized basis Q
            self.B = self.Q  # Shape (384, K)
        else:
            # Use raw basis
            self.B = self.W_affect.T  # Shape (384, K)
        
        # Complementary projection (onto non-affect subspace)
        self.P_non_affect = np.eye(self.d, dtype=np.float32) - self.P_affect
    
    def affect_component(self, H: np.ndarray) -> np.ndarray:
        """
        Compute low-dimensional affect coordinates.
        
        Projects embeddings onto the K-dimensional coordinate system defined
        by the affect basis. This gives the representation used by M_affect_only.
        
        With orthonormalization:
            u = Q^T @ h (where Q has orthonormal columns)
        
        Without orthonormalization:
            u = B^T @ h (where B = W_affect^T)
        
        Args:
            H: Text embeddings of shape (N, 384)
            
        Returns:
            Affect coordinates U of shape (N, K)
        """
        # U = H @ B, where B has shape (384, K)
        # Result: (N, K)
        return (H @ self.B).astype(np.float32)
    
    def affect_component_full(self, H: np.ndarray) -> np.ndarray:
        """
        Compute the affect component in original embedding space.
        
        Projects each embedding onto the affect subspace, keeping the result
        in the full 384-dimensional space.
        
        Mathematically: h_affect_full = P_affect @ h
        
        Args:
            H: Text embeddings of shape (N, 384)
            
        Returns:
            Affect components of shape (N, 384) in original space
        """
        # H_affect = H @ P_affect^T, but P is symmetric so P^T = P
        return (H @ self.P_affect).astype(np.float32)
    
    def non_affect_component(self, H: np.ndarray) -> np.ndarray:
        """
        Compute the non-affect component.
        
        Removes the affect subspace projection, leaving content orthogonal
        to all affect directions (topic, style, factual content, etc.).
        
        Mathematically: h_non_affect = h - P_affect @ h = (I - P_affect) @ h
        
        Args:
            H: Text embeddings of shape (N, 384)
            
        Returns:
            Non-affect embeddings of shape (N, 384)
        """
        return (H @ self.P_non_affect).astype(np.float32)
    
    def decompose(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fully decompose embeddings into affect coordinates and non-affect residual.
        
        Args:
            H: Text embeddings of shape (N, 384)
            
        Returns:
            Tuple of:
                U: Affect coordinates (N, K)
                H_affect_full: Affect component in original space (N, 384)
                H_non_affect: Non-affect residual (N, 384)
        """
        U = self.affect_coords(H)
        H_affect_full = self.affect_component_full(H)
        H_non_affect = self.non_affect_component(H)
        return U, H_affect_full, H_non_affect
    
    def reconstruction_error(self, H: np.ndarray) -> float:
        """
        Compute reconstruction error: ||H - (H_affect + H_non_affect)||.
        
        Should be near zero since the decomposition is exact (up to numerical error).
        With QR orthonormalization, this should be essentially machine precision.
        
        Args:
            H: Text embeddings of shape (N, 384)
            
        Returns:
            Mean squared reconstruction error
        """
        H_affect_full = self.affect_component_full(H)
        H_non_affect = self.non_affect_residual(H)
        reconstructed = H_affect_full + H_non_affect
        return float(np.mean((H - reconstructed) ** 2))
    
    def orthogonality_check(self, H: np.ndarray) -> float:
        """
        Verify that affect and non-affect components are orthogonal.
        
        Computes mean |<h_affect, h_non_affect>| across samples.
        Should be near zero with QR orthonormalization.
        
        Args:
            H: Text embeddings of shape (N, 384)
            
        Returns:
            Mean absolute inner product
        """
        H_affect = self.affect_component_full(H)
        H_non_affect = self.non_affect_residual(H)
        
        # Compute inner products row-wise
        inner_products = np.sum(H_affect * H_non_affect, axis=1)
        return float(np.mean(np.abs(inner_products)))
    
    def explained_variance_ratio(self, H: np.ndarray) -> float:
        """
        Compute fraction of embedding variance explained by affect subspace.
        
        This measures how much of the total embedding variance lies in the
        affect directions. Higher values suggest emotion is a dominant feature.
        
        Args:
            H: Text embeddings of shape (N, 384)
            
        Returns:
            Ratio in [0, 1] of variance in affect subspace
        """
        total_var = np.var(H)
        affect_var = np.var(self.affect_component_full(H))
        return float(affect_var / total_var) if total_var > 0 else 0.0
    
    def get_subspace_info(self) -> Dict:
        """
        Return information about the subspace configuration.
        
        Returns:
            Dict with subspace properties
        """
        props = verify_projection(self.P_affect)
        return {
            "affect_dim": self.K,
            "embedding_dim": self.d,
            "orthonormalized": self.orthonormalized,
            "projection_rank": props["rank"],
            "is_valid_projection": props["is_idempotent"] and props["is_symmetric"],
            "idempotent_error": props["idempotent_error"],
            "symmetry_error": props["symmetry_error"],
        }


# =============================================================================
# Feature Preparation for Subspace Models
# =============================================================================

def prepare_affect_only_input(U: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Prepare input for M_affect_only model.
    
    Concatenates low-dimensional affect coordinates with metadata.
    
    Args:
        U: Affect coordinates of shape (N, 15)
        C: Metadata features of shape (N, 10)
        
    Returns:
        Input array of shape (N, 25)
    """
    return np.concatenate([U, C], axis=1).astype(np.float32)


def prepare_non_affect_input(H_non: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Prepare input for M_non_affect model.
    
    Concatenates non-affect embedding residual with metadata.
    
    Args:
        H_non: Non-affect embeddings of shape (N, 384)
        C: Metadata features of shape (N, 10)
        
    Returns:
        Input array of shape (N, 394)
    """
    return np.concatenate([H_non, C], axis=1).astype(np.float32)


# =============================================================================
# Subspace Analysis Utilities
# =============================================================================

def analyze_affect_directions(
    W_affect: np.ndarray,
    affect_names: list = None
) -> dict:
    """
    Analyze properties of individual affect direction vectors.
    
    Args:
        W_affect: Probe weight matrix (K, 384)
        affect_names: Optional list of affect dimension names
        
    Returns:
        Dict with analysis results per dimension
    """
    K = W_affect.shape[0]
    
    if affect_names is None:
        affect_names = [f"affect_{i}" for i in range(K)]
    
    # Compute norms and pairwise cosine similarities
    norms = np.linalg.norm(W_affect, axis=1)
    
    # Normalize for cosine similarity
    W_normed = W_affect / (norms[:, np.newaxis] + 1e-8)
    cosine_sim = W_normed @ W_normed.T  # (K, K) pairwise similarities
    
    # Off-diagonal elements (exclude self-similarity)
    mask = ~np.eye(K, dtype=bool)
    off_diag = cosine_sim[mask]
    
    return {
        "norms": {name: float(norm) for name, norm in zip(affect_names, norms)},
        "mean_pairwise_cosine": float(np.mean(np.abs(off_diag))),
        "max_pairwise_cosine": float(np.max(np.abs(off_diag))),
        "cosine_matrix": cosine_sim
    }