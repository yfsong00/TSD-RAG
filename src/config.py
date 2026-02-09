from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class TSDRAGConfig:
    # -----------------------------------------------------------------------
    # Basic Configuration
    # -----------------------------------------------------------------------
    dataset_name: str
    embedding_model: str = "all-mpnet-base-v2"
    llm_model: Optional[Any] = None  # Instance of LLM_Model class
    working_dir: str = "./data"
    
    # -----------------------------------------------------------------------
    # Data Processing
    # -----------------------------------------------------------------------
    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100
    spacy_model: str = "en_core_web_trf"
    batch_size: int = 128
    max_workers: int = 16
    
    # -----------------------------------------------------------------------
    # Retrieval Settings
    # -----------------------------------------------------------------------
    retrieval_top_k: int = 5
    use_vectorized_retrieval: bool = True
    
    # -----------------------------------------------------------------------
    # Graph Propagation & Algorithm Parameters
    # -----------------------------------------------------------------------
    # Number of iterations for activation spreading
    max_iterations: int = 3
    
    # Threshold to prune weak activation signals
    iteration_threshold: float = 0.4
    
    # Damping factor for Personalized PageRank
    damping: float = 0.5
    
    # Number of top sentences to select during propagation
    top_k_sentence: int = 1
    
    # Weight of the cosine similarity in the edge weighting formula
    sim_weight: float = 0.6
    
    # Smoothing coefficient for dynamic embedding updates (0.0 to 1.0)
    alpha_smooth: float = 0.7
    
    # -----------------------------------------------------------------------
    # Scoring & Ranking
    # -----------------------------------------------------------------------
    # Ratio to blend semantic score with graph structural score
    passage_ratio: float = 0.05
    
    # Global weight applied to passage nodes
    passage_node_weight: float = 0.05
    
    # Coefficient for entropy penalty in passage scoring
    lambda_entropy: float = 0.1
    
    # -----------------------------------------------------------------------
    # Seed Entity Retrieval (Viterbi / HMM)
    # -----------------------------------------------------------------------
    # Number of candidate entities to consider per query span
    retrieval_candidate_top_k: int = 5
    
    # Base for the distance penalty in Viterbi path finding
    penalty_base: float = 2.0
    
    # Scaling factor for the distance penalty
    lambda_dist: float = 0.1