import argparse
import json
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from src.config import TSDRAGConfig
from src.tsdrag import TSDRAG  # Renamed from LinearRAG
import os
import warnings
from src.evaluate import Evaluator
from src.utils import LLM_Model
from src.utils import setup_logging
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
warnings.filterwarnings('ignore')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf", help="The spacy model to use")
    parser.add_argument("--embedding_model", type=str, default="all-mpnet-base-v2", help="The path of embedding model to use")
    parser.add_argument("--dataset_name", type=str, default="/root/TriDRAG/dataset/2wikimultihop", help="The dataset to use")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="The LLM model to use")
    parser.add_argument("--max_workers", type=int, default=16, help="The max number of workers to use")
    parser.add_argument("--max_iterations", type=int, default=3, help="The max number of iterations to use")
    parser.add_argument("--iteration_threshold", type=float, default=0.4, help="The threshold for iteration")
    parser.add_argument("--passage_ratio", type=float, default=0.05, help="The ratio for passage")
    parser.add_argument("--top_k_sentence", type=int, default=1, help="The top k sentence to use")
    parser.add_argument("--use_vectorized_retrieval", action="store_true", help="Use vectorized matrix-based retrieval instead of BFS iteration")
    
    # New arguments
    parser.add_argument("--lambda_entropy", type=float, default=0.2, help="Coefficient for entropy penalty")
    parser.add_argument("--alpha_smooth", type=float, default=0.1, help="Smoothing coefficient for embedding updates")
    parser.add_argument("--sim_weight", type=float, default=0.1, help="Weight for similarity")
    parser.add_argument("--retrieval_candidate_top_k", type=int, default=5, help="Top K candidates for seed entity retrieval")
    parser.add_argument("--penalty_base", type=float, default=3.0, help="Base for penalty in Viterbi")
    parser.add_argument("--lambda_dist", type=float, default=0.1, help="Distance lambda for Viterbi")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Directory for datasets")

    return parser.parse_args()


def load_dataset(data_dir, dataset_name): 
    questions_path = os.path.join(data_dir, dataset_name, "questions.json")
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    chunks_path = os.path.join(data_dir, dataset_name, "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    passages = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]
    return questions, passages

def load_embedding_model(embedding_model):
    embedding_model = SentenceTransformer(embedding_model, device="cuda")
    return embedding_model

def main():
    time = datetime.now()
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_arguments()
    embedding_model = load_embedding_model(args.embedding_model)
    questions, passages = load_dataset(args.data_dir, args.dataset_name)
    setup_logging(f"results/{args.dataset_name}/{time_str}/log.txt")
    llm_model = LLM_Model(args.llm_model)
    config = TSDRAGConfig(
        dataset_name=args.dataset_name,
        embedding_model=embedding_model,
        spacy_model=args.spacy_model,
        max_workers=args.max_workers,
        llm_model=llm_model,
        max_iterations=args.max_iterations,
        iteration_threshold=args.iteration_threshold,
        passage_ratio=args.passage_ratio,
        top_k_sentence=args.top_k_sentence,
        use_vectorized_retrieval=args.use_vectorized_retrieval,
        lambda_entropy=args.lambda_entropy,
        alpha_smooth=args.alpha_smooth,
        sim_weight=args.sim_weight,
        retrieval_candidate_top_k=args.retrieval_candidate_top_k,
        penalty_base=args.penalty_base,
        lambda_dist=args.lambda_dist
    )
    rag_model = TSDRAG(global_config=config)
    rag_model.index(passages)
    questions = rag_model.qa(questions)
    os.makedirs(f"results/{args.dataset_name}/{time_str}", exist_ok=True)
    with open(f"results/{args.dataset_name}/{time_str}/predictions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)
    evaluator = Evaluator(llm_model=llm_model, predictions_path=f"results/{args.dataset_name}/{time_str}/predictions.json")
    evaluator.evaluate(max_workers=args.max_workers)
if __name__ == "__main__":
    main()