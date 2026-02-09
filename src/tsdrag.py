from src.embedding_store import EmbeddingStore
from src.utils import min_max_normalize
import os
import json
from collections import defaultdict
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.ner import SpacyNER
import igraph as ig
import re
import logging
import torch
import spacy
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class TSDRAG:
    def __init__(self, global_config):
        self.config = global_config
        logger.info(f"Initializing TSDRAG with config: {self.config}")
        
        retrieval_method = "Vectorized Matrix-based" if self.config.use_vectorized_retrieval else "BFS Iteration"
        logger.info(f"Using retrieval method: {retrieval_method}")
        
        self.nlp = spacy.load(self.config.spacy_model)
        self.infnumber = 0
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.config.use_vectorized_retrieval:
            logger.info(f"Using device: {self.device} for vectorized retrieval")
        
        self.dataset_name = global_config.dataset_name
        self.load_embedding_store()
        self.llm_model = self.config.llm_model
        self.spacy_ner = SpacyNER(self.config.spacy_model)
        self.graph = ig.Graph(directed=False)

    def load_embedding_store(self):
        self.passage_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir, self.dataset_name, "passage_embedding.parquet"), batch_size=self.config.batch_size, namespace="passage")
        self.entity_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir, self.dataset_name, "entity_embedding.parquet"), batch_size=self.config.batch_size, namespace="entity")
        self.sentence_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir, self.dataset_name, "sentence_embedding.parquet"), batch_size=self.config.batch_size, namespace="sentence")

    def load_existing_data(self, passage_hash_ids):
        self.ner_results_path = os.path.join(self.config.working_dir, self.dataset_name, "ner_results.json")
        if os.path.exists(self.ner_results_path):
            existing_ner_reuslts = json.load(open(self.ner_results_path))
            existing_passage_hash_id_to_entities = existing_ner_reuslts["passage_hash_id_to_entities"]
            existing_sentence_to_entities = existing_ner_reuslts["sentence_to_entities"]
            existing_passage_hash_ids = set(existing_passage_hash_id_to_entities.keys())
            new_passage_hash_ids = set(passage_hash_ids) - existing_passage_hash_ids
            return existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids
        else:
            return {}, {}, passage_hash_ids

    def qa(self, questions):
        retrieval_results = self.retrieve(questions)
        system_prompt = f"""As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations. Please do not give additional prefixes such as repeat the question beyond the clear answer, and try to be consistent with the language that the answer may need. Pay attention to the case of letters when answering. If the text paragraph does not provide enough information, try to answer using your own knowledge."""
        all_messages = []
        for retrieval_result in retrieval_results:
            question = retrieval_result["question"]
            sorted_passage = retrieval_result["sorted_passage"]
            prompt_user = """"""
            for passage in sorted_passage:
                prompt_user += f"{passage}\n"
            prompt_user += f"Question: {question}\n Thought: "
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_user}
            ]
            all_messages.append(messages)
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            all_qa_results = list(tqdm(
                executor.map(self.llm_model.infer, all_messages),
                total=len(all_messages),
                desc="QA Reading (Parallel)"
            ))

        for qa_result, question_info in zip(all_qa_results, retrieval_results):
            try:
                pred_ans = qa_result.split('Answer:')[1].strip()
            except:
                pred_ans = qa_result
            question_info["pred_answer"] = pred_ans
        return retrieval_results
        
    def retrieve(self, questions):
        self.entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self.entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        self.passage_hash_ids = list(self.passage_embedding_store.hash_id_to_text.keys())
        self.passage_embeddings = np.array(self.passage_embedding_store.embeddings)
        self.sentence_hash_ids = list(self.sentence_embedding_store.hash_id_to_text.keys())
        self.sentence_embeddings = np.array(self.sentence_embedding_store.embeddings)
        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}
        self.vertex_idx_to_node_name = {v.index: v["name"] for v in self.graph.vs if "name" in v.attributes()}

        if self.config.use_vectorized_retrieval:
            logger.info("Precomputing sparse adjacency matrices for vectorized retrieval...")
            self._precompute_sparse_matrices()
            e2s_shape = self.entity_to_sentence_sparse.shape
            s2e_shape = self.sentence_to_entity_sparse.shape
            e2s_nnz = self.entity_to_sentence_sparse._nnz()
            s2e_nnz = self.sentence_to_entity_sparse._nnz()
            logger.info(f"Matrices built: Entity-Sentence {e2s_shape}, Sentence-Entity {s2e_shape}")
            logger.info(f"E2S Sparsity: {(1 - e2s_nnz / (e2s_shape[0] * e2s_shape[1])) * 100:.2f}% (nnz={e2s_nnz})")
            logger.info(f"S2E Sparsity: {(1 - s2e_nnz / (s2e_shape[0] * s2e_shape[1])) * 100:.2f}% (nnz={s2e_nnz})")
            logger.info(f"Device: {self.device}")

        retrieval_results = []
        for question_info in tqdm(questions, desc="Retrieving"):
            question = question_info["question"]
            question_embedding = self.config.embedding_model.encode(question, normalize_embeddings=True, show_progress_bar=False, batch_size=self.config.batch_size)
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores = self.get_seed_entities(question, question_embedding)
            
            activation_history = []
            
            if len(seed_entities) != 0:
                sorted_passage_hash_ids, sorted_passage_scores, activation_history = self.graph_search_with_seed_entities(question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores)
                final_passage_hash_ids = sorted_passage_hash_ids[:self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                final_passages = [self.passage_embedding_store.hash_id_to_text[passage_hash_id] for passage_hash_id in final_passage_hash_ids]
            else:
                sorted_passage_indices, sorted_passage_scores = self.dense_passage_retrieval(question_embedding)
                final_passage_indices = sorted_passage_indices[:self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                final_passages = [self.passage_embedding_store.texts[idx] for idx in final_passage_indices]
            
            result = {
                "question": question,
                "sorted_passage": final_passages,
                "sorted_passage_scores": final_passage_scores,
                "gold_answer": question_info["answer"],
                "aligned_entities": seed_entities,
                "activated_nodes_history": activation_history
            }
            retrieval_results.append(result)
        return retrieval_results
    
    def _precompute_sparse_matrices(self):
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)
        
        entity_to_sentence_indices = []
        entity_to_sentence_values = []
        
        for entity_hash_id, sentence_hash_ids in self.entity_hash_id_to_sentence_hash_ids.items():
            entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
            for sentence_hash_id in sentence_hash_ids:
                sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
                entity_to_sentence_indices.append([entity_idx, sentence_idx])
                entity_to_sentence_values.append(1.0)
        
        sentence_to_entity_indices = []
        sentence_to_entity_values = []
        
        for sentence_hash_id, entity_hash_ids in self.sentence_hash_id_to_entity_hash_ids.items():
            sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
            for entity_hash_id in entity_hash_ids:
                entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
                sentence_to_entity_indices.append([sentence_idx, entity_idx])
                sentence_to_entity_values.append(1.0)
        
        if len(entity_to_sentence_indices) > 0:
            e2s_indices = torch.tensor(entity_to_sentence_indices, dtype=torch.long).t()
            e2s_values = torch.tensor(entity_to_sentence_values, dtype=torch.float32)
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                e2s_indices, e2s_values, (num_entities, num_sentences), device=self.device
            ).coalesce()
        else:
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_entities, num_sentences), device=self.device
            )
        
        if len(sentence_to_entity_indices) > 0:
            s2e_indices = torch.tensor(sentence_to_entity_indices, dtype=torch.long).t()
            s2e_values = torch.tensor(sentence_to_entity_values, dtype=torch.float32)
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                s2e_indices, s2e_values, (num_sentences, num_entities), device=self.device
            ).coalesce()
        else:
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_sentences, num_entities), device=self.device
            )
            
    def graph_search_with_seed_entities(self, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        if self.config.use_vectorized_retrieval:
            entity_weights, actived_entities, activation_history = self.calculate_entity_scores_vectorized(question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores)
        else:
            entity_weights, actived_entities, activation_history = self.calculate_entity_scores(question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores)
        passage_weights = self.calculate_passage_scores(question_embedding, actived_entities)
        node_weights = entity_weights + passage_weights
        ppr_sorted_passage_indices, ppr_sorted_passage_scores = self.run_ppr(node_weights)
        return ppr_sorted_passage_indices, ppr_sorted_passage_scores, activation_history

    def run_ppr(self, node_weights):        
        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=self.config.damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )
        
        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_indices])
        sorted_indices_in_doc_scores = np.argsort(doc_scores)[::-1]
        sorted_passage_scores = doc_scores[sorted_indices_in_doc_scores]
        
        sorted_passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[i]] 
            for i in sorted_indices_in_doc_scores
        ]
        
        return sorted_passage_hash_ids, sorted_passage_scores.tolist()

    def calculate_entity_scores_vectorized(self, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        # 1. Initialization
        Q_vec = torch.from_numpy(question_embedding).float().to(self.device).view(1, -1)
        Q_vec = F.normalize(Q_vec, p=2, dim=1)
        
        E_emb_static = torch.from_numpy(self.entity_embeddings).float().to(self.device)
        S_emb_static = torch.from_numpy(self.sentence_embeddings).float().to(self.device)
        
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)
        total_graph_nodes = len(self.graph.vs)
        
        E_emb_dynamic = E_emb_static.clone() 
        h_e_score = torch.zeros(num_entities, device=self.device)
        
        activation_history = []

        if seed_entity_indices and len(seed_entity_indices) > 0:
            indices_tensor = torch.tensor(seed_entity_indices, device=self.device, dtype=torch.long)
            scores_tensor = torch.tensor(seed_entity_scores, device=self.device, dtype=torch.float)
            h_e_score.index_add_(0, indices_tensor, scores_tensor)
            h_e_init = h_e_score.clone()
        else:
            return np.zeros(total_graph_nodes), {}, []

        h_s_score = torch.zeros(num_sentences, device=self.device)
        
        alpha_smooth = self.config.alpha_smooth
        beta_prune = self.config.iteration_threshold
        top_k = self.config.top_k_sentence
        
        e2s_indices = self.entity_to_sentence_sparse.indices()
        s2e_indices = self.sentence_to_entity_sparse.indices()

        actived_entities_dict = {}

        # 2. Evolution Loop
        for t in range(self.config.max_iterations):
            active_mask = h_e_score > 1e-6
            active_indices = torch.nonzero(active_mask).squeeze(1)
            
            if active_indices.numel() > 0:
                active_scores = h_e_score[active_indices]
                
                active_indices_np = active_indices.cpu().numpy()
                active_scores_np = active_scores.cpu().numpy()
                
                current_iter_nodes = []
                for idx, score in zip(active_indices_np, active_scores_np):
                    hash_id = self.entity_hash_ids[idx]
                    actived_entities_dict[hash_id] = (int(idx), float(score), t + 1)
                    
                    text = self.entity_embedding_store.hash_id_to_text.get(hash_id, str(hash_id))
                    current_iter_nodes.append({"text": text, "score": float(score)})
                
                current_iter_nodes.sort(key=lambda x: x["score"], reverse=True)
                
                if current_iter_nodes:
                    activation_history.append(current_iter_nodes[:10])
            
            if active_indices.numel() == 0: break
            
            # PHASE 1: Entity -> Sentence
            h_s_score.zero_() 
            
            mask_edges = torch.isin(e2s_indices[0], active_indices)
            
            if mask_edges.any():
                rel_e = e2s_indices[0, mask_edges]
                rel_s = e2s_indices[1, mask_edges]
                
                sims = F.cosine_similarity(E_emb_dynamic[rel_e], S_emb_static[rel_s])
                q_align = F.cosine_similarity(S_emb_static[rel_s], Q_vec.expand(rel_s.size(0), -1))
                
                sim_weight = self.config.sim_weight
                edge_weights = sim_weight * sims + (1 - sim_weight) * q_align
                
                valid_mask = edge_weights > beta_prune
                
                if valid_mask.any():
                    cand_e = rel_e[valid_mask]
                    cand_s = rel_s[valid_mask]
                    cand_w = edge_weights[valid_mask]
                    cand_sort_metric = edge_weights[valid_mask] 

                    # Vectorized Top-K Selection
                    if top_k > 0:
                        _, sort_idx_w = torch.sort(cand_sort_metric, descending=True)
                        cand_e_sorted = cand_e[sort_idx_w]
                        cand_s_sorted = cand_s[sort_idx_w]
                        cand_w_sorted = cand_w[sort_idx_w]
                        
                        cand_e_final, sort_idx_e = torch.sort(cand_e_sorted, stable=True)
                        cand_s_final = cand_s_sorted[sort_idx_e]
                        cand_w_final = cand_w_sorted[sort_idx_e]
                        
                        unique_e, counts = torch.unique_consecutive(cand_e_final, return_counts=True)
                        ends = torch.cumsum(counts, dim=0)
                        starts = torch.cat((torch.zeros(1, device=self.device, dtype=torch.long), ends[:-1]))
                        starts_expanded = starts.repeat_interleave(counts)
                        
                        ranks = torch.arange(len(cand_e_final), device=self.device) - starts_expanded
                        
                        topk_mask = ranks < top_k
                        
                        final_e = cand_e_final[topk_mask]
                        final_s = cand_s_final[topk_mask]
                        final_w = cand_w_final[topk_mask]
                    else:
                        final_e, final_s, final_w = cand_e, cand_s, cand_w

                    flow = h_e_score[final_e] * final_w
                    h_s_score.index_add_(0, final_s, flow)

            h_s_score = F.relu(h_s_score)
            curr_s = torch.nonzero(h_s_score > 1e-6).squeeze(1)
            if curr_s.numel() == 0: break

            # PHASE 2: Sentence -> Entity
            h_e_new_score = torch.zeros_like(h_e_score)
            emb_updates_sum = torch.zeros_like(E_emb_dynamic)
            emb_updates_count = torch.zeros(num_entities, 1, device=self.device)

            mask_back = torch.isin(s2e_indices[0], curr_s)
            if mask_back.any():
                back_s = s2e_indices[0, mask_back]
                back_e = s2e_indices[1, mask_back]
                
                h_e_new_score.index_add_(0, back_e, h_s_score[back_s])
                
                u_w = h_s_score[back_s].clamp(max=1.0).unsqueeze(1)
                emb_updates_sum.index_add_(0, back_e, S_emb_static[back_s] * u_w)
                emb_updates_count.index_add_(0, back_e, u_w)

            # Embedding Smoothing
            upd_m = (emb_updates_count.squeeze() > 0)
            if upd_m.any():
                ctx = emb_updates_sum[upd_m] / emb_updates_count[upd_m]
                E_emb_dynamic[upd_m] = F.normalize((1 - alpha_smooth) * E_emb_dynamic[upd_m] + alpha_smooth * ctx, p=2, dim=1)

            # Update & Fidelity
            h_e_score += h_e_new_score
            beta_prune += 0.05

        # 3. Format Output
        entity_weights_array = np.zeros(total_graph_nodes)
        
        final_active_mask = h_e_score > 0.001
        if final_active_mask.any():
            final_e_indices = torch.nonzero(final_active_mask).squeeze(1).cpu().numpy()
            final_e_scores = h_e_score[final_active_mask].cpu().numpy()
            
            for idx, score in zip(final_e_indices, final_e_scores):
                hash_id = self.entity_hash_ids[idx]
                f_score = float(score)
                
                if hash_id in self.node_name_to_vertex_idx:
                    v_idx = self.node_name_to_vertex_idx[hash_id]
                    entity_weights_array[v_idx] = f_score
                
                if hash_id not in actived_entities_dict:
                    actived_entities_dict[hash_id] = (int(idx), f_score, self.config.max_iterations)
            
        return entity_weights_array, actived_entities_dict, activation_history

    def calculate_entity_scores(self, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        return self.calculate_entity_scores_vectorized(question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores)

    def calculate_passage_scores(self, question_embedding, actived_entities):
        # 1. Semantic Relevance - Base Score
        dpr_indices, dpr_raw_scores = self.dense_passage_retrieval(question_embedding)
        
        if len(dpr_raw_scores) > 0:
            dpr_min, dpr_max = min(dpr_raw_scores), max(dpr_raw_scores)
            norm_dpr_scores = [(s - dpr_min) / (dpr_max - dpr_min + 1e-9) for s in dpr_raw_scores]
        else:
            norm_dpr_scores = []

        passage_sem_map = {}
        candidate_p_hashes = []
        
        for idx, score in zip(dpr_indices, norm_dpr_scores):
            p_hash = self.passage_embedding_store.hash_ids[idx]
            passage_sem_map[p_hash] = score
            candidate_p_hashes.append(p_hash)
            
        # 2. Structured Evidence Aggregation
        passage_weights = np.zeros(len(self.graph.vs), dtype=np.float32)
        lambda_entropy = self.config.lambda_entropy 
        
        active_e_map = {k: (v[1], v[2]) for k, v in actived_entities.items()}
        
        for p_hash in candidate_p_hashes:
            if p_hash not in self.node_name_to_vertex_idx:
                continue
                
            sem_score = passage_sem_map[p_hash]
            
            p_node_idx = self.node_name_to_vertex_idx[p_hash]
            
            neighbors = self.graph.neighbors(p_node_idx)
            
            neighbor_energies = []
            energy_sum = 0.0
            
            for n_idx in neighbors:
                n_name = self.vertex_idx_to_node_name[n_idx]
                
                if n_name in active_e_map:
                    e_score, tier = active_e_map[n_name]
                    denom = max(tier, 1.0)
                    
                    edge_weight = self.graph.es[self.graph.get_eid(p_node_idx, n_idx)]['weight']
                    val = e_score * edge_weight / denom
                    
                    energy_sum += val
                    neighbor_energies.append(val)
            
            entropy = 0.0
            if len(neighbor_energies) > 0:
                probs = np.array(neighbor_energies)
                if probs.sum() > 0:
                    probs = probs / probs.sum() 
                    entropy = -np.sum(probs * np.log(probs + 1e-8))
            
            final_score = (self.config.passage_ratio * sem_score) + \
                          math.log(1 + energy_sum) - \
                          (lambda_entropy * entropy)
            
            passage_weights[p_node_idx] = max(0.0, final_score) * self.config.passage_node_weight
            
        return passage_weights

    def dense_passage_retrieval(self, question_embedding):
        question_emb = question_embedding.reshape(1, -1)
        question_passage_similarities = np.dot(self.passage_embeddings, question_emb.T).flatten()
        sorted_passage_indices = np.argsort(question_passage_similarities)[::-1]
        sorted_passage_scores = question_passage_similarities[sorted_passage_indices].tolist()
        return sorted_passage_indices, sorted_passage_scores
    
    def get_seed_entities(self, question, question_emb, sharpening_factor=3.0):
        # 1. Query Parsing (Spacy NER)
        doc = self.nlp(question)
        
        question_entities = []
        for ent in doc.ents:
            if ent.label_ in ["ORDINAL", "CARDINAL"]: continue
            question_entities.append(ent)
        
        question_entities = list(dict.fromkeys(question_entities))

        question_entities.sort(key=lambda x: x.start_char)
        query_spans_text = [q.text.lower() for q in question_entities]
        
        num_spans = len(query_spans_text)
        if num_spans == 0:
            return [], [], [], []

        # 2. Batch Candidate Retrieval
        top_k = self.config.retrieval_candidate_top_k                 
        lambda_global = 0.1       
        lambda_local = 0.9        
        
        all_entity_embeddings = torch.from_numpy(self.entity_embeddings).float().to(self.device)
        question_emb_tensor = torch.from_numpy(question_emb).float().to(self.device)
        
        span_embs_numpy = self.config.embedding_model.encode(
            query_spans_text, 
            normalize_embeddings=True, 
            show_progress_bar=False, 
            batch_size=len(query_spans_text) 
        )
        span_embs = torch.from_numpy(span_embs_numpy).float().to(self.device)
        
        global_sims = torch.mm(question_emb_tensor.unsqueeze(0), all_entity_embeddings.t())
        local_sims_matrix = torch.mm(span_embs, all_entity_embeddings.t())
        combined_scores_matrix = lambda_local * local_sims_matrix + lambda_global * global_sims
        
        batch_vals, batch_inds = torch.topk(combined_scores_matrix, k=top_k, dim=1)
        
        batch_vals = batch_vals.cpu().numpy()
        batch_inds = batch_inds.cpu().numpy()
        
        lattice = []
        
        for i in range(num_spans):
            step_candidates = []
            for k in range(top_k):
                score = float(batch_vals[i][k])
                original_idx = int(batch_inds[i][k]) 
                
                entity_hash = self.entity_hash_ids[original_idx]
                
                if entity_hash in self.node_name_to_vertex_idx:
                    step_candidates.append({
                        "vertex": self.node_name_to_vertex_idx[entity_hash], 
                        "original_idx": original_idx,                          
                        "score": score,                                        
                        "hash": entity_hash                                    
                    })
            
            if step_candidates:
                lattice.append(step_candidates)
                
        num_spans = len(lattice)
        if num_spans == 0:
            return [], [], [], []

        # 3. Viterbi Algorithm
        lambda_dist = self.config.lambda_dist       
        penalty_base = self.config.penalty_base       
        max_hard_limit = 6       
        
        dp = []          
        backpointers = [] 
        
        first_step_scores = [cand["score"] for cand in lattice[0]]
        dp.append(np.array(first_step_scores))
        backpointers.append([-1] * len(lattice[0]))

        for i in range(1, num_spans):
            curr_candidates = lattice[i]
            prev_candidates = lattice[i-1]
            
            curr_dp_scores = np.full(len(curr_candidates), -np.inf)
            curr_backpointers = [-1] * len(curr_candidates)
            
            prev_vertices = [p["vertex"] for p in prev_candidates]

            for curr_idx, curr_cand in enumerate(curr_candidates):
                u_vertex = curr_cand["vertex"]
                emission = curr_cand["score"]
                
                best_prev_score = -np.inf
                best_prev_idx = -1
                
                try:
                    dists = self.graph.shortest_paths(source=u_vertex, target=prev_vertices, mode='all')[0]
                except:
                    dists = [float('inf')] * len(prev_vertices)

                for prev_idx, dist in enumerate(dists):
                    if dist == float('inf') or dist > max_hard_limit:
                        penalty = np.inf 
                    else:
                        penalty = lambda_dist * (penalty_base ** dist)
                    
                    prev_path_score = dp[i-1][prev_idx]
                    
                    if prev_path_score > -np.inf:
                        total_score = prev_path_score + emission - penalty
                        
                        if total_score > best_prev_score:
                            best_prev_score = total_score
                            best_prev_idx = prev_idx
                
                curr_dp_scores[curr_idx] = best_prev_score
                curr_backpointers[curr_idx] = best_prev_idx
                
            dp.append(curr_dp_scores)
            backpointers.append(curr_backpointers)

        # 4. Backtracking
        last_step_scores = dp[-1]
        best_last_idx = np.argmax(last_step_scores)
        
        final_path_indices = [0] * num_spans
        
        if last_step_scores[best_last_idx] <= -np.inf:
            final_path_indices = [np.argmax([c["score"] for c in layer]) for layer in lattice]
            self.infnumber += 1
        else:
            final_path_indices[-1] = best_last_idx
            curr_idx = best_last_idx
            for i in range(num_spans - 1, 0, -1):
                prev_idx = backpointers[i][curr_idx]
                if prev_idx == -1: 
                    prev_idx = np.argmax([c["score"] for c in lattice[i-1]])
                
                final_path_indices[i-1] = prev_idx
                curr_idx = prev_idx

        # 5. Format Output
        seed_entity_indices = []
        seed_entity_texts = []
        seed_entity_hash_ids = []
        seed_entity_scores = []
        
        for i, valid_idx in enumerate(final_path_indices):
            cand = lattice[i][valid_idx]
            
            idx = cand["original_idx"]
            hash_id = cand["hash"]
            score = cand["score"] 
            
            text = self.entity_embedding_store.hash_id_to_text.get(hash_id, "")
            
            seed_entity_indices.append(idx)
            seed_entity_texts.append(text)
            seed_entity_hash_ids.append(hash_id)
            seed_entity_scores.append(score)
            
        return seed_entity_indices, seed_entity_texts, seed_entity_hash_ids, seed_entity_scores

    def index(self, passages):
        self.node_to_node_stats = defaultdict(dict)
        self.entity_to_sentence_stats = defaultdict(dict)
        self.passage_embedding_store.insert_text(passages)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()
        existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids = self.load_existing_data(hash_id_to_passage.keys())
        if len(new_passage_hash_ids) > 0:
            new_hash_id_to_passage = {k : hash_id_to_passage[k] for k in new_passage_hash_ids}
            new_passage_hash_id_to_entities, new_sentence_to_entities = self.spacy_ner.batch_ner(new_hash_id_to_passage, self.config.max_workers)
            self.merge_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities)
        self.save_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        entity_nodes, sentence_nodes, passage_hash_id_to_entities, self.entity_to_sentence, self.sentence_to_entity = self.extract_nodes_and_edges(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        self.sentence_embedding_store.insert_text(list(sentence_nodes))
        self.entity_embedding_store.insert_text(list(entity_nodes))
        self.entity_hash_id_to_sentence_hash_ids = {}
        for entity, sentence in self.entity_to_sentence.items():
            entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
            self.entity_hash_id_to_sentence_hash_ids[entity_hash_id] = [self.sentence_embedding_store.text_to_hash_id[s] for s in sentence]
        self.sentence_hash_id_to_entity_hash_ids = {}
        for sentence, entities in self.sentence_to_entity.items():
            sentence_hash_id = self.sentence_embedding_store.text_to_hash_id[sentence]
            self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id] = [self.entity_embedding_store.text_to_hash_id[e] for e in entities]
        self.add_entity_to_passage_edges(passage_hash_id_to_entities)
        self.add_adjacent_passage_edges()
        self.augment_graph()
        output_graphml_path = os.path.join(self.config.working_dir, self.dataset_name, "TSDRAG.graphml")
        os.makedirs(os.path.dirname(output_graphml_path), exist_ok=True)   
        self.graph.write_graphml(output_graphml_path)

    def add_adjacent_passage_edges(self):
        passage_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        index_pattern = re.compile(r'^(\d+):')
        indexed_items = [
            (int(match.group(1)), node_key)
            for node_key, text in passage_id_to_text.items()
            if (match := index_pattern.match(text.strip()))
        ]
        indexed_items.sort(key=lambda x: x[0])
        for i in range(len(indexed_items) - 1):
            current_node = indexed_items[i][1]
            next_node = indexed_items[i + 1][1]
            self.node_to_node_stats[current_node][next_node] = 1.0

    def augment_graph(self):
        self.add_nodes()
        self.add_edges()

    def add_nodes(self):
        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()} 
        entity_hash_id_to_text = self.entity_embedding_store.get_hash_id_to_text()
        passage_hash_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text}
        
        passage_hash_ids = set(passage_hash_id_to_text.keys())
        
        for hash_id, text in all_hash_id_to_text.items():
            if hash_id not in existing_nodes:
                self.graph.add_vertex(name=hash_id, content=text)
        
        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}   
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[passage_id] 
            for passage_id in passage_hash_ids 
            if passage_id in self.node_name_to_vertex_idx
        ]

    def add_edges(self):
        edges = []
        weights = []
        
        for node_hash_id, node_to_node_stats in self.node_to_node_stats.items():
            for neighbor_hash_id, weight in node_to_node_stats.items():
                if node_hash_id == neighbor_hash_id:
                    continue
                edges.append((node_hash_id, neighbor_hash_id))
                weights.append(weight)
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights

    def add_entity_to_passage_edges(self, passage_hash_id_to_entities):
        passage_to_entity_count ={} 
        passage_to_all_score = defaultdict(int)
        for passage_hash_id, entities in passage_hash_id_to_entities.items():
            passage = self.passage_embedding_store.hash_id_to_text[passage_hash_id]
            for entity in entities:
                entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
                count = passage.count(entity)
                passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
                passage_to_all_score[passage_hash_id] += count
        for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
            score = count / passage_to_all_score[passage_hash_id]
            self.node_to_node_stats[passage_hash_id][entity_hash_id] = score

    def extract_nodes_and_edges(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        entity_nodes = set()
        sentence_nodes = set()
        passage_hash_id_to_entities = defaultdict(set)
        entity_to_sentence= defaultdict(set)
        sentence_to_entity = defaultdict(set)
        for passage_hash_id, entities in existing_passage_hash_id_to_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_hash_id_to_entities[passage_hash_id].add(entity)
        for sentence, entities in existing_sentence_to_entities.items():
            sentence_nodes.add(sentence)
            for entity in entities:
                entity_to_sentence[entity].add(sentence)
                sentence_to_entity[sentence].add(entity)
        return entity_nodes, sentence_nodes, passage_hash_id_to_entities, entity_to_sentence, sentence_to_entity

    def merge_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities):
        existing_passage_hash_id_to_entities.update(new_passage_hash_id_to_entities)
        existing_sentence_to_entities.update(new_sentence_to_entities)
        return existing_passage_hash_id_to_entities, existing_sentence_to_entities

    def save_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        with open(self.ner_results_path, "w") as f:
            json.dump({"passage_hash_id_to_entities": existing_passage_hash_id_to_entities, "sentence_to_entities": existing_sentence_to_entities}, f)