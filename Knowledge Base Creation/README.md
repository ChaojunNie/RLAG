# Knowledge Base Creation
We tailored retrieval strategies to each task’s specific characteristics:
## USMLE TASK
We merged keyword (Elasticsearch, BM25) and embedding searches. For each question-option pair, 200 document snippets were retrieved, vectorized, and filtered for semantic relevance.
(Note: You should have elasticsearch already)
```
Bash elastic+nvembed.sh
```
## Astronomy and Current Events TASKs
Documents were segmented (spaCy), embedded, and stored in FAISS. Questions were embedded to retrieve top matches via vector similarity, retaining $\leq$ 1,000 tokens per query.
```
CUDA_VISIBLE_DEVICES=0 python build_faiss_index.py
```