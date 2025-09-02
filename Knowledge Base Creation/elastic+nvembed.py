import json
from elasticsearch import Elasticsearch
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import gc
from typing import List
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--input_data_path', type=str, required=True, help="")
    parser.add_argument('--output_data_path', type=str, required=True, help="")
    return parser.parse_args()
args = parse_args()
es = Elasticsearch(['http://localhost:9200'])
retrieve_size = 300
query_template = lambda que:{
    "size": retrieve_size,
    "query": {
        "match": {
            "txt": que
        }
    }
}
index_name = "output"
model = AutoModel.from_pretrained("NV-Embed-v2", trust_remote_code=True, device_map="balanced")
model.eval()  
tokenizer = AutoTokenizer.from_pretrained('Qwen2-7B-Instruct', use_fast=True)
def get_embedding(query,
                 query_prefix: str,
                 max_length: int=1024,
                 batch_size: int=128,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if len(query) <= batch_size:
        with torch.no_grad():
            embeddings = model.encode(query, instruction=query_prefix, max_length=max_length)
            embeddings_nor = F.normalize(embeddings.clone().detach().to(device), p=2, dim=1)
            return embeddings_nor
    else:
        embeddings_nor = []
        with torch.no_grad():
            for i in range(0, len(query), batch_size):
                batch = query[i:i + batch_size]
                embeddings = model.encode(batch, instruction=query_prefix, max_length=max_length)
                embeddings = embeddings.clone().detach().to(device)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                embeddings_nor.append(embeddings)
        embeddings_nor = torch.cat(embeddings_nor, dim=0)
        return embeddings_nor


data_file_path:str = args.input_data_path
with open(data_file_path, "r", encoding="utf-8") as file:
    
    if data_file_path.endswith('json'):
        contents = json.load(file)
    elif data_file_path.endswith('jsonl'):
        contents = [json.loads(line) for line in file]
        
    for num, content in enumerate(tqdm(contents, desc="Processing contents", unit='samples')):
        question = content["question"]
        options = content["options"]
        es_retrieval_options = set()
        
        
        for idx in ["A", "B", "C", "D", "E"]:
            option = options[idx]
            que = question + " " + (option + " ") * 6
            query = query_template(que)
            es_response = es.search(index=index_name, body=query)
            for hit in es_response['hits']['hits']:
                es_retrieval_text = hit["_source"]['txt'].strip()
                if len(tokenizer.encode(es_retrieval_text)) <= 200:
                    es_retrieval_options.add(hit["_source"]['txt'].strip())
                
        es_retrieval_options = list(es_retrieval_options)        
        query_before_embedding = [question]
        
        #-------------------embedding----------------
        query_embeddings = get_embedding(query=query_before_embedding, query_prefix="Instruct: Given a complex medical question, retrieve relevant documents or resources that provide the correct answer \nQuery: ")
        passage_embeddings = get_embedding(query=es_retrieval_options, query_prefix="Medical textbooks, peer-reviewed medical journal articles, and authoritative medical guidelines")
        with torch.no_grad():
            similarity_scores = (passage_embeddings @ query_embeddings.T).flatten()
            top_scores, top_indices = torch.topk(similarity_scores, k=1, dim=0)

        top_scores_np = top_scores.cpu().numpy()
        top_indices_np = top_indices.cpu().numpy()

        filtered_results = [
            (es_retrieval_options[i], top_scores_np[idx]) 
            for idx, i in enumerate(top_indices_np)
        ]
        ctx = "\n".join([text for text, _ in filtered_results])
        ctx_score = str(top_scores_np)
        content["idx"] = num + 1
        content["ctx"] = ctx
        content["ctx_score"] = ctx_score
        print(f"{num + 1}/{len(contents)}, score:{top_scores_np}")

with open(args.output_data_path, "w", encoding="utf-8") as newfile:
    json.dump(contents, newfile, ensure_ascii=False, indent=2)
        
            