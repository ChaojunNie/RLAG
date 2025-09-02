import faiss
import torch
from transformers import AutoModel
import torch.nn.functional as F
from tqdm import tqdm
import json
import torch.nn as nn
import uuid
import math
from collections import defaultdict
import os

def get_embedding(  query,
                    model,
                    query_prefix: str,
                    max_length: int=1024,
                    batch_size: int=128,
                    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if len(query) <= batch_size:
        with torch.no_grad():
            embeddings = model.encode(query, instruction=query_prefix, max_length=max_length)
            embeddings_nor = F.normalize(embeddings.clone().detach().to(device), p=2, dim=1)
            return embeddings_nor.cpu().numpy()
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
        return embeddings_nor.cpu().numpy()

def build_faiss_index(embedding_model_path_or_name: str,
                      raw_data_path: str,
                      output_file_path: str,
                      batch_size: int,
                      window_size:int,
                      step_size:int) -> None:
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    print(f'embedding model path:{embedding_model_path_or_name}')
    model = AutoModel.from_pretrained(embedding_model_path_or_name, trust_remote_code=True, device_map="balanced")

    embedding_dim = model.module.config.hidden_size if isinstance(model, nn.DataParallel) else model.config.hidden_size 
    print(f'\nembedding dimension:\t{embedding_dim}\n')
    index = faiss.IndexFlatL2(embedding_dim)

    doc_ids = []
    doc_content_dict = {} 

    total_sentences = 0
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        content = []
        for line in f:
            content.append(line.strip())
        if len(content)>=window_size:
            total_sentences += math.ceil((len(content) - window_size + 1) / (step_size + 1))
                
    progress_sentences = tqdm(total=total_sentences, unit="sentences", desc="Processing sentences")
    num = 0
    num_sentences = len(content)
    text_before_embeddings = []

    for start_idx in range(0, num_sentences - window_size + 1, (step_size+1)):
        sentence_chunk = content[start_idx:start_idx + window_size]
        text_chunk = ". ".join(sentence_chunk)
        text_before_embeddings.append(text_chunk)
        doc_id = str(uuid.uuid4())
        doc_ids.append(doc_id)
        doc_content_dict[doc_id] = text_chunk 

        if len(text_before_embeddings) >= batch_size:
            embeddings = get_embedding(query=text_before_embeddings,model=model, query_prefix="")
            index.add(embeddings)
            text_before_embeddings = []
            num += 1
            
            progress_sentences.update(batch_size)
        if num >= 10:
            faiss.write_index(index, output_file_path)
            print(f"\nIndex saved to {output_file_path}\n")
            
            with open(output_file_path + "_doc_ids.txt", "w") as f:
                f.write("\n".join(doc_ids) + "\n")
            print(f"Document IDs saved to {output_file_path}_doc_ids.txt")
            
            with open(output_file_path + "_doc_contents.json", "w") as f:
                json.dump(doc_content_dict, f, ensure_ascii=False, indent=4)
            print(f"Document contents saved to {output_file_path}_doc_contents.json")
            num = 0
            
            
    if text_before_embeddings:
        embeddings = get_embedding(query=text_before_embeddings, model=model, query_prefix="")
        index.add(embeddings)
        progress_sentences.update(len(text_before_embeddings))
    
    faiss.write_index(index, output_file_path)
    print(f"\nIndex saved to {output_file_path}\n")
    
    with open(output_file_path + "_doc_ids.txt", "w") as f:
        f.write("\n".join(doc_ids) + "\n")
    print(f"Document IDs saved to {output_file_path}_doc_ids.txt")
    
    with open(output_file_path + "_doc_contents.json", "w") as f:
        json.dump(doc_content_dict, f, ensure_ascii=False, indent=4)
    print(f"Document contents saved to {output_file_path}_doc_contents.json")

class FaissQuery:
    def __init__(self, 
                 embedding_model_path_or_name: str,
                 index_file_path: str,
                 doc_ids_file_path: str,
                 doc_contents_file_path: str):
        self.model = AutoModel.from_pretrained(embedding_model_path_or_name, trust_remote_code=True, device_map="balanced")
        self.model.eval()

        self.index = faiss.read_index(index_file_path)

        with open(doc_ids_file_path, "r") as f:
            self.doc_ids = f.read().splitlines()

        with open(doc_contents_file_path, "r") as f:
            self.doc_contents = json.load(f)

    def query(self, query: List[str], top_k: int = 5) -> List[str]:
        query_embedding = get_embedding(query=query, model=self.model, query_prefix="Instruct: Given a complex medical question, retrieve relevant documents or resources that provide the correct answer \nQuery: ")

        D, I = self.index.search(query_embedding, top_k)  
        for idxs, distances in zip(I, D):
            
            query_results = defaultdict(list)
            
            for idx, distance in zip(idxs, distances):
                if idx != -1:
                    try:
                        doc_id = self.doc_ids[idx]
                    except:
                        print(idx)
                    content = self.doc_contents.get(doc_id, "Content not found")
                    if content not in query_results:
                        query_results['content'].append(content)
                        query_results['score'].append(distance)
                        query_results['content_length'].append(len(content))
                        
        sorted_results_by_score = { key: [query_results[key][i] for i in sorted(range(len(query_results['score'])), key = lambda x: query_results["score"][x])] for key in query_results}
        
        length = 0
        results = []
        scores = []
        for result in sorted_results_by_score['content']:
            if length + len(result) <= 1000:
                length += len(result)
                results.append(result)
                scores.append(sorted_results_by_score['score'])
            else:
                break
        return results, scores


# 查询示例
if __name__ == "__main__":
    
    embedding_model_path = "NV-Embed-v2"
    index_file_path = "Astronomy/Faiss/faiss_nvembed_index"
    doc_ids_file_path = "Astronomy/Faiss/faiss_nvembed_index_doc_ids.txt"
    doc_contents_file_path = "Astronomy/Faiss/faiss_nvembed_index_doc_contents.json"
    output_file_path = "Astronomy/Faiss/faiss_nvembed_index"
    print(f'embedding model path:{embedding_model_path}')
    
    build_faiss_index(embedding_model_path_or_name = embedding_model_path,
                      raw_data_path="Astronomy/Astronomy_document.txt",
                      output_file_path = output_file_path,
                      batch_size=4096,
                      window_size=1,
                      step_size=0)
    
    faiss_query = FaissQuery(
        embedding_model_path_or_name=embedding_model_path,
        index_file_path=index_file_path,
        doc_ids_file_path=doc_ids_file_path,
        doc_contents_file_path=doc_contents_file_path
    )
    with open("Astronomy/astronomy_train.json", "r", encoding='utf-8') as file:
        contents = json.load(file)
        for content in tqdm(contents, total=len(contents)):
            query_text = [content["question"]]
            
            results, scores = faiss_query.query(query=query_text, top_k=20)
            scores = [str(item) for item in scores]
            content["ctx"] = "\n".join(results)
    with open("Astronomy/astronomy_train_CTX.json", "w", encoding='utf-8') as newfile:        
        json.dump(contents, newfile, ensure_ascii=False, indent=2)