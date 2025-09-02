import json
from torch.utils.data import Dataset, random_split, DataLoader
from typing import Union, List, Dict
import torch
from transformers import AutoTokenizer, Qwen2TokenizerFast, LlamaTokenizerFast
from functools import partial
from tqdm import tqdm
from template import Template

class ContinualPreTrainDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: Union[Qwen2TokenizerFast, LlamaTokenizerFast], rank: int, block_size: int=256):
        self.data = {'input_ids':[], 'attention_mask':[]}
        buffer = []
        
        self.tokenizer = tokenizer
        
        if self.tokenizer.bos_token is None:
            if 'qwen' in str(type(self.tokenizer)).lower():
                self.tokenizer.add_special_tokens({'bos_token': '<|im_start|>'})
            
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '<EOS>'})

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        with open(file_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
            
        with open(file_path, "r", encoding="utf-8") as file:
            for line in tqdm(file, total=total_lines, desc="Tokenizing Data", unit='lines', disable=(rank!=0)):
                text = f"{self.tokenizer.bos_token}{line.strip()}{self.tokenizer.eos_token}"
                tokens = self.tokenizer(text, truncation=True)['input_ids']
                
                buffer.extend(tokens)
                
                while len(buffer) >= block_size:
                    self.data['input_ids'].append(buffer[:block_size])
                    self.data['attention_mask'].append([1]*block_size)
                    buffer = buffer[block_size:]

    def __len__(self):
        return len(self.data['input_ids'])
    
    def __getitem__(self, i):
        return {
            'input_ids': torch.tensor(self.data['input_ids'][i]),
            'attention_mask': torch.tensor(self.data['attention_mask'][i])}

class SFTDataset(Dataset):
     
    def __init__(self, file_path: str, tokenizer: AutoTokenizer):        
        with open(file_path, "r", encoding="utf-8") as file:
            self.data_list = json.load(file)
        self.tokenizer = tokenizer
        if 'qwen' in str(type(tokenizer)).lower():
            self.tokenizer.bos_token = '<|im_start|>'
            
        if self.tokenizer.chat_template is None:
            ChatML_template = (
                "{{% if not add_generation_prompt is defined %}}"
                "{{% set add_generation_prompt = false %}}"
                "{{% endif %}}"
                "{{% for message in messages %}}"
                "{{{{'{bos_token}' + message['role'] + '\n' + message['content'] + '{eos_token}' + '\n'}}}}"
                "{{% endfor %}}"
                "{{% if add_generation_prompt %}}"
                "{{{{ '{bos_token}assistant\n' }}}}"
                "{{% endif %}}"
            ) # see https://huggingface.co/docs/transformers/main/en/chat_templating?template=Mistral
            Chat_template = ChatML_template.format(bos_token=self.tokenizer.bos_token, eos_token=self.tokenizer.eos_token)
            self.tokenizer.chat_template = Chat_template

    
    def __getitem__(self, item):
        data = self.data_list[item]
        question = data["question"]
        answer = data['answer']
                        
        chosen_input_prefix = self.tokenizer.apply_chat_template([{"role": "user", "content": question}],
                                                                 tokenize=False, 
                                                                 add_generation_prompt=False)
        
        chosen_full_text = self.tokenizer.apply_chat_template([{"role": "user", "content": question},{"role": "assistant", "content": answer}],
                                                                 tokenize=False, 
                                                                 add_generation_prompt=False)
                
        chosen_prefix_tokens = self.tokenizer.encode(chosen_input_prefix, add_special_tokens=False)
        chosen_full_text_tokens_with_mask = self.tokenizer(chosen_full_text, add_special_tokens=False)

        input = {
            "chosen": chosen_full_text_tokens_with_mask,
            "chosen_prefix": chosen_prefix_tokens,
        }

        return input

    def __len__(self):
        return len(self.data_list)
        
class RLAGDataset(Dataset):
    
    information_en_template = """Question: {que}\nResponse:"""
    
    CoT_Instruction = Template.CoT_Instruction
    
    Information_Instruction = '''You are given a question or a prompt. Your task is NOT to directly answer the question, but instead to provide a list of relevant knowledge that could potentially be useful in addressing the question. Please adhere to the following instructions:
    1. Do NOT try to answer the question directly.
    2. List the pieces of knowledge that are most relevant to the question.
    3. Break down the response into distinct bullet points to make it easy to understand.
    4. Avoid unrelated details or any form of unnecessary elaboration.
    5. Be precise and informative.'''    
    
    def __init__(self, file_path: str, tokenizer: Union[Qwen2TokenizerFast, LlamaTokenizerFast], chosen_is_standard_answer:bool=False):        
        with open(file_path, "r", encoding="utf-8") as file:
            self.data_list = json.load(file)
        self.tokenizer = tokenizer
        if 'qwen' in str(type(tokenizer)).lower():
            self.tokenizer.bos_token = '<|im_start|>'
            
        if self.tokenizer.chat_template is None:
            ChatML_template = (
                "{{% if not add_generation_prompt is defined %}}"
                "{{% set add_generation_prompt = false %}}"
                "{{% endif %}}"
                "{{% for message in messages %}}"
                "{{{{'{bos_token}' + message['role'] + '\n' + message['content'] + '{eos_token}' + '\n'}}}}"
                "{{% endfor %}}"
                "{{% if add_generation_prompt %}}"
                "{{{{ '{bos_token}assistant\n' }}}}"
                "{{% endif %}}"
            ) # see https://huggingface.co/docs/transformers/main/en/chat_templating?template=Mistral
            Chat_template = ChatML_template.format(bos_token=self.tokenizer.bos_token, eos_token=self.tokenizer.eos_token)
            self.tokenizer.chat_template = Chat_template
            
            
    def __getitem__(self, item):
        data = self.data_list[item]
        question = data["question"]
        ori_prompt = data["ori_input"]
        rag_prompt = data["rag_input"]
        chosen = data["chosen"]
        rejected = data["rejected"]
        information_raw = data['ctx']
        information_list = information_raw.split('\n')
        information_chunk = []
        for c in information_list:
            c = self.tokenizer.bos_token + c + self.tokenizer.eos_token
            information_chunk.append(c)
        information_chunk = ''.join(information_chunk)
        
                        
        chosen_input_prefix = self.tokenizer.apply_chat_template([{"role": "user", "content": rag_prompt}],
                                                                 tokenize=False, 
                                                                 add_generation_prompt=False)
        rejected_input_prefix = self.tokenizer.apply_chat_template([{"role": "user", "content": ori_prompt}],
                                                                 tokenize=False, 
                                                                 add_generation_prompt=False)
        chosen_full_text = self.tokenizer.apply_chat_template([{"role": "user", "content": rag_prompt},
                                                               {"role": "assistant", "content": chosen}],
                                                                 tokenize=False, 
                                                                 add_generation_prompt=False)
        rejected_full_text = self.tokenizer.apply_chat_template([{"role": "user", "content": ori_prompt},
                                                                 {"role": "assistant", "content": rejected}],
                                                                 tokenize=False, 
                                                                 add_generation_prompt=False) 
        chosen_prefix_tokens = self.tokenizer.encode(chosen_input_prefix, add_special_tokens=False)
        rejected_prefix_tokens = self.tokenizer.encode(rejected_input_prefix, add_special_tokens=False)
        chosen_full_text_tokens_with_mask = self.tokenizer(chosen_full_text, add_special_tokens=False)
        rejected_full_text_tokens_with_mask = self.tokenizer(rejected_full_text, add_special_tokens=False)
        information_tokens = self.tokenizer(information_chunk, add_special_tokens=False, truncation=False)
        
        if chosen == rejected:
            beta_D=0.5
            beta_A=0.0
        else:
            beta_D=0.2
            beta_A=0.5
                     
        input = {
            "chosen": chosen_full_text_tokens_with_mask,
            "chosen_prefix": chosen_prefix_tokens,
            "rejected": rejected_full_text_tokens_with_mask,
            "rejected_prefix": rejected_prefix_tokens,
            "information": information_tokens,
            "beta_D": beta_D,
            "beta_A": beta_A
        }

        return input

    def __len__(self):
        return len(self.data_list)
    
def SFT_data_collate(batch: List[Dict[str, int]], 
                pad_token_id: int, 
                max_length: Union[None, int]=None, 
                if_mask_prompt: bool=True) -> Dict[str, List[torch.Tensor]]:
    
    batch_data = {
        "chosen": [],
        "chosen_attention_mask": [],
        "chosen_mask": []
    }
        
    max_len = max(len(item["chosen"]['input_ids']) for item in batch)
    
    for item in batch:
        chosen_prefix = torch.tensor(item["chosen_prefix"])
        
        out_ids = item["chosen"]['input_ids']
        pad_len = max_len - len(out_ids)
        out_ids_padding = [pad_token_id] * pad_len + out_ids
        out_attention_mask_padding = [0] * pad_len + item["chosen"]['attention_mask']
        
        mask = torch.ones(max_len).bool()

        mask[:pad_len] = False      

        if if_mask_prompt:
            mask[: pad_len + (len(chosen_prefix))] = False             
        batch_data["chosen"].append(torch.Tensor(out_ids_padding))
        batch_data[f"chosen_mask"].append(mask)       
        batch_data[f"chosen_attention_mask"].append(torch.Tensor(out_attention_mask_padding))
            
    tensor_stack = torch.stack(batch_data["chosen"])
    if max_length is not None:
        tensor_stack = tensor_stack[:, :max_length]
    batch_data["chosen"] = tensor_stack.to(torch.long)

    mask_stack = torch.stack(batch_data["chosen_mask"])
    if max_length is not None:
        mask_stack = mask_stack[:, :max_length]
    batch_data["chosen_mask"] = mask_stack
    
    attention_mask_stack = torch.stack(batch_data["chosen_attention_mask"])
    if max_length is not None:
        attention_mask_stack = attention_mask_stack[:, :max_length]
    batch_data["chosen_attention_mask"] = attention_mask_stack.to(torch.long)
        
    return batch_data
    
def RLAG_data_collate(batch: List[Dict[str, int]], 
                pad_token_id: int, 
                max_length: Union[None, int]=None, 
                if_mask_prompt: bool=True) -> Dict[str, List[torch.Tensor]]:
    batch_data = {
        "chosen": [],
        "rejected": [],
        "information": [],
        "chosen_attention_mask": [],
        "rejected_attention_mask": [],
        "information_attention_mask": [],
        "rejected_mask": [],
        "chosen_mask": [],
        "information_mask":[],
        "beta_D":[],
        "beta_A":[]
    }
    max_len = 0
    
    for key in ["chosen", "rejected", "information"]:
        current_max = max(len(item[key]['input_ids']) for item in batch)
        max_len = max(max_len, current_max)
    
    for item in batch:
        rejected_prefix = torch.tensor(item["rejected_prefix"])
        chosen_prefix = torch.tensor(item["chosen_prefix"])
        
        for key in ["chosen", "rejected", "information"]:
            out_ids = item[key]['input_ids']
            pad_len = max_len - len(out_ids)
            out_ids_padding = [pad_token_id] * pad_len + out_ids 
            out_attention_mask_padding = [0] * pad_len + item[key]['attention_mask']   
            
            mask = torch.ones(max_len).bool()

            mask[:pad_len] = False      

            if if_mask_prompt:
                if key == "chosen":
                    mask[: pad_len + (len(chosen_prefix))] = False            
                elif key == "rejected":
                    mask[: pad_len + (len(rejected_prefix))] = False
            batch_data[key].append(torch.Tensor(out_ids_padding))
            batch_data[f"{key}_mask"].append(mask)       
            batch_data[f"{key}_attention_mask"].append(torch.Tensor(out_attention_mask_padding))
                    
        for key_beta in ["beta_D", "beta_A"]:
            batch_data[key_beta].append(torch.tensor(item[key_beta]))
            
    for key in ["chosen", "rejected", "information"]:
        tensor_stack = torch.stack(batch_data[key])
        if max_length is not None:
            tensor_stack = tensor_stack[:, :max_length]
        batch_data[key] = tensor_stack.to(torch.long)
        
        attention_mask_stack = torch.stack(batch_data[f"{key}_attention_mask"])
        if max_length is not None:
            attention_mask_stack = attention_mask_stack[:, :max_length]
        batch_data[f"{key}_attention_mask"] = attention_mask_stack.to(torch.long)

        mask_stack = torch.stack(batch_data[f"{key}_mask"])
        if max_length is not None:
            mask_stack = mask_stack[:, :max_length]
        batch_data[f"{key}_mask"] = mask_stack
        
    for key_beta in ["beta_D", "beta_A"]:
        beta_stack = torch.stack(batch_data[key_beta])
        batch_data[key_beta] = beta_stack
        
    return batch_data
    
def getDataLoader(data_file_path: str, 
                tokenizer: Union[Qwen2TokenizerFast, LlamaTokenizerFast], 
                rank: int,
                train_size_ratio: float, 
                train_batch_size: int,
                val_batch_size: int,
                sequence_max_len: Union[None, int] = None,
                if_mask_prompt: bool = True,
                data_type: str = 'RLAG',
                chosen_is_standard_answer:bool=False):
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if data_type == 'RLAG':
        dataset = RLAGDataset(file_path=data_file_path, tokenizer=tokenizer, chosen_is_standard_answer=chosen_is_standard_answer)
        
        collate_fn = partial(
                            RLAG_data_collate,
                            pad_token_id=tokenizer.pad_token_id,
                            max_length=sequence_max_len,
                            if_mask_prompt=if_mask_prompt)
        
    elif data_type == 'SFT':
        dataset = SFTDataset(file_path=data_file_path, tokenizer=tokenizer)
        
        collate_fn = partial(
                            SFT_data_collate,
                            pad_token_id=tokenizer.pad_token_id,
                            max_length=sequence_max_len,
                            if_mask_prompt=if_mask_prompt)
        
    elif data_type == 'CPT':
        dataset = ContinualPreTrainDataset(file_path=data_file_path, tokenizer=tokenizer, block_size=256, rank=rank)    
        train_loader = DataLoader(dataset, 
                                  batch_size=train_batch_size, 
                                  num_workers=2, 
                                  shuffle=True, 
                                  pin_memory=True, 
                                  persistent_workers=True, 
                                  drop_last=True)
        return train_loader, None
    
    train_size = int(len(dataset) * train_size_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        collate_fn=collate_fn,
        num_workers=2, 
        pin_memory=True,    
        persistent_workers=True,  
        drop_last=True
    )
    
    if val_size != 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            collate_fn=collate_fn, 
            shuffle=False,
            drop_last=True,
            num_workers=2, 
            pin_memory=True,
            persistent_workers=True
        )
    else:
        val_loader = None
        
    return train_loader, val_loader