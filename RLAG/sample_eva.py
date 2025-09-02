import json
from tqdm import tqdm
import torch
import time
from typing import List, Callable, Union, Dict
from multiprocessing import current_process
import logging
from template import Template
from utils import compute_logprobs

def batch_generate(model, 
                   tokenizer, 
                   batch_prompt: List[str], 
                   system_content: str="You are a helpful assistant.", 
                   max_new_tokens: int=1024) -> List[str]:
    text = []
    messages_batch = [[{"role": "system", "content": system_content}, {"role": "user", "content": prompt}] for prompt in batch_prompt]
    
    text = tokenizer.apply_chat_template(
        messages_batch, 
        tokenize=False, 
        add_generation_prompt=True
    )
        
    device = next(model.parameters()).device
    model_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    model.eval()
    
    with torch.no_grad():  
        generated_ids = model.generate(
                                        **model_inputs,
                                        max_new_tokens=max_new_tokens,
                                        pad_token_id=tokenizer.eos_token_id,
                                        do_sample=False)
                                    
    res = []
    for i in range(len(generated_ids)):      
        response = tokenizer.decode(generated_ids[i][len(model_inputs.input_ids[i]):], skip_special_tokens=True)
        res.append(response)
        
    return res


def is_answer_match(answer: str, response: str) -> bool:
    answer = answer.lower()
    response = response.lower()
    return (answer == response or 
            response.endswith(answer) or 
            response.startswith(answer))

def get_Log_Prob_Inp(content, tokenizer, device):
    
        question: str = content["question"]
        sorted_keys = sorted(content["options"].keys())
        options = [content["options"][k] for k in sorted_keys]
        input_texts = [f"{question} {option}" for option in options]
        prefix = question
        
        k_list = []
        for input_text in input_texts:
            full_ids = tokenizer.encode(input_text, add_special_tokens=False)
            q_ids = tokenizer.encode(prefix, add_special_tokens=False)
            k = len(full_ids) - len(q_ids)
            k_list.append(k)
            
        inp_ids_attnm: Dict[str, List] = tokenizer(input_texts, padding=True, padding_side='left', add_special_tokens=False, return_tensors='pt').to(device)
        seq_len = inp_ids_attnm.input_ids.size(1)
        
        label_mask = []
        for k in k_list:
            mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            if k > 0:
                mask[-k:] = True 
            label_mask.append(mask)
        label_mask = torch.stack(label_mask).to(device)
        return inp_ids_attnm, label_mask

def mp_logprob_eva(local_model, tokenizer, rank, data_partition):
    
    correct = 0
    device = next(local_model.parameters()).device
            
    for content in tqdm(data_partition, total=len(data_partition), desc="MP LogProb val", unit='sample', disable=(rank!=0)):
        
        answer = content['answer']
        input_ids_attnm, mask = get_Log_Prob_Inp(   content=content,
                                                    tokenizer=tokenizer, 
                                                    device=device)
        options: List[str] = list(content["options"].values())
        with torch.no_grad():
            logits = local_model(**input_ids_attnm).logits.to(torch.float32)
            log_probs = compute_logprobs(logits, input_ids_attnm.input_ids, mask, use_sum=False)
            pred = options[torch.argmax(log_probs, dim=-1).item()]
        
        if answer == pred:
            correct += 1
    return correct
    
def log_prob_evaluate(model, tokenizer, config, logger):
    
    # ------------- logger ----------------
    ori_start = time.time()
    logger.info("Starting evaluate function")
    logger.info("Evaluation started")
    for handler in logger.handlers:
        handler.flush()
    
    archive_path=f"{config.state_dict_save_path}/Iteration-{config.iteration_id}-policy.pt"
    state_dict = torch.load(archive_path, map_location='cpu', weights_only=False)
    Iteration, metrics = state_dict['Iteration'], state_dict['metrics']
    print(f'Begin evaluation loading trained weights at Iteration {Iteration} from {archive_path} with metrics {json.dumps(metrics, indent=2)}')
    model.load_state_dict(state_dict['state'])
    model = model.to("cuda")

    
    correct = 0
    device = next(model.parameters()).device
    
    with open(config.val_data_path, "r", encoding='utf-8') as file:
        if config.val_data_path.endswith('json'):
            contents = json.load(file)
        elif config.val_data_path.endswith('jsonl'):
            contents = [json.loads(line) for line in file]
            
        for content in tqdm(contents, total=len(contents), desc="LogProb val", unit='sample'):
            
            answer = content['answer']
            input_ids_attnm, mask = get_Log_Prob_Inp(question=content['question'],
                                                     options=content['options'],
                                                     tokenizer=tokenizer, 
                                                     device=device)
            options: List[str] = list(content["options"].values())
            
            with torch.no_grad():
                logits = model(**input_ids_attnm).logits.to(torch.float32)
            log_probs = compute_logprobs(logits, input_ids_attnm.input_ids, mask, use_sum=False)
            
            pred = options[torch.argmax(log_probs, dim=-1).item()]
            
            if answer == pred:
                correct += 1
        
        end = time.time()
    total_accuracy = correct / len(contents)
    logger.info(f"Iteration-{config.iteration_id} | evaluate cost time:{(end-ori_start)/60:.1f} | 总正确率：{correct}/{len(contents)}={total_accuracy:.4f}")  
    print(f"\nIteration-{config.iteration_id} | evaluate cost time:{(end-ori_start)/60:.1f} | 总正确率：{correct}/{len(contents)}={total_accuracy:.4f}")

def evaluate(model, tokenizer, config, logger, batch_size: int=64):

    # ------------- logger ----------------
    ori_start = time.time()
    logger.info("Starting evaluate function")
    logger.info("Evaluation started")
    for handler in logger.handlers:
        handler.flush()

    archive_path=f"{config.state_dict_save_path}/Iteration-{config.iteration_id}-policy.pt"
    state_dict = torch.load(archive_path, map_location='cpu')
    Iteration, metrics = state_dict['Iteration'], state_dict['metrics']
    print(f'Begin evaluation loading trained weights at Iteration {Iteration} from {archive_path} with metrics {json.dumps(metrics, indent=2)}')
    model.load_state_dict(state_dict['state'])
    model = model.to("cuda")
    extract = lambda text: text[text.lower().find("answer:") + len("answer:"):].strip() if text.lower().find("answer:") != -1 else text.strip()
    
    en_instruction = Template.en_usmle_instruction
    en_prompt_prefix = Template.en_usmle_sample_without_ctx_prompt_prefix
    correct = 0

    with open(config.val_data_path, "r", encoding='utf-8') as file:

        contents = [json.loads(line) for line in file]

        for i in tqdm(range(0, len(contents), batch_size), desc="Evaluating", unit='batch'):

            batch_content = contents[i: i + batch_size]
            batch_inp = get_batch_inp(batch_content, en_prompt_prefix)
            res_all = batch_generate(model, tokenizer, batch_inp, en_instruction)
            
            for idx, content in enumerate(batch_content):
                answer = content["answer"]
                res = res_all[idx]
                
                if res[-1] == ';':
                    res = res[:-1]
                res = extract(res)    
                if is_answer_match(answer, res):                    
                    correct += 1                    
                
    end = time.time()
    total_accuracy = correct / len(contents)
    logger.info(f"Iteration-{config.iteration_id} | evaluate cost time:{(end-ori_start)/60:.1f} | ACC：{correct}/{len(contents)}={total_accuracy:.4f}")  
    print(f"\nIteration-{config.iteration_id} | evaluate cost time:{(end-ori_start)/60:.1f} | ACC：{correct}/{len(contents)}={total_accuracy:.4f}")   

 
def setup_logger(log_file):
    logger = logging.getLogger(current_process().name)  
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(processName)s] %(message)s')
    file_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger

def get_batch_inp(batch_content: List[str], ori_prefix: Union[Callable, None]=None) -> List[str]:
    ori_input = []
    for content in batch_content:
        
        question = content["question"]
        options = '; '.join(content['options'].values())
        
        if ori_prefix is not None:
            ori_input.append(ori_prefix(que=question, options=options))
    return ori_input

def get_LogProb_SampleInp(question: List[str], options: Dict[str, str], tokenizer, device, CoT: bool=False):
    if CoT:
        options_list = [f'\n<answer>{options[k]}</answer>' for k in sorted(options.keys())]
    else:
        options_list = [options[k] for k in sorted(options.keys())]

    q0_len = len(tokenizer.encode(question[0], add_special_tokens=False))
    q1_len = len(tokenizer.encode(question[1], add_special_tokens=False))
    
    input_texts = [f"{question[0]} {opt}" for opt in options_list] + [f"{question[1]} {opt}" for opt in options_list]
    options_size = len(options_list)
    
    encoded = tokenizer(input_texts, add_special_tokens=False, return_length=True)
    lengths = encoded['length'] if 'length' in encoded else [len(ids) for ids in encoded['input_ids']]
    k_list = [l - q0_len for l in lengths[:options_size]] + [l - q1_len for l in lengths[options_size:]]
    inp_ids_attnm = tokenizer(
        input_texts, 
        padding=True, 
        padding_side='left', 
        add_special_tokens=False, 
        return_tensors='pt'
    ).to(device)
    
    seq_len = inp_ids_attnm['input_ids'].size(1)
    k_tensor = torch.tensor(k_list, device=device).unsqueeze(1)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    label_mask = positions >= (seq_len - k_tensor)
    
    return inp_ids_attnm, label_mask, options_size
def sample_logprob(model, tokenizer, rank, config, data_partition=None):
        
    device = next(model.parameters()).device
    
    # ------------ Sample template ----------------
    ctx_template = Template.logprob_sample_ctx_template
    ori_template = Template.logprob_sample_ori_template
    
    # ------------ logprob sampling ------------
    chosen_correct = 0
    rejected_correct = 0
    chosen_failure = 0
    
    for content in tqdm(data_partition, total=len(data_partition), desc="LogProb Sampling", unit='sample', disable=(rank!=0)):
        
        answer = content['answer']
        que = content['question']
        options: List[str] = list(content["options"].values())
        ctx_question = ctx_template(ctx=content['ctx'], que=que)
        ori_question = ori_template(que=que)
        
        input_ids_attnm, mask, options_size = get_LogProb_SampleInp(question=[ctx_question, ori_question], 
                                                    options=content['options'],
                                                    tokenizer=tokenizer, 
                                                    device=device,
                                                    CoT=False)
        input_ids_attnm = {k:v.to(device) for k, v in input_ids_attnm.items()}
        mask = mask.to(device)
        with torch.no_grad():
            logits = model(**input_ids_attnm).logits.to(torch.float32) 
        ctx_log_probs = compute_logprobs(logits[:options_size, :, :], input_ids_attnm['input_ids'][:options_size, :], mask[:options_size, :], use_sum=False)
        ori_log_probs = compute_logprobs(logits[options_size:, :, :], input_ids_attnm['input_ids'][options_size:, :], mask[options_size:, :], use_sum=False)
        ctx_pred = options[torch.argmax(ctx_log_probs, dim=-1).item()]
        ori_pred = options[torch.argmax(ori_log_probs, dim=-1).item()]
        
        ctx_right = False
        ori_right = False
        
        content["rejected"] = ori_pred
        content["chosen"] = ctx_pred
        
        if answer == ctx_pred:
            ctx_right = True
            chosen_correct += 1
            
        if answer == ori_pred:
            ori_right = True
            rejected_correct += 1
        
        if ori_right and (not ctx_right):
            chosen_failure += 1
            
    
    result = {"data": data_partition,
              "chosen_correct": chosen_correct,
              "rejected_correct": rejected_correct,
              "chosen_failure": chosen_failure}
    
    return result