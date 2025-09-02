import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, Union, List
import os
import socket
import shutil

class RLAGLoss(nn.Module):
    """
    RLAG Loss
    """
    def __init__(self, 
                 Abalation:bool, 
                 NoRejected: bool,
                 NoInformation: bool,
                 NoGamma: bool, 
                 PureChosen: bool, 
                 PureInformation: bool, 
                 PureRejected: bool, 
                 FixBeta_AD: bool, 
                 chosen_is_standard_answer:bool,
                 beta_D: float=0.001, 
                 beta_A: float=0.1, 
                 gamma: float=0.1):
        super(RLAGLoss, self).__init__()
        self.beta_D = beta_D
        self.beta_A = beta_A
        self.gamma = gamma
        self.Abalation = Abalation
        self.Norejected = NoRejected
        self.NoInformation = NoInformation
        self.NoGamma = NoGamma
        self.PureChosen = PureChosen
        self.PureInformation = PureInformation
        self.PureRejected = PureRejected
        self.FixBeta_AD = FixBeta_AD
        self.chosen_is_standard_answer = chosen_is_standard_answer
    def forward(self,
                policy_information_logps:torch.Tensor,
                policy_rag_logps:torch.Tensor,
                policy_rejected_logps:torch.Tensor):
        logits = self.beta_D * policy_information_logps + self.beta_A* policy_rag_logps - self.beta_A * policy_rejected_logps - self.gamma
            
        loss = -F.logsigmoid(logits)
        
        return loss.mean()
    
def check_and_delete_files(path, logger, threshold_gb=1):
    threshold_bytes = threshold_gb * (1024 ** 3)
    free_space = shutil.disk_usage(path).free
    i = 0
    d = 0
    while (free_space < threshold_bytes) and (i < 50):
        file_path = os.path.join(path, f"Iteration-{i}-policy.pt")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                free_space = shutil.disk_usage(path).free
                d += 1
            except Exception as e:
                logger.info(f"Delete {file_path}：{e}")
        i += 1
    free_space = shutil.disk_usage(path).free

def niechaojun_chat(model, 
         tokenizer, 
         prompt, 
         system_content: str="You are a helpful assistant", 
         history: Union[List[Dict], None]=None, 
         max_new_tokens: int=512, 
         temperature: float=0.0):
    if history == None:
        history = [{"role": "system", "content": system_content}]
        messages = history
    else:
        if 'system' not in history[0]['role']:
            messages = [{"role": "system", "content": system_content}] + history
        else:
            messages = history
    messages += [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    device = next(model.parameters()).device
    model_inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    if temperature <= 0.0:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    else:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
    
    response = tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
    messages.append({"role": "assistant", "content": response})
    return response, messages
    
            
def compute_logprobs(logits:torch.Tensor, 
                     labels:torch.Tensor, 
                     mask, 
                     use_sum: bool=False) -> torch.Tensor:
    assert logits.shape[:-1] == labels.shape
    logits = logits[:, :-1, :]  
    labels = labels[:, 1:]
    mask = mask[:, 1:]
    
    select_logprobs = torch.gather(input=logits.log_softmax(-1), dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    if use_sum:
        return (select_logprobs * mask).sum(-1)
    else:
        return (select_logprobs * mask).sum(-1) / mask.sum(-1)
    
def compute_rlag_batch_loss(batch: Dict[str, torch.Tensor], 
                       policy_model: nn.Module, 
                       reference_model: Union[nn.Module, None],
                       gamma: float,
                       config) -> List[float]:
    
    loss_fn = RLAGLoss(Abalation=config.Ablation, 
                       NoRejected=config.NoRejected,
                       NoGamma=config.NoGamma,
                       NoInformation=config.NoInformation,
                       PureChosen=config.PureChosen,
                       PureInformation=config.PureInformation,
                       PureRejected=config.PureRejected,
                       FixBeta_AD=config.FixBeta_AD, 
                       chosen_is_standard_answer=config.chosen_is_standard_answer,
                       beta_D=batch['beta_D'], 
                       beta_A=batch['beta_A'], 
                       gamma=gamma)
        
    all_policy_logps: torch.Tensor = compute_logprobs(
        logits = policy_model(  batch["concatenated"],
                                attention_mask=batch["concatenated_attention_mask"]).logits.to(torch.float32),
        labels = batch["concatenated"],
        mask = batch["concatenated_mask"],
        use_sum=False
    )
    policy_rag_logps: torch.Tensor = all_policy_logps[:batch["chosen"].shape[0]]
    policy_rejected_logps: torch.Tensor = all_policy_logps[batch["chosen"].shape[0]:(batch["chosen"].shape[0] + batch["rejected"].shape[0])]
    policy_information_logps: torch.Tensor = all_policy_logps[(batch["chosen"].shape[0] + batch["rejected"].shape[0]):]
    reference_rag_logps: torch.Tensor = torch.tensor([0]*policy_rag_logps.shape[0]).to(policy_model.device)
    reference_rejected_logps: torch.Tensor  = torch.tensor([0]*policy_rejected_logps.shape[0]).to(policy_model.device)
    reference_information_logps:torch.Tensor = torch.tensor([0]*policy_information_logps.shape[0]).to(policy_model.device)
    
    information_rewards = policy_information_logps - reference_information_logps
    chosen_rewards = policy_rag_logps - reference_rag_logps
    rejected_rewards = reference_rejected_logps - policy_rejected_logps
    policy_rag_logps_clips = -0.2
    policy_rag_logps = torch.minimum(policy_rag_logps, torch.tensor(policy_rag_logps_clips))
    
    policy_rejected_logps_clip = -5.0
    policy_rejected_logps = torch.maximum(policy_rejected_logps, torch.tensor(policy_rejected_logps_clip))
         
    loss = loss_fn(
        policy_information_logps = policy_information_logps,
        policy_rag_logps = policy_rag_logps,
        policy_rejected_logps = policy_rejected_logps
    )    
    return loss, information_rewards.mean().detach(), chosen_rewards.mean().detach(), rejected_rewards.mean().detach()

def compute_sft_loss(batch: Dict[str, torch.Tensor],
                     policy_model: nn.Module):
    
    sft_logps = compute_logprobs(logits=policy_model(batch['chosen'], 
                                                     attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32),
                                 labels=batch['chosen'],
                                 mask=batch['chosen_mask'],
                                 use_sum=False)
    return -sft_logps.mean()

def compute_sft_loss_byModel(batch: Dict[str, torch.Tensor],
                     policy_model: nn.Module):
    
    labels = batch['chosen'].clone()
    labels[:, :-1] = batch['chosen'][:, 1:]
    labels[:, -1] = -100  # 最后一个位置无法预测
    outputs = policy_model(batch['chosen'], attention_mask=batch['chosen_attention_mask'], labels=labels)
    loss = outputs.loss
    
    return loss.to(torch.float32)

def compute_cpt_loss(batch: Dict[str, torch.Tensor],
                     policy_model: nn.Module):
    cpt_logps = compute_logprobs(logits=policy_model(batch['input_ids'], 
                                                     attention_mask=batch['attention_mask']).logits.to(torch.float32),
                                 labels=batch['input_ids'],
                                 mask=batch['attention_mask'],
                                 use_sum=False)
    return -cpt_logps.mean()
            
def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def gather_return_objects(value, world_size):
    if isinstance(value, torch.Tensor):
        value = value.cpu()
    object_list = [None] * world_size
    dist.all_gather_object(object_list, value)
    return object_list
            
def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)
        
def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int, device: str) -> Dict:
    """Slice a batch into chunks, and move each chunk to the specified device."""
    chunk_size = len(list(batch.values())[0]) // world_size
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}
    return on_device

def get_block_class_from_model(model: torch.nn.Module, block_class_name: str) -> torch.nn.Module:
    """Get the class of a block from a model, using the block's class name."""
    for module in model.modules():
        if module.__class__.__name__ == block_class_name:
            return module.__class__
    raise ValueError(f"Could not find block class {block_class_name} in model {model}")


def init_distributed(rank: int, world_size: int, master_addr: str = 'localhost', port: int = 12355, backend: str = 'nccl'):
    print(rank, 'initializing distributed')
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank) 
    
def all_gather_if_needed(values: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if world_size == 1:
        return values

    all_values = [torch.empty_like(values).to(rank) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)

def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)) 
        return s.getsockname()[1]
    
def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)

def get_concatenated_batch(batch):
    
    concatenated_batch = {}
    concatenated_batch['chosen'] = batch['chosen']
    concatenated_batch['rejected'] = batch['rejected']
    concatenated_batch['information'] = batch['information']
    concatenated_batch['beta_D'] = batch['beta_D']
    concatenated_batch['beta_A'] = batch['beta_A']
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = batch[k]            
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((concatenated_batch[concatenated_key], batch[k]), dim=0)
    for k in batch:
        if k.startswith('information') and isinstance(batch[k], torch.Tensor):
            concatenated_key = k.replace('information', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((concatenated_batch[concatenated_key], batch[k]), dim=0)        
    return concatenated_batch
        
def chat(model, tokenizer, prompt, system_content: str="You are an advanced reasoning assistant", max_new_tokens: int=512):
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    device = next(model.parameters()).device
    model_inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    
    response = tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)

    return response