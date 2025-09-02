import torch
torch.backends.cuda.matmul.allow_tf32 = True
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from typing import Dict
from utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import random
import hydra
from omegaconf import OmegaConf, DictConfig
import resource
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    StateDictType,
    CPUOffload,
    FullStateDictConfig
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import time
import json
import functools
from collections import defaultdict
from sample_eva import setup_logger, sample_logprob, mp_logprob_eva
from Dataprepare import getDataLoader
import math
from math import ceil
import copy
from template import Template
import logging

class WorkPipeline:
    def __init__(self, config) -> None:
        
        self.config: Dict = config
        self.world_size = torch.cuda.device_count()
        self.total_epochs = config.total_epochs
        self.per_iteration_epochs = config.per_iteration_epochs
        self.state_archive = None
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        self.LOSS = self.config.loss
        # ----------- logger -----------
        os.makedirs(os.path.dirname(config.logfile_path), exist_ok=True)
        self.logger = logging.getLogger()
        self.logger.handlers.clear()
        logging.basicConfig(filename=config.logfile_path, level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger.info("Creating Logger")
        print("Logger created:", self.logger, self.logger.handlers, self.logger.level)
        self.logger.info("Logger started")
        
        self.policy_dtype = getattr(torch, self.config.policy_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path, padding_side='left', use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.splitted_data_paths = [self.config.data_file_path]
        
    def split_data(self, shuffle: bool=False):
        rag_inp_template = Template.en_usmle_sample_with_ctx_prompt_prefix
        ori_inp_template = Template.en_usmle_sample_without_ctx_prompt_prefix
        with open(self.config.data_file_path, "r", encoding="utf-8") as file:
            if self.config.data_file_path.endswith('json'):
                contents = json.load(file)
            elif self.config.data_file_path.endswith('jsonl'):
                contents = [json.loads(line) for line in file]
                
            for content in contents:
                options = '; '.join(content['options'].values())
                content["rag_input"]=rag_inp_template(ctx=content['ctx'], que=content["question"], options=options)
                content["ori_input"]=ori_inp_template(que=content["question"], options=options)
        if shuffle:
            random.shuffle(contents)
        
        total = len(contents)
        chunk_size = math.ceil(total / self.config.iteration_num)  

        os.makedirs(self.config.data_split_path, exist_ok=True) 

        base_name = os.path.splitext(os.path.basename(self.config.data_file_path))[0]

        self.splitted_data_paths = []
        for i in range(self.config.iteration_num):
            start = i * chunk_size
            end = start + chunk_size
            chunk = contents[start:end]
            output_file = os.path.join(self.config.data_split_path, f"{base_name}_part{i+1}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False, indent=4)
            print(f"Saving {len(chunk)} data to {output_file}")
            self.splitted_data_paths.append(output_file)

        print("Data splitting completed.")

    def main_worker(self, rank, world_size, tokenizer, policy, config, data_file_path, iter_id, next_stage):

        # ----------- Initialize distributed environment -----------
        init_distributed(rank, world_size=world_size, port=config.mp_port)
        self.setup_logger()
        iteration_start = time.time()
        while next_stage != "exit":
            process = getattr(self, next_stage)
            next_stage = process(rank, data_file_path, tokenizer, policy, world_size, config, iter_id)
            dist.barrier()
        if config.loss == 'RLAG':
            rank0_print(f"Iteration {iter_id} training completed in {(time.time() - iteration_start)/60:.2f} mins")
        else:
            rank0_print(f"{config.loss} training {config.per_iteration_epochs} EPOCHS completed in {(time.time() - iteration_start)/60:.2f} mins")

        if dist.is_initialized():
            dist.destroy_process_group()
    
    def setup_logger(self):
        self.logger = setup_logger(self.config.logfile_path)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
    
    def setup_components(self, data_file_path, iteration_id, config):
        if config.loss == 'RLAG':
            print(f"Iteration: {iteration_id + 1}, training on: {data_file_path}")
        else:
            print(f"Training: {data_file_path}")   
        policy = AutoModelForCausalLM.from_pretrained(self.config.base_model_name_or_path, low_cpu_mem_usage=True, torch_dtype=self.policy_dtype)
        disable_dropout(policy)
        if iteration_id > 0:
            filename = "Iteration-" + str(iteration_id)+ "-" + "policy.pt"
            state_archive = os.path.join(config.state_dict_save_path, filename)
            state_dict = torch.load(state_archive, map_location='cpu', weights_only=False)
            Iteration, metrics = state_dict['Iteration'], state_dict['metrics']
            print(f'loading pre-trained weights at Iteration {Iteration} from {state_archive} with metrics {json.dumps(metrics, indent=2)}')
            self.logger.info(f'loading pre-trained weights at Iteration {Iteration} from {state_archive} with metrics {json.dumps(metrics, indent=2)}')
            policy.load_state_dict(state_dict['state']) 
            
        if config.archive is not None:
            state_dict = torch.load(config.archive, map_location='cpu', weights_only=False)
            Iteration, metrics = state_dict['Iteration'], state_dict['metrics']
            print(f'loading pre-trained weights from {config.archive} with metrics {json.dumps(metrics, indent=2)}')
            self.logger.info(f'loading pre-trained weights from {config.archive} with metrics {json.dumps(metrics, indent=2)}')
            policy.load_state_dict(state_dict['state'])

        return policy
    
    def sample(self, rank, data_file_path, tokenizer, policy, world_size, config, iteration_id):
        
        sample_ori_start = time.time()
        with open(data_file_path, "r") as f:
            data = json.load(f)
        partitioned_data = data[rank::world_size]
        
        device = torch.device(f"cuda:{rank}")
        local_model = copy.deepcopy(policy).to(device).eval()
        result = sample_logprob(local_model, tokenizer, rank, config, data_partition=partitioned_data)
    
        all_results = gather_return_objects(result, world_size)
        gather_results = {"data": [],
                "chosen_correct": [],
                "rejected_correct": [],
                "chosen_failure": []}
        dist.barrier()
        for results in all_results:
            gather_results['data'].extend(results['data'])
            gather_results['chosen_correct'].append(results['chosen_correct']) 
            gather_results['rejected_correct'].append(results['rejected_correct'])
            gather_results['chosen_failure'].append(results['chosen_failure'])
        for key in ['chosen_correct', 'rejected_correct', 'chosen_failure']:
            gather_results[key] = sum(gather_results[key])    
            
        if rank == 0:
            
            self.logger.info(f"Iteration-{iteration_id} Sampling | chosen_correct: {gather_results['chosen_correct']}  rejected_correct: {gather_results['rejected_correct']} chosen_failure: {gather_results['chosen_failure']}, Sample cost time:{time.time() - sample_ori_start}")
            print(f"Iteration-{iteration_id} Sampling | chosen_correct: {gather_results['chosen_correct']}  rejected_correct: {gather_results['rejected_correct']} chosen_failure: {gather_results['chosen_failure']}")
        
            with open(data_file_path, "w", encoding="utf-8") as newfile:
                json.dump(gather_results['data'], newfile, ensure_ascii=False, indent=2)

            print("="*15+"Sampling completed"+"="*15)
            self.logger.info(f"Iteration-{iteration_id} Sampling completed.")
        
        del local_model
        torch.cuda.empty_cache()
        
        return "RLAG_trainer"
    
    def RLAG_trainer(self, rank, data_file_path, tokenizer, policy, world_size, config, iteration_id):
        # ------------ Set parameters -----------
        update_step = 0
        gamma = config.gamma
        gradient_accumulation_steps = config.gradient_accumulation_steps
        num_epochs = self.per_iteration_epochs
        train_size_ratio = float(config.train_size_ratio)
        val_batch_size = int(config.val_batch_size)

        # ----------- Load dataset ------------
        train_batch_size = world_size * gradient_accumulation_steps * config.per_device_batch_size
        train_loader, val_loader = getDataLoader(
            data_file_path=data_file_path,
            rank=rank,
            tokenizer=tokenizer,
            train_size_ratio=train_size_ratio,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            if_mask_prompt=True,
            data_type=config.loss,
            chosen_is_standard_answer=config.chosen_is_standard_answer
        )
        rank0_print("="*15,f"train batch size:{train_batch_size}","="*15)
        total_steps = len(train_loader) * num_epochs
        warmup_steps = ceil(config.warmup_ratio * total_steps) if total_steps != 1 else 0
        rank0_print("="*15,f"total steps:{total_steps}, warmup steps:{warmup_steps}","="*15)
        
        # ---------- FSDP ---------
        wrap_class = get_block_class_from_model(policy, config.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class}) 
        
        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=True,
            sync_module_states=False,
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.fsdp_policy_mp) if config.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype) 
        policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)
        print('Loaded model on rank', rank)
        dist.barrier()              

        # ------------ Set optimizer -----------
        rank0_print(f'Using {config.optimizer} optimizer')
        optimizer = getattr(torch.optim, config.optimizer)(policy.parameters(), lr=config.lr, eps=1e-7)    
        lr_cosine_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.5 * (1.0 - math.cos(math.pi * step / warmup_steps)) if step < warmup_steps else 1.0)
        
        def FSDP_clip_gradient(policy, max_grad_norm):
            """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
            return policy.clip_grad_norm_(max_grad_norm).item()
        
        def save(iteration_id, policy, metrics, output_dir: str, logger, filename):
            """Save policy, gathering from all processes and saving only on the rank 0 process."""
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
                policy_state_dict = policy.state_dict()
            filename = "Iteration-" + str(iteration_id)+ "-" + filename
            output_path = os.path.join(output_dir, filename)
            self.state_archive = output_path
            if rank == 0:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                check_and_delete_files(path=output_dir, logger=logger, threshold_gb=200)
                torch.save({'Iteration': iteration_id,
                            'metrics': metrics,
                            'state': policy_state_dict}, output_path)
                
                print(f"Iteration: {iteration_id} state_dict 保存到 {output_path}")
                logger.info(f"Iteration: {iteration_id} state_dict 保存到 {output_path}")  
            del policy_state_dict
            dist.barrier()
        
        # --------- Training --------------
        for epoch in range(num_epochs):
            policy.train()  
            
            for batch in train_loader:
                batch_metrics = defaultdict(list)
                loss_metrics = defaultdict(list)
                start_time = time.time()
                
                for microbatch_idx in range(gradient_accumulation_steps):
                    
                    global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, gradient_accumulation_steps, rank)
                    local_microbatch = slice_and_move_batch_for_device(global_microbatch, rank, world_size, rank)
                    concatenated_batch = get_concatenated_batch(local_microbatch)    
                    loss, information_rewards, chosen_rewards, rejected_rewards = compute_rlag_batch_loss(
                        config=config,
                        batch = concatenated_batch,
                        policy_model=policy,
                        reference_model=None,
                        gamma=gamma
                    )
                    (loss / gradient_accumulation_steps).backward()
                    all_devices_losses = all_gather_if_needed(loss.detach(), rank, world_size)
                    all_information_rewards = all_gather_if_needed(information_rewards.detach(), rank, world_size)
                    all_chosen_rewards = all_gather_if_needed(chosen_rewards.detach(), rank, world_size)
                    all_rejected_rewards = all_gather_if_needed(rejected_rewards.detach(), rank, world_size)
                    loss_metrics['loss'].extend(all_devices_losses.float().cpu().numpy().tolist())
                    batch_metrics['information_rewards'].extend(all_information_rewards.float().cpu().numpy().tolist())
                    batch_metrics['chosen_rewards'].extend(all_chosen_rewards.float().cpu().numpy().tolist())
                    batch_metrics['rejected_rewards'].extend(all_rejected_rewards.float().cpu().numpy().tolist())
                    
                grad_norm = FSDP_clip_gradient(policy, config.max_grad_norm)
                optimizer.step()
                lr_cosine_scheduler.step()
                optimizer.zero_grad()
                update_step += 1
                mean_metrics = {k: sum(v)/len(v) for k,v in batch_metrics.items()}  
                loss_mean_metrics = {k: sum(v)/len(v) for k,v in loss_metrics.items()}  
                reward_all = sum([v for _,v in mean_metrics.items()])       
                rank0_print(
                    f"Iteration:{iteration_id} Ep {epoch + 1} (Update {update_step:04d}): "
                    f"grad norm {grad_norm:.2f}, "
                    f"Learning rate: {optimizer.param_groups[0]['lr']}, "
                    f"loss:{loss_mean_metrics},"
                    f"mean batch metrics:{formatted_dict(mean_metrics)}, "
                    f"reward_all:{reward_all},"
                    f"update cost time:{(time.time()-start_time):.2f}"
                    )
                if rank==0:
                    self.logger.info(
                                    f"Iteration:{iteration_id} Ep {epoch + 1} (Update {update_step:04d}): "
                                    f"grad norm {grad_norm:.2f}, "
                                    f"Learning rate: {optimizer.param_groups[0]['lr']}, "
                                    f"loss:{loss_mean_metrics},"
                                    f"mean batch metrics:{formatted_dict(mean_metrics)}, "
                                    f"reward_all:{reward_all},"
                                    f"update cost time:{(time.time()-start_time):.2f}"
                                    )
        save(iteration_id, policy, formatted_dict(mean_metrics), config.state_dict_save_path, self.logger, "policy.pt")
        del policy
        torch.cuda.empty_cache()
        return "eval"
    
    def eval(self, rank, _, tokenizer, policy, world_size, config, iteration_id):
       
        # ----------- GPU -----------
        device = torch.device(f"cuda:{rank}")
        
        eva_start = time.time()
        state_dict = torch.load(self.state_archive, map_location='cpu', weights_only=False)
        Iteration, metrics = state_dict['Iteration'], state_dict['metrics']
        rank0_print(f'Begin evaluation loading trained weights at Iteration {Iteration} from {self.state_archive} with metrics {json.dumps(metrics, indent=2)}')
        local_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, low_cpu_mem_usage=True, torch_dtype=config.policy_dtype).to(device).eval()
        local_model.load_state_dict(state_dict['state'])
        
        with open(config.val_data_path, "r", encoding="utf-8") as f:
            if config.val_data_path.endswith('json'):
                data = json.load(f)
            elif config.val_data_path.endswith('jsonl'):
                data = [json.loads(line) for line in f]
            
        partitioned_data = data[rank::world_size]
        
        result = mp_logprob_eva(local_model, tokenizer, rank, data_partition=partitioned_data)
        all_results = gather_return_objects(result, world_size)
        correct = sum(all_results)
        dist.barrier()   
                    
        if rank == 0:
            total_accuracy = correct / len(data)
            self.logger.info(f"Iteration-{iteration_id} | acc：{correct}/{len(data)}={total_accuracy:.2f}, eval cost: {(time.time() - eva_start):.2f} s")  
            print(f"\nIteration-{iteration_id} | acc：{correct}/{len(data)}={total_accuracy:.2f}, eval cost: {(time.time() - eva_start):.2f} s")
        del local_model
        torch.cuda.empty_cache()
        
        return "exit"
    
    def start_pipeline(self):
        print('Starting', self.world_size, f'processes for {self.LOSS}')
        iter_id = 0        
        for epoch_idx in range(self.total_epochs):
            if self.LOSS == "RLAG":
                print("=" * 5, f"EPOCH {epoch_idx + 1} BEGIN", "=" * 5)
                self.split_data()
                Init_stage = "sample"
            else:
                Init_stage = self.LOSS
            for data_file_path in self.splitted_data_paths:
                policy=self.setup_components(data_file_path, iter_id, self.config)
                iter_id += 1
                mp.spawn(self.main_worker, nprocs=self.world_size, 
                        args=(self.world_size, self.tokenizer, policy, self.config, data_file_path, iter_id, Init_stage), join=True)

    def SFT(self, rank, data_file_path, tokenizer, policy, world_size, config, Placeholder):
        # ------------ Set parameters -----------
        update_step = 0
        gradient_accumulation_steps = config.gradient_accumulation_steps
        eval_freq = config.eval_freq
        num_epochs = self.per_iteration_epochs
        train_size_ratio = float(config.train_size_ratio)
        val_batch_size = int(config.val_batch_size)

        # ----------- Load dataset ------------
        train_batch_size = world_size * gradient_accumulation_steps * config.per_device_batch_size
        train_loader, val_loader = getDataLoader(
            data_file_path=data_file_path,
            rank=rank,
            tokenizer=tokenizer,
            train_size_ratio=train_size_ratio,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            if_mask_prompt=True,
            data_type=config.loss
        )
        rank0_print("="*15,f"train batch size:{train_batch_size}","="*15)
        total_steps = len(train_loader) * num_epochs
        warmup_steps = ceil(config.warmup_ratio * total_steps) if total_steps != 1 else 0
        rank0_print("="*15,f"total steps:{total_steps}, warmup steps:{warmup_steps}","="*15)
        
                # ---------- FSDP ---------
        wrap_class = get_block_class_from_model(policy, config.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class}) 
        
        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=True,
            sync_module_states=False,
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.fsdp_policy_mp) if config.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype) 
        policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)
        print('Loaded model on rank', rank)
        self.logger.info(f'Loaded model on rank{rank}')

        # ------------ Set optimizer -----------
        rank0_print(f'Using {config.optimizer} optimizer')
        self.logger.info(f'Using {config.optimizer} optimizer')
        optimizer = getattr(torch.optim, config.optimizer)(policy.parameters(), lr=config.lr, eps=1e-7)    
        lr_cosine_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.5 * (1.0 - math.cos(math.pi * step / warmup_steps)) if step < warmup_steps else 1.0)
        
        def FSDP_clip_gradient(policy, max_grad_norm):
            """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
            return policy.clip_grad_norm_(max_grad_norm).item()
        
        def save(iteration_id, policy, metrics, output_dir: str, logger, filename):
            """Save policy, gathering from all processes and saving only on the rank 0 process."""
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
                policy_state_dict = policy.state_dict()
            filename = "Iteration-" + str(iteration_id)+ "-" + filename
            output_path = os.path.join(output_dir, filename)
            if rank == 0:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torch.save({'Iteration': iteration_id,
                            'metrics': metrics,
                            'state': policy_state_dict}, output_path)

                print(f"Iteration: {iteration_id} state_dict saved to {output_path}")
                logger.info(f"Iteration: {iteration_id} state_dict saved to {output_path}")
            del policy_state_dict
            dist.barrier()
        # ------------ Set optimizer -----------
        optimizer = getattr(torch.optim, config.optimizer)(policy.parameters(), lr=config.lr, eps=1e-7)
        lr_cosine_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.5 * (1.0 - math.cos(math.pi * step / warmup_steps)) if step < warmup_steps else 1.0)
        for epoch in range(num_epochs): 
            for batch in train_loader:
                # -------------- Evaluate -----------------
                if update_step % eval_freq == 0:
                    batch_val_metrics = defaultdict(list)
                    policy.eval()
                    for val_batch in val_loader:
                        local_val_batch = slice_and_move_batch_for_device(val_batch, rank, world_size, rank)
                        with torch.no_grad():
                            val_loss = compute_sft_loss(batch=local_val_batch, policy_model=policy)
                        all_devices_val_losses = all_gather_if_needed(val_loss.detach(), rank, world_size)
                        batch_val_metrics['val_loss'].extend(all_devices_val_losses.float().cpu().numpy().tolist())   
                    mean_val_metrics = {k: sum(v)/len(v) for k,v in batch_val_metrics.items()}          
                    rank0_print(
                        f"Ep {epoch + 1} (Update {update_step:04d}): "
                        f"mean val batch metrics:{formatted_dict(mean_val_metrics)}, "
                        )
                    self.logger.info(
                        f"Ep {epoch + 1} (Update {update_step:04d}): "
                        f"mean batch metrics:{formatted_dict(mean_val_metrics)}, "
                        )
                    policy.train()
                # -------------- Training -----------------
                batch_metrics = defaultdict(list)
                start_time = time.time()
                
                for microbatch_idx in range(gradient_accumulation_steps):
                    
                    global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, gradient_accumulation_steps, rank)
                    local_microbatch = slice_and_move_batch_for_device(global_microbatch, rank, world_size, rank)

                    loss = compute_sft_loss(batch=local_microbatch, policy_model=policy)
                    (loss / gradient_accumulation_steps).backward()
                grad_norm = FSDP_clip_gradient(policy, config.max_grad_norm)
                optimizer.step()
                lr_cosine_scheduler.step()
                optimizer.zero_grad()
                all_devices_losses = all_gather_if_needed(loss.detach(), rank, world_size)
                batch_metrics['loss'].extend(all_devices_losses.float().cpu().numpy().tolist())
                update_step += 1
                mean_metrics = {k: sum(v)/len(v) for k,v in batch_metrics.items()}          
                rank0_print(
                    f"Ep {epoch + 1} (Update {update_step:04d}): "
                    f"grad norm {grad_norm:.2f}, "
                    f"Learning rate: {optimizer.param_groups[0]['lr']}, "
                    f"mean batch metrics:{formatted_dict(mean_metrics)}, "
                    f"update cost time:{(time.time()-start_time):.2f}"
                    )
                self.logger.info(
                    f"Ep {epoch + 1} (Update {update_step:04d}): "
                    f"grad norm {grad_norm:.2f}, "
                    f"Learning rate: {optimizer.param_groups[0]['lr']}, "
                    f"mean batch metrics:{formatted_dict(mean_metrics)}, "
                    f"update cost time:{(time.time()-start_time):.2f}"
                    )
            save(epoch+1, policy, formatted_dict(mean_metrics), config.state_dict_save_path, self.logger, "policy.pt")
        return "exit"
                
    def CPT(self, rank, data_file_path, tokenizer, policy, world_size, config, Placeholder):
        # ------------ Set parameters -----------
        update_step = 0
        gradient_accumulation_steps = config.gradient_accumulation_steps
        num_epochs = self.per_iteration_epochs
        train_size_ratio = float(config.train_size_ratio)
        val_batch_size = int(config.val_batch_size)

        # ----------- Load dataset ------------
        train_batch_size = world_size * gradient_accumulation_steps * config.per_device_batch_size
        train_loader, _ = getDataLoader(
            data_file_path=data_file_path,
            rank=rank,
            tokenizer=tokenizer,
            train_size_ratio=train_size_ratio,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            if_mask_prompt=True,
            data_type=config.loss
        )
        rank0_print("="*15,f"train batch size:{train_batch_size}","="*15)
        total_steps = len(train_loader) * num_epochs
        warmup_steps = ceil(config.warmup_ratio * total_steps) if total_steps != 1 else 0
        rank0_print("="*15,f"total steps:{total_steps}, warmup steps:{warmup_steps}","="*15)
                # ---------- FSDP ---------
        wrap_class = get_block_class_from_model(policy, config.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class}) 
        
        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=True,
            sync_module_states=False,
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.fsdp_policy_mp) if config.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)
        print('Loaded model on rank', rank)
        self.logger.info(f'Loaded model on rank{rank}')
        dist.barrier()              

        # ------------ Set optimizer -----------
        rank0_print(f'Using {config.optimizer} optimizer')
        self.logger.info(f'Using {config.optimizer} optimizer')
        optimizer = getattr(torch.optim, config.optimizer)(policy.parameters(), lr=config.lr, eps=1e-7)    
        lr_cosine_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.5 * (1.0 - math.cos(math.pi * step / warmup_steps)) if step < warmup_steps else 1.0)
        
        def FSDP_clip_gradient(policy, max_grad_norm):
            """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
            return policy.clip_grad_norm_(max_grad_norm).item()
        
        def save(iteration_id, policy, metrics, output_dir: str, logger, filename):
            """Save policy, gathering from all processes and saving only on the rank 0 process."""
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
                policy_state_dict = policy.state_dict()
            filename = "Iteration-" + str(iteration_id)+ "-" + filename
            output_path = os.path.join(output_dir, filename)
            if rank == 0:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torch.save({'Iteration': iteration_id,
                            'metrics': metrics,
                            'state': policy_state_dict}, output_path)
                
                print(f"Iteration: {iteration_id} state_dict 保存到 {output_path}")
                logger.info(f"Iteration: {iteration_id} state_dict 保存到 {output_path}")  
            del policy_state_dict
            dist.barrier()

        # ------------ Set optimizer -----------
        optimizer = getattr(torch.optim, config.optimizer)(policy.parameters(), lr=config.lr, eps=1e-7)
        lr_cosine_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.5 * (1.0 - math.cos(math.pi * step / warmup_steps)) if step < warmup_steps else 1.0)
        for epoch in range(num_epochs):
            policy.train()  
            
            for batch in train_loader:
                batch_metrics = defaultdict(list)
                start_time = time.time()
                
                for microbatch_idx in range(gradient_accumulation_steps):
                    
                    global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, gradient_accumulation_steps, rank)
                    local_microbatch = slice_and_move_batch_for_device(global_microbatch, rank, world_size, rank)
                    
                    loss = compute_cpt_loss(
                        batch = local_microbatch,
                        policy_model=policy)
                    (loss / gradient_accumulation_steps).backward()
                    
                grad_norm = FSDP_clip_gradient(policy, config.max_grad_norm)
                optimizer.step()
                lr_cosine_scheduler.step()
                optimizer.zero_grad()
                all_devices_losses = all_gather_if_needed(loss.detach(), rank, world_size)
                batch_metrics['loss'].extend(all_devices_losses.float().cpu().numpy().tolist())
                update_step += 1
                mean_metrics = {k: sum(v)/len(v) for k,v in batch_metrics.items()}          
                rank0_print(
                    f"Ep {epoch + 1} (Update {update_step:04d}): "
                    f"grad norm {grad_norm:.2f}, "
                    f"Learning rate: {optimizer.param_groups[0]['lr']}, "
                    f"mean batch metrics:{formatted_dict(mean_metrics)}, "
                    f"update cost time:{(time.time()-start_time):.2f}"
                    )
                self.logger.info(
                    f"Ep {epoch + 1} (Update {update_step:04d}): "
                    f"grad norm {grad_norm:.2f}, "
                    f"Learning rate: {optimizer.param_groups[0]['lr']}, "
                    f"mean batch metrics:{formatted_dict(mean_metrics)}, "
                    f"update cost time:{(time.time()-start_time):.2f}"
                    )
            
            save(epoch+1, policy, formatted_dict(mean_metrics), config.state_dict_save_path, self.logger, "policy.pt")                   
        return "exit"

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    
    start_time = time.time()    
    # ----------- config ------------
    OmegaConf.resolve(config)
    config_save_path = config.logfile_path[:-7] + 'config.yaml'
    if config.mp_port is None:
        config.mp_port = get_open_port()
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    with open(config_save_path, 'w') as f:
        OmegaConf.save(config, f)
    print(OmegaConf.to_yaml(config))    
    # ----------- pipeline ----------- 
    pipeline = WorkPipeline(config)
    pipeline.start_pipeline()
    print(f'Training cost:{(time.time() - start_time)/60:.2f} mins')

if __name__ == "__main__":
    main()