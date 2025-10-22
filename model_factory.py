import os.path
from transformers import AutoConfig, AutoModelForCausalLM
import torch
from transformers import AutoTokenizer, LlamaTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from accelerate import load_checkpoint_and_dispatch

from config import ShareConfig
from tqdm import tqdm
import math
import torch.nn as nn
from utils import match_state_dict
from calib import Calib
import numpy as np
import random
from prepare_data import prepare_data
from utils import compute_num_basis
from group import change_model, update_model
from models.gpt2 import ShareGPT2LMHeadModel
from typing import Optional
from models.llama import ShareLlamaForCausalLM
from models.opt import ShareOPTForCausalLM
from models.model_utils import build_dynamic_basis_collection
from models.mistral import ShareMistralForCausalLM


def reallocate_llama3_2(model_config, all_w_lists, all_k_lists, max_ranks, min_k=1):

    print("\n" + "="*80)
    print("Executing V-Maxout by GroupSize Parameter Reallocation (Q,K -> V)...")

    # --- 0. 准备数据 ---
    if not all(key in all_w_lists for key in ['q', 'k', 'v']):
        return all_k_lists
        
    q_k_list = list(all_k_lists['q'])
    k_k_list = list(all_k_lists['k'])
    v_k_list_initial = list(all_k_lists['v'])

    q_w_list, k_w_list, v_w_list = all_w_lists['q'], all_w_lists['k'], all_w_lists['v']
    q_groups, k_groups, v_groups = getattr(model_config, "q_groups"), getattr(model_config, "k_groups"), getattr(model_config, "v_groups")
    max_rank_q, max_rank_k, max_rank_v = max_ranks['q'], max_ranks['k'], max_ranks['v']

    # --- 1. 计算V模块的总参数“赤字” ---
    params_deficit_v = 0
    for i, k_val in enumerate(v_k_list_initial):
        needed_k = max_rank_v - k_val
        if needed_k > 0:
            params_deficit_v += needed_k * v_w_list[i]
    print(f"  V-Module needs {params_deficit_v:.0f} parameters to max out all k-values.")
    if params_deficit_v == 0:
        return all_k_lists

    # --- 2. 确定Q和K的募捐目标 (按4:1) ---
    q_target_revenue = params_deficit_v * 0.8
    k_target_revenue = params_deficit_v * 0.2
    
    # --- 3. 按【分组大小】从Q和K中平均削减k值来“募捐” ---
    params_revenue_q, params_revenue_k = 0, 0
    
    # 从Q募捐
    total_q_layers = sum(len(g) for g in q_groups)
    if total_q_layers > 0:
        # 按分组大小，计算每个Q组需要贡献多少参数
        q_revenues_params = [q_target_revenue * (len(g) / total_q_layers) for g in q_groups]
        for i in range(len(q_k_list)):
            # 将参数换算成需要削减的k值
            k_reduction = int(round(q_revenues_params[i] / q_w_list[i])) if q_w_list[i] > 0 else 0
            original_k = q_k_list[i]
            # 执行削减
            q_k_list[i] = max(min_k, original_k - k_reduction)
            # 记录实际募集到的参数量
            params_revenue_q += (original_k - q_k_list[i]) * q_w_list[i]

    # 从K募捐 (逻辑同上)
    total_k_layers = sum(len(g) for g in k_groups)
    if total_k_layers > 0:
        k_revenues_params = [k_target_revenue * (len(g) / total_k_layers) for g in k_groups]
        for i in range(len(k_k_list)):
            k_reduction = int(round(k_revenues_params[i] / k_w_list[i])) if k_w_list[i] > 0 else 0
            original_k = k_k_list[i]
            k_k_list[i] = max(min_k, original_k - k_reduction)
            params_revenue_k += (original_k - k_k_list[i]) * k_w_list[i]
            
    total_revenue = params_revenue_q + params_revenue_k
    print(f"  Collected from Q: {params_revenue_q:.0f}, from K: {params_revenue_k:.0f}. Total collected: {total_revenue:.0f}")

    # --- 4. 将募集到的资金按【分组大小】分配给V，并处理溢出 ---
    final_v_k_list = list(v_k_list_initial)
    params_spillover = 0
    
    if total_revenue > 0:
        total_layers_in_v = sum(len(g) for g in v_groups)
        if total_layers_in_v > 0:
            v_subsidies_params = [total_revenue * (len(g) / total_layers_in_v) for g in v_groups]
            for i in range(len(v_k_list_initial)):
                cost_per_rank = v_w_list[i]
                k_increase = int(round(v_subsidies_params[i] / cost_per_rank)) if cost_per_rank > 0 else 0
                new_k = v_k_list_initial[i] + k_increase
                if new_k > max_rank_v:
                    params_spillover += (new_k - max_rank_v) * cost_per_rank
                    final_v_k_list[i] = max_rank_v
                else:
                    final_v_k_list[i] = new_k
    
    print(f"  Parameter spillover collected from V: {params_spillover:.0f}")

    # --- 5. 将【溢出的参数】按【分组大小】退还给Q和K ---
    if params_spillover > 0 and total_revenue > 0:
        q_refund_params = int(round(params_spillover * (params_revenue_q / total_revenue)))
        k_refund_params = params_spillover - q_refund_params
        
        if q_refund_params > 0:
            total_q_layers = sum(len(g) for g in q_groups)
            if total_q_layers > 0:
                q_refunds_params = [q_refund_params * (len(g) / total_q_layers) for g in q_groups]
                for i in range(len(q_k_list)):
                    k_increase = int(round(q_refunds_params[i] / q_w_list[i])) if q_w_list[i] > 0 else 0
                    q_k_list[i] = min(max_rank_q, q_k_list[i] + k_increase)

        if k_refund_params > 0:
            total_k_layers = sum(len(g) for g in k_groups)
            if total_k_layers > 0:
                k_refunds_params = [k_refund_params * (len(g) / total_k_layers) for g in k_groups]
                for i in range(len(k_k_list)):
                    k_increase = int(round(k_refunds_params[i] / k_w_list[i])) if k_w_list[i] > 0 else 0
                    k_k_list[i] = min(max_rank_k, k_k_list[i] + k_increase)

    # --- 6. 更新并返回k_lists字典 ---
    all_k_lists['q'] = q_k_list
    all_k_lists['k'] = k_k_list
    all_k_lists['v'] = final_v_k_list
    
    print(f"  Final Q k_list after refund: {q_k_list}")
    print(f"  Final K k_list after refund: {k_k_list}")
    print(f"  Final V k_list after subsidy: {final_v_k_list}")
    print("="*80 + "\n")
    
    return all_k_lists


def reallocate_llama3(model_config, all_w_lists, all_k_lists, max_ranks, tax_rate=0.20, min_k=1):

    print("\n" + "="*80)
    print("Executing Final, Simplified Parameter Reallocation (by Group Size)...")

    # --- 0. 准备数据 ---
    if not all(key in all_w_lists for key in ['q', 'k', 'v']):
        return all_k_lists # 直接返回未修改的k_lists
        
    q_k_list_initial = list(all_k_lists['q'])
    k_k_list_initial = list(all_k_lists['k'])
    v_k_list_initial = list(all_k_lists['v'])

    q_w_list, k_w_list, v_w_list = all_w_lists['q'], all_w_lists['k'], all_w_lists['v']
    q_groups, k_groups, v_groups = getattr(model_config, "q_groups"), getattr(model_config, "k_groups"), getattr(model_config, "v_groups")

    # --- 1. 按【参数量】从Q和K征税 ---
    params_revenue_q, params_revenue_k = 0, 0
    new_q_k_list = list(q_k_list_initial)
    for i, k_val in enumerate(q_k_list_initial):
        tax_k = int(round(k_val * tax_rate))
        new_k = max(min_k, k_val - tax_k)
        params_revenue_q += (k_val - new_k) * q_w_list[i]
        new_q_k_list[i] = new_k

    new_k_k_list = list(k_k_list_initial)
    for i, k_val in enumerate(k_k_list_initial):
        tax_k = int(round(k_val * tax_rate))
        new_k = max(min_k, k_val - tax_k)
        params_revenue_k += (k_val - new_k) * k_w_list[i]
        new_k_k_list[i] = new_k

    total_params_subsidy = params_revenue_q + params_revenue_k
    print(f"  Parameter revenue collected: {total_params_subsidy:.0f}")

    # --- 2. 按【分组大小】将参数补贴分配给V，并计算溢出 ---
    final_v_k_list = list(v_k_list_initial)
    params_spillover = 0
    max_rank_v = max_ranks.get('v', 1024) # 获取v的max_rank
    if total_params_subsidy > 0:
        total_layers_in_v = sum(len(g) for g in v_groups)
        if total_layers_in_v > 0:
            # 2a. 初步分配参数预算
            v_subsidies_params = [total_params_subsidy * (len(g) / total_layers_in_v) for g in v_groups]
            
            # 2b. 应用补贴，检查溢出，并收集
            for i in range(len(v_k_list_initial)):
                cost_per_rank = v_w_list[i]
                k_increase = int(round(v_subsidies_params[i] / cost_per_rank))
                
                new_k = v_k_list_initial[i] + k_increase
                
                if new_k > max_rank_v:
                    params_spillover += (new_k - max_rank_v) * cost_per_rank
                    final_v_k_list[i] = max_rank_v
                else:
                    final_v_k_list[i] = new_k
    
    print(f"  Parameter spillover collected from V: {params_spillover:.0f}")

    # --- 3. 将【溢出的参数】按【分组大小】退还给Q和K ---
    if params_spillover > 0 and total_params_subsidy > 0:
        q_refund_params = int(round(params_spillover * (params_revenue_q / total_params_subsidy)))
        k_refund_params = params_spillover - q_refund_params
        
        # 退还给 Q
        max_rank_q = max_ranks.get('q', 4096)
        if q_refund_params > 0:
            total_q_layers = sum(len(g) for g in q_groups)
            if total_q_layers > 0:
                q_refunds_params = [q_refund_params * (len(g) / total_q_layers) for g in q_groups]
                for i in range(len(new_q_k_list)):
                    k_increase = int(round(q_refunds_params[i] / q_w_list[i]))
                    new_q_k_list[i] = min(max_rank_q, new_q_k_list[i] + k_increase)

        # 退还给 K (逻辑同上)
        max_rank_k = max_ranks.get('k', 1024)
        if k_refund_params > 0:
            total_k_layers = sum(len(g) for g in k_groups)
            if total_k_layers > 0:
                k_refunds_params = [k_refund_params * (len(g) / total_k_layers) for g in k_groups]
                for i in range(len(new_k_k_list)):
                    k_increase = int(round(k_refunds_params[i] / k_w_list[i]))
                    new_k_k_list[i] = min(max_rank_k, new_k_k_list[i] + k_increase)

    # --- 4. 更新最终的k_list ---
    setattr(model_config, "dynamic_basis_q_proj", new_q_k_list)
    setattr(model_config, "dynamic_basis_k_proj", new_k_k_list)
    setattr(model_config, "dynamic_basis_v_proj", final_v_k_list)

    all_k_lists['q'] = new_q_k_list
    all_k_lists['k'] = new_k_k_list
    all_k_lists['v'] = final_v_k_list
    
    print(f"  Final Q k_list after refund: {new_q_k_list}")
    print(f"  Final K k_list after refund: {new_k_k_list}")
    print(f"  Final V k_list after subsidy: {final_v_k_list}")
    print("="*80 + "\n")
    
    return all_k_lists

def reallocate_k_budget(model_config, tax_rate=0.10, min_k=1, max_rank=4096):

    print("\n" + "="*80)
    print("Executing Final Closed-Loop Budget Reallocation (Q,K <-> V)...")

    if not all(hasattr(model_config, f"dynamic_basis_{p}_proj") for p in ['q', 'k', 'v']):
        print("  Skipping reallocation: Required k_lists not found in config.")
        return model_config
        
    # --- 1. 从Q和K征税，并【分开】记录税收额 ---
    q_k_list_initial = list(getattr(model_config, "dynamic_basis_q_proj"))
    k_k_list_initial = list(getattr(model_config, "dynamic_basis_k_proj"))
    
    q_tax_revenue = 0
    k_tax_revenue = 0
    
    new_q_k_list = []
    for k_val in q_k_list_initial:
        tax = int(round(k_val * tax_rate))
        new_k = max(min_k, k_val - tax)
        q_tax_revenue += (k_val - new_k)
        new_q_k_list.append(new_k)

    new_k_k_list = []
    for k_val in k_k_list_initial:
        tax = int(round(k_val * tax_rate))
        new_k = max(min_k, k_val - tax)
        k_tax_revenue += (k_val - new_k)
        new_k_k_list.append(new_k)

    total_tax_revenue = q_tax_revenue + k_tax_revenue
    print(f"  Tax collected from Q: {q_tax_revenue}, from K: {k_tax_revenue}. Total subsidy pool: {total_tax_revenue}")

    # --- 2. 将补贴分配给V，并处理溢出 ---
    v_k_list_initial = list(getattr(model_config, "dynamic_basis_v_proj"))
    v_groups = getattr(model_config, "v_groups")
    print(f"  Initial V k_list: {v_k_list_initial}")
    
    # 2a. 按分组大小，初步分配补贴
    temp_v_k_list = list(v_k_list_initial)
    if total_tax_revenue > 0:
        total_layers_in_v = sum(len(g) for g in v_groups)
        if total_layers_in_v > 0:
            subsidies = [int(round(total_tax_revenue * (len(g) / total_layers_in_v))) for g in v_groups]
            remainder = total_tax_revenue - sum(subsidies)
            temp_v_k_list = [k + s for k, s in zip(temp_v_k_list, subsidies)]
            if remainder > 0:
                for i in range(remainder): temp_v_k_list[i % len(temp_v_k_list)] += 1
    
    # 2b. 处理溢出
    spillover_pool = 0
    final_v_k_list = [0] * len(v_k_list_initial)
    for i in range(len(temp_v_k_list)):
        if temp_v_k_list[i] > max_rank:
            spillover_pool += (temp_v_k_list[i] - max_rank)
            final_v_k_list[i] = max_rank
        else:
            final_v_k_list[i] = temp_v_k_list[i]
    
    if spillover_pool > 0:
        for _ in range(spillover_pool):
            best_group_idx = -1
            min_k_value = float('inf')
            for i in range(len(final_v_k_list)):
                if final_v_k_list[i] < max_rank:
                    if final_v_k_list[i] < min_k_value:
                        min_k_value = final_v_k_list[i]
                        best_group_idx = i
            if best_group_idx != -1:
                final_v_k_list[best_group_idx] += 1
            else: break 
    
    # --- 3. 计算V没用完的“退款” ---
    v_subsidy_used = sum(final_v_k_list) - sum(v_k_list_initial)
    refund_amount = total_tax_revenue - v_subsidy_used
    print(f"  V module actually used {v_subsidy_used} k-values. Final V k_list: {final_v_k_list}")
    print(f"  Refund amount to be returned to Q & K: {refund_amount}")

    # --- 4. 将“退款”按比例、普惠地退还给Q和K ---
    if refund_amount > 0:
        if total_tax_revenue > 0:
            q_refund_share = int(round(refund_amount * (q_tax_revenue / total_tax_revenue)))
            k_refund_share = refund_amount - q_refund_share
        else: # 预防除零错误
            q_refund_share = refund_amount // 2
            k_refund_share = refund_amount - q_refund_share
        print(f"  Refunding {q_refund_share} to Q, and {k_refund_share} to K.")

        if q_refund_share > 0:
            q_groups = model_config.q_groups
            total_q_layers = sum(len(g) for g in q_groups)
            if total_q_layers > 0:
                q_refunds = [int(round(q_refund_share * (len(g) / total_q_layers))) for g in q_groups]
                remainder = q_refund_share - sum(q_refunds)
                if remainder > 0:
                    for i in range(remainder): new_q_k_list[i % len(new_q_k_list)] += 1
                for i in range(len(new_q_k_list)):
                    new_q_k_list[i] = min(max_rank, new_q_k_list[i] + q_refunds[i])

        if k_refund_share > 0:
            k_groups = model_config.k_groups
            total_k_layers = sum(len(g) for g in k_groups)
            if total_k_layers > 0:
                k_refunds = [int(round(k_refund_share * (len(g) / total_k_layers))) for g in k_groups]
                remainder = k_refund_share - sum(k_refunds)
                if remainder > 0:
                    for i in range(remainder): new_k_k_list[i % len(new_k_k_list)] += 1
                for i in range(len(new_k_k_list)):
                    new_k_k_list[i] = min(max_rank, new_k_k_list[i] + k_refunds[i])

    # --- 5. 更新最终的k_list ---
    setattr(model_config, "dynamic_basis_q_proj", new_q_k_list)
    setattr(model_config, "dynamic_basis_k_proj", new_k_k_list)
    setattr(model_config, "dynamic_basis_v_proj", final_v_k_list)

    print(f"  Final Q k_list after refund: {new_q_k_list}")
    print(f"  Final K k_list after refund: {new_k_k_list}")
    print("="*80 + "\n")
    
    return model_config
    



# --- 步骤1: 增加权重提取的辅助函数 ---
# (这部分逻辑借鉴自 group.py，以避免循环导入)
def _get_llama2_weights(std_model, group_member, name):
    w = []
    # 定义一个统一的目标设备
    target_device = 'cuda:0' 
    model = std_model.model.layers
    for layer in group_member:
        data = model[layer].get_submodule(name).weight.data
        # 在添加进列表前，将张量移动到目标设备
        w.append(data.T.to(target_device))
    return torch.cat(w, dim=-1)

def _get_gpt2_weights(std_model, group_member, name):
    w = []
    model = std_model.transformer.h
    for layer in group_member:
        data = model[layer].get_submodule(name).weight.data
        w.append(data)
    return torch.cat(w, dim=-1)

def _get_opt_weights(std_model, group_member, name):
    w = []
    model = std_model.model.decoder.layers
    for layer in group_member:
        data = model[layer].get_submodule(name).weight.data
        w.append(data.T)
    return torch.cat(w, dim=-1)

def _get_mistral_weights(std_model, group_member, name):
    w = []
    model = std_model.model.layers
    for layer in group_member:
        data = model[layer].get_submodule(name).weight.data
        w.append(data.T)
    return torch.cat(w, dim=-1)



def get_sensitivity_and_singular_values(std_model, model_type, group, name, calib_path):
    """
    Computes the weighted matrix SW, then returns its singular value spectrum 
    and the sensitivity (effective rank) calculated from it.
    """
    s, _ = Calib.get_s_inv_s(group, name, model_type, calib_path)
    
    if model_type == 'llama2':
        w_cat = _get_llama2_weights(std_model, group, name)
    else:
        raise NotImplementedError

    s = s.to(w_cat.device, dtype=torch.float32)
    w_cat = w_cat.float()
    sw = s @ w_cat
    
    # Use svdvals to efficiently compute only the singular values
    singular_values = torch.linalg.svdvals(sw)

    sig = singular_values.pow(2)

    # Calculate effective rank from the singular values
    p = sig / (sig.sum() + 1e-12)
    reff = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))
    
    return reff, singular_values


def do_update_model(config, model, dataset, tokenizer, data_collator):
    if os.path.exists(config.updated_model_path):
        print("Start load model!")
        print("Load: {}".format(config.updated_model_path))
        if config.model_type == "gpt2":
            model = ShareGPT2LMHeadModel.from_pretrained(config.updated_model_path, device_map='auto')
        elif config.model_type == "llama2":
            model = ShareLlamaForCausalLM.from_pretrained(config.updated_model_path, device_map='auto')
        elif config.model_type == "opt":
            model = ShareOPTForCausalLM.from_pretrained(config.updated_model_path, device_map='auto')
        elif config.model_type == "mistral":
            model = ShareMistralForCausalLM.from_pretrained(config.updated_model_path, device_map='auto')
        else:
            raise ValueError
    else:
        std_model = AutoModelForCausalLM.from_pretrained(config.model_name, cache_dir="llm_weights", device_map="cpu")
        std_model.config.use_cache = False
        model = load_checkpoint_and_dispatch(model, config.untrained_model_path, device_map="auto")

        # Prepare Dataloader for calibration data
        torch.manual_seed(2023)
        index = torch.randperm(len(dataset))
        index = index[:config.calibration_size]
        subset = Subset(dataset, index)
        dataloader = DataLoader(subset, batch_size=config.calib_batch_size, shuffle=False, collate_fn=data_collator,
                                pin_memory=True, num_workers=4)

        if config.build_update_calib:
            print("Start build update calib!")
            names = config.share_part + config.private_part
            basis_name = []
            for name in names:
                if name == "q" or name == "v" or name == "gate":
                    continue
                basis_name.append(name + "_basis")

            Calib.build_update_dataset(model, dataloader, basis_name, config.model_type, config.update_calib_path)

        model_config = model.config
        short_model_name = ShareConfig.name_map[config.model_name]

        names = config.share_part + config.private_part
        for name in names:
            print("Update {}".format(name))
            model = update_model(std_model=std_model,
                                 model=model,
                                 model_type=config.model_type,
                                 groups=getattr(model_config, name + "_groups"),
                                 name=getattr(config, name + "_name"),
                                 step=
                                 ShareConfig.weight_info[short_model_name][getattr(config, name + "_name")][
                                     1],
                                 num_basis=getattr(model_config, "num_basis_" + name),
                                 basis_name=name + "_basis",
                                 calib_path=config.update_calib_path,
                                 )
        if config.save_updated_model:
            model.save_pretrained(config.updated_model_path, safe_serialization=False)
            tokenizer.save_pretrained(config.updated_model_path)
    return model










#########for basis sharing

# def create_model(config):
#     if os.path.exists(config.untrained_model_path):
#         model_path = config.untrained_model_path
#         print("Start load model!")
#         print("Start load: {}".format(config.untrained_model_path))
#         if config.model_type == "gpt2":
#             model = ShareGPT2LMHeadModel.from_pretrained(model_path, device_map='auto', )
#         elif config.model_type == "llama2":
#             if "30b" in config.untrained_model_path:
#                 model = ShareLlamaForCausalLM.from_pretrained(model_path, device_map='auto',
#                                                               torch_dtype=torch.float16)
#             else:
#                 model = ShareLlamaForCausalLM.from_pretrained(model_path, device_map='cpu')
#         elif config.model_type == "opt":
#             model = ShareOPTForCausalLM.from_pretrained(model_path, device_map='auto')
#         elif config.model_type == "mistral":
#             model = ShareMistralForCausalLM.from_pretrained(model_path, device_map='auto')
#         else:
#             raise ValueError

#     else:
#         if config.model_type == "llama2":
#             tokenizer = LlamaTokenizer.from_pretrained(config.model_name)
#         else:
#             tokenizer = AutoTokenizer.from_pretrained(config.model_name)
#         #tokenizer.pad_token = "[PAD]"
#         # 对于Llama系列模型，最佳实践是将pad_token设置为eos_token
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#         # --- 修改结束 ---
#         print("Start create model!")
#         model_config = AutoConfig.from_pretrained(config.model_name)
#         model_config.use_cache = False
#         if config.model_name == "jeffwan/llama-30b-hf":
#             std_model = AutoModelForCausalLM.from_pretrained(config.model_name, cache_dir="llm_weights", device_map="auto",
#                                                              torch_dtype=torch.float16)
#         else:
#             std_model = AutoModelForCausalLM.from_pretrained(config.model_name, cache_dir="llm_weights", device_map="auto")

        
#         original_params = sum(p.numel() for p in std_model.parameters() if p.requires_grad)
#         print("\n" + "="*50)
#         print(f"Original Model Trainable Parameters: {original_params / 1e9:.3f}B")
#         print("="*50 + "\n")

#         if config.build_calib:
#             train_dataset, val_dataset, tokenized_test, data_collator = prepare_data(config.dataset_name, tokenizer,
#                                                                                      config.context_length, config.dataset_cache_dir)
#             # Prepare Dataloader for calibration data
#             torch.manual_seed(2023)
#             index = torch.randperm(len(train_dataset))
#             index = index[:config.calibration_size]
#             subset = Subset(train_dataset, index)
#             dataloader = DataLoader(subset, batch_size=config.calib_batch_size, shuffle=False, collate_fn=data_collator,
#                                     pin_memory=True, num_workers=4)

#             print("Start create calib!")
#             calib_names = []
#             if hasattr(config, "k_name"):
#                 # calibration data for k, q, v is the same
#                 calib_names.append(config.k_name)
#             if hasattr(config, "attn_name"):
#                 calib_names.append(config.attn_name)
#             calib_names.append(config.o_name)
#             calib_names.append(config.up_name)
#             calib_names.append(config.down_name)
#             Calib.build_calibration_dataset(std_model, dataloader, calib_names, config.model_type, config.calib_path)
#             print("Calib build done!")

#         short_model_name = ShareConfig.name_map[config.model_name]

#         # Share Part
#         names = config.share_part
#         for name in names:
#             print("Config for {}".format(name))
#             nx, nf = ShareConfig.weight_info[short_model_name][getattr(config, name + "_name")]
#             num_group = model_config.num_hidden_layers // config.group_size
#             rest = model_config.num_hidden_layers % config.group_size
#             gs = config.group_size
#             group = [[gs * i + j for j in range(config.group_size)] for i in range(num_group)]
#             if rest != 0:
#                 group += [[num_group * config.group_size + i for i in range(rest)]]
#             setattr(model_config, name + "_groups", group)
#             num_basis = compute_num_basis(nx, nf, config.group_size, config.compression_ratio)
#             setattr(model_config, "num_basis_" + name, num_basis)
#             print("num_basis {}".format(num_basis))

#         # Private Part
#         names = config.private_part
#         for name in names:
#             print("Config for {}".format(name))
#             setattr(model_config, name + "_groups", [[i] for i in range(model_config.num_hidden_layers)])
#             nx, nf = ShareConfig.weight_info[short_model_name][getattr(config, name + "_name")]
#             num_basis = compute_num_basis(nx, nf, 1, config.compression_ratio)
#             setattr(model_config, "num_basis_" + name, num_basis)
#             print("num_basis {}".format(num_basis))

#         if config.model_type == "llama2":
#             if "30b" in config.model_name:
#                 model_config.torch_dtype = torch.float16
#             model = ShareLlamaForCausalLM(model_config)
#         elif config.model_type == "gpt2":
#             model = ShareGPT2LMHeadModel(model_config)
#         elif config.model_type == "opt":
#             model = ShareOPTForCausalLM(model_config)
#         elif config.model_type == "mistral":
#             model = ShareMistralForCausalLM(model_config)
#         else:
#             raise NotImplementedError

#         print("Model init finished!")
#         if not hasattr(config, "tfs"):
#             matched_state_dict, _ = match_state_dict(model.state_dict(), std_model.state_dict())
#             model.load_state_dict(matched_state_dict, strict=False)

#             # Share Part
#             names = config.share_part + config.private_part
#             for name in names:
#                 print("Change {}".format(name))
#                 model = change_model(std_model=std_model,
#                                      model=model,
#                                      model_type=config.model_type,
#                                      groups=getattr(model_config, name + "_groups"),
#                                      name=getattr(config, name + "_name"),
#                                      step=ShareConfig.weight_info[short_model_name][getattr(config, name + "_name")][1],
#                                      num_basis=getattr(model_config, "num_basis_" + name),
#                                      basis_name=name + "_basis",
#                                      calib_path=config.calib_path,
#                                      )
            

#             compressed_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#             print("\n" + "="*60)
#             print("--- Final Parameter Count Verification ---")
#             print(f"Original Model Parameters:   {original_params / 1e9:.3f}B")
#             print(f"Compressed Model Parameters: {compressed_params / 1e9:.3f}B")
#             print(f"Achieved Final Size:         {compressed_params / original_params * 100:.2f}% of original")
#             print(f"Total Reduction:             {(1 - compressed_params / original_params) * 100:.2f}%")
#             print("="*60 + "\n")
#             if config.save_untrained_model:
#                 model.save_pretrained(config.untrained_model_path, safe_serialization=False)
#                 tokenizer.save_pretrained(config.untrained_model_path)

#     return model











##### for d rank


def create_model(config):
    if os.path.exists(config.untrained_model_path):
        model_path = config.untrained_model_path

        ###需要加！！！
        model_config = AutoConfig.from_pretrained(model_path)
        
        # 2. 根据config中的信息，重建空的basis模块骨架
        k_basis = build_dynamic_basis_collection(model_config.k_groups, model_config.dynamic_basis_k_proj, model_config.hidden_size)
        q_basis = build_dynamic_basis_collection(model_config.q_groups, model_config.dynamic_basis_q_proj, model_config.hidden_size)
        v_basis = build_dynamic_basis_collection(model_config.v_groups, model_config.dynamic_basis_v_proj, model_config.hidden_size)
        o_basis = build_dynamic_basis_collection(model_config.o_groups, model_config.dynamic_basis_o_proj, model_config.hidden_size)
        up_basis = build_dynamic_basis_collection(model_config.up_groups, model_config.dynamic_basis_up_proj, model_config.hidden_size)
        gate_basis = build_dynamic_basis_collection(model_config.gate_groups, model_config.dynamic_basis_gate_proj, model_config.hidden_size)
        down_basis = build_dynamic_basis_collection(model_config.down_groups, model_config.dynamic_basis_down_proj, model_config.intermediate_size)


        torch.manual_seed(2023)
        print("Start load model!")
        print("Start load: {}".format(config.untrained_model_path))
        if config.model_type == "gpt2":
            model = ShareGPT2LMHeadModel.from_pretrained(model_path, device_map='auto', )
        elif config.model_type == "llama2":
            if "30b" in config.untrained_model_path:
                #model = ShareLlamaForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16)
                model = ShareLlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                config=model_config,
                k_basis=k_basis,
                q_basis=q_basis,
                v_basis=v_basis,
                o_basis=o_basis,
                up_basis=up_basis,
                gate_basis=gate_basis,
                down_basis=down_basis,
                device_map='cpu'
            )
                main_dtype = torch.bfloat16  # 或 torch.bfloat16
                model.to(dtype=main_dtype)

            else:
                #model = ShareLlamaForCausalLM.from_pretrained(model_path, device_map='cpu')
                model = ShareLlamaForCausalLM.from_pretrained(
                model_path,
                config=model_config,
                k_basis=k_basis,
                q_basis=q_basis,
                v_basis=v_basis,
                o_basis=o_basis,
                up_basis=up_basis,
                gate_basis=gate_basis,
                down_basis=down_basis,
                device_map='cpu'
            )
      
        elif config.model_type == "opt":
            model = ShareOPTForCausalLM.from_pretrained(model_path, device_map='auto')
        elif config.model_type == "mistral":
            model = ShareMistralForCausalLM.from_pretrained(
                model_path,
                config=model_config,
                k_basis=k_basis,
                q_basis=q_basis,
                v_basis=v_basis,
                o_basis=o_basis,
                up_basis=up_basis,
                gate_basis=gate_basis,
                down_basis=down_basis,
                device_map='cpu'
            )
        else:
            raise ValueError
    
        
        print("Successfully loaded pre-compressed model.")
    else:
        if config.model_type == "llama2":
            tokenizer = LlamaTokenizer.from_pretrained(config.model_name, cache_dir="llm_weights")
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir="llm_weights")
        tokenizer.pad_token = "[PAD]"

        #         # 对于Llama系列模型，最佳实践是将pad_token设置为eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token


        print("Start create model!")
        model_config = AutoConfig.from_pretrained(config.model_name, cache_dir="llm_weights")
        model_config.use_cache = False
        if "30b" in config.model_name:
            std_model = AutoModelForCausalLM.from_pretrained(config.model_name, cache_dir="llm_weights", device_map="auto", torch_dtype=torch.float16)
        else:
            std_model = AutoModelForCausalLM.from_pretrained(config.model_name, cache_dir="llm_weights", device_map="auto")







        

        if config.build_calib:
            train_dataset, val_dataset, tokenized_test, data_collator = prepare_data(config.dataset_name, tokenizer, config.context_length, config.dataset_cache_dir)
            torch.manual_seed(2023)
            index = torch.randperm(len(train_dataset))[:config.calibration_size]
            subset = Subset(train_dataset, index)
            dataloader = DataLoader(subset, batch_size=config.calib_batch_size, shuffle=False, collate_fn=data_collator, pin_memory=True, num_workers=4)
            print("Start create calib!")
            calib_names = [config.o_name, config.up_name, config.down_name]
            if hasattr(config, "k_name"):
                calib_names.append(config.k_name)
            if hasattr(config, "attn_name"):
                calib_names.append(config.attn_name)
            Calib.build_calibration_dataset(std_model, dataloader, calib_names, config.model_type, config.calib_path)
            print("Calib build done!")

        short_model_name = ShareConfig.name_map[config.model_name]

        all_singular_values = {}
        all_s_lists = {}


        ####针对llama3 and mistral:

        # all_initial_k_lists = {}
        # all_w_lists = {}
        # all_s_lists = {}
        # all_P_budgets = {}
        # all_base_groups = {}
        # max_ranks = {}

        # # --- 步骤 1: 首先，为所有模块计算出初始的动态k值 ---
        # for part in ["share_part", "private_part"]:
        #     names = getattr(config, part)
            
        #     for name in names:
        #         print(f"Calculating initial k_list for {name}...")
                
        #         # 获取分组策略
        #         strategy = getattr(config, "grouping_strategy", {}).get(name, "default")
        #         if isinstance(strategy, list):
        #             base_groups = strategy
        #         else: 
        #             gs = config.group_size if part == "share_part" else 1
        #             num_group = model_config.num_hidden_layers // gs
        #             rest = model_config.num_hidden_layers % gs
        #             base_groups = [[gs * i + j for j in range(gs)] for i in range(num_group)]
        #             if rest: base_groups.append([num_group * gs + i for i in range(rest)])

        #         nx, nf = ShareConfig.weight_info[short_model_name][getattr(config, name + "_name")]
        #         max_ranks[name] = min(nx, nf)
        #         w_list = [nx + len(g) * nf for g in base_groups]
                
        #         results = [get_sensitivity_and_singular_values(std_model, config.model_type, g, getattr(config, name + "_name"), config.calib_path) for g in base_groups]
        #         s_list = [res[0] for res in results]
        #         print(s_list)
        #         singular_values_list = [res[1] for res in results]
                
        #         P_orig = sum(nx * len(g) * nf for g in base_groups)
        #         P_budget = int(round(P_orig * (1 - config.compression_ratio / 100)))
                
        #         denom = sum(math.sqrt(si * wi) for si, wi in zip(s_list, w_list)) or 1e-12
        #         k_cont = [math.sqrt(si / wi) * (P_budget / denom) for si, wi in zip(s_list, w_list)]
        #         max_rank = min(nx, nf)

        #         dynamic_k_list = [max(1, min(max_rank, int(round(ci)))) for ci in k_cont]

        #         static_k = compute_num_basis(nx, nf, gs, config.compression_ratio)
        #         print(f"  Static k baseline: {static_k}")
        #         print(f"  Initial dynamic k_list proposal: {dynamic_k_list}")




        #         total_error_dynamic = 0
        #         total_error_static = 0
        #         for i, singular_values in enumerate(singular_values_list):
        #             k_dyn = dynamic_k_list[i]
        #             error_dyn = torch.sum(singular_values[k_dyn:]**2)
        #             total_error_dynamic += error_dyn.item()
                    
        #             k_stat = static_k
        #             error_stat = torch.sum(singular_values[k_stat:]**2)
        #             total_error_static += error_stat.item()

        #         print(f"  Total Dynamic Reconstruction Error (Squared F-Norm): {total_error_dynamic:.4f}")
        #         print(f"  Total Static Reconstruction Error (Squared F-Norm):  {total_error_static:.4f}")

        #         if total_error_dynamic < total_error_static and 'gate' not in name and 'k' not in name and 'q' not in name:
        #             print("  Decision: Dynamic k_list has lower reconstruction error. Adopting.")
        #             k_list = dynamic_k_list


        #             # protection_floor = int(static_k * 0.99)
        #             # print(f"  Protection floor: {protection_floor}")

        #             #     # 2. 抬高低于下限的k值，并计算总参数“赤字” (params_deficit)
        #             # params_deficit = 0
        #             # adjusted_k_list = list(k_list) # 创建一个副本
        #             # for i, k in enumerate(adjusted_k_list):
        #             #     if k < protection_floor:
        #             #         params_deficit += (protection_floor - k) * w_list[i]
        #             #         adjusted_k_list[i] = protection_floor

        #             # print(f"  Initial parameter deficit after raising floors: {params_deficit}")

        #             #     # 3. 从高于静态k值的“富裕”层中，按比例扣除参数赤字
        #             # if params_deficit > 0:
        #             #     donors = [(i, k) for i, k in enumerate(adjusted_k_list) if k > static_k]
                            
        #             #         # 计算总的“参数盈余”
        #             #     total_params_surplus = sum((k - static_k) * w_list[i] for i, k in donors)

        #             #     if total_params_surplus > 0:
        #             #         print(f"  Found {len(donors)} donor(s) with a total parameter surplus of {total_params_surplus} to cover the deficit.")
                                
        #             #             # 按比例扣除
        #             #         for i, k_donor in donors:
        #             #             params_surplus_i = (k_donor - static_k) * w_list[i]
        #             #             params_to_deduct = params_deficit * (params_surplus_i / total_params_surplus)
                                    
        #             #                 # 将要扣除的参数量转换回k值的减少量
        #             #             k_deduction = params_to_deduct / w_list[i]
                                    
        #             #                 # 扣除，并确保不低于静态k值
        #             #             adjusted_k_list[i] = max(static_k, int(round(k_donor - k_deduction)))
        #             #     else:
        #             #         print("  Warning: No donors with surplus found to cover the deficit. Final compression ratio might be lower.")
                        
        #             # k_list = adjusted_k_list # 使用调整后的k_list
        #             # print(f"  k_list after reallocation: {k_list}")
        #         else:
        #             if 'gate' in name or 'up' in name or 'down' in name or 'o' in name or 'v' in name or 'k' in name or 'q' in name:
        #                 print("  Decision: Static k has lower or equal error. Falling back to static k.")
        #                 k_list = [static_k] * len(base_groups)
        #             else:
        #                 k_list = dynamic_k_list
        #             # print("  Decision: Static k has lower or equal error. Falling back to static k.")
        #             # k_list = [static_k] * len(base_groups)
                
        #         all_initial_k_lists[name] = k_list
        #         all_w_lists[name] = w_list
        #         all_s_lists[name] = s_list
        #         all_P_budgets[name] = P_budget
        #         setattr(model_config, name + "_groups", base_groups)
        
        # # ================================================================= #
        # # ==========  步骤 2: 执行您提出的、非对称的跨模块预算重分配  ==========
        # # ================================================================= #
        # print("\n" + "="*50)
        # all_initial_k_lists = reallocate_llama3(model_config, all_w_lists, all_initial_k_lists, max_ranks, tax_rate=0.1, min_k=1)
        # #all_initial_k_lists = reallocate_llama3_2(model_config, all_w_lists, all_initial_k_lists, max_ranks, min_k=1)
        # # ================================================================= #
        
        # # --- 步骤 3: 为每个模块进行最终的预算微调，以严格保证压缩率 ---
        # for part in ["share_part", "private_part"]:
        #     names = getattr(config, part)
        #     for name in names:
        #         print(f"Finalizing k_list for {name}")
        #         k_list = all_initial_k_lists[name] # 使用经过重分配的k_list作为起点
        #         w_list = all_w_lists[name]
        #         s_list = all_s_lists[name]
        #         P_budget = all_P_budgets[name]
                
        #         def total_params(ks): return sum(ki * wi for ki, wi in zip(ks, w_list))
        #         # 使用while循环来确保预算被精确匹配
        #         for _ in range(len(k_list) * 2):
        #         #while total_params(k_list) != P_budget:
        #             diff = total_params(k_list) - P_budget
        #             if diff == 0: break
                    
        #             if diff > 0:
        #                 candidates = [i for i, k in enumerate(k_list) if k > 1]
        #                 if not candidates: break
        #                 idx_to_reduce = min(candidates, key=lambda i: s_list[i] / (w_list[i] + 1e-9))
        #                 k_list[idx_to_reduce] -= 1
        #             else:
        #                 max_rank_for_group = min(ShareConfig.weight_info[short_model_name][getattr(config, name + "_name")])
        #                 candidates = [i for i, k in enumerate(k_list) if k < max_rank_for_group]
        #                 if not candidates: break
        #                 idx_to_increase = max(candidates, key=lambda i: s_list[i] / (w_list[i] + 1e-9))
        #                 k_list[idx_to_increase] += 1
                
        #         setattr(model_config, f"dynamic_basis_{name}_proj", k_list)
        #         print(f"  Final selected k_list for {name}: {k_list}")


        #####针对llama1-2

        

        for part in ["share_part", "private_part"]:
            names = getattr(config, part)
            
            # --- ここで修正 ---
            if part == "share_part":
                gs = config.group_size
                num_group = model_config.num_hidden_layers // gs
                rest = model_config.num_hidden_layers % gs
                base_groups = [[gs * i + j for j in range(gs)] for i in range(num_group)]
                if rest: base_groups.append([num_group * gs + i for i in range(rest)])

            else: # private_part
                gs = 1
                base_groups = [[i] for i in range(model_config.num_hidden_layers)]

                
            # --- 修正完了 ---

            for name in names:
                print(f"Config for {name} (Adaptive k selection)")
                nx, nf = ShareConfig.weight_info[short_model_name][getattr(config, name + "_name")]
                w_list = [nx + len(g) * nf for g in base_groups]

                results = [get_sensitivity_and_singular_values(std_model, config.model_type, g, getattr(config, name + "_name"), config.calib_path) for g in base_groups]
                #results = [calculate_saliency_and_svd(std_model, config.model_type, g, getattr(config, name + "_name"), config.calib_path) for g in base_groups]
                
                #results = [get_sensitivity_and_singular_values(std_model, config.model_type, g, getattr(config, name + "_name"), config.calib_path) for g in base_groups]
            #     results = [
            #     calculate_theoretical_loss(
            #         std_model, 
            #         config.model_type, 
            #         g, 
            #         getattr(config, name + "_name"), 
            #         config.calib_path,
            #         config.compression_ratio, # 传入全局压缩率
            #         nx, nf, len(g) # 传入计算基准k值所需的参数
            #     ) for g in base_groups
            # ]
                # ----------------------------------------------
                s_list = [res[0] for res in results]
                print(s_list)
                #print(f"  Group Saliency Scores (Total Energy): {s_list}")
                singular_values_list = [res[1] for res in results]

                all_singular_values[name] = singular_values_list
                all_s_lists[name] = s_list


                # # === 新增：按域读取层熵，并换算每组的熵权重 m_g ===
                # # 建议把路径放进 config，比如 config.entropy_csv_path 指向 Wiki 或 C4 的 csv
                # entropy_csv_path = "/eagle/lc-mpi/Zhendong/Basis_Sharing/entropy_token_lp.csv"
                # if os.path.exists(entropy_csv_path):
                #     H_layer = _load_layer_entropy_mean(entropy_csv_path)
                #     # 两层一组：对组内层做均值后得到 m_g
                #     m_entropy = _entropy_weights_for_groups(
                #         groups=base_groups,
                #         H_layer=H_layer,
                #         gamma=0.3,          # 可在 config 里调
                #         tail_protect_last_n=0,
                #         last_boost=1.10
                #     )
                # else:
                #     # 没提供 csv 就退化为全 1
                #     m_entropy = np.ones(len(base_groups), dtype=np.float64)

                
                # #s_list = [s**0 * m for s, m in zip(s_list, m_entropy)]
                P_orig = sum(nx * len(g) * nf for g in base_groups)
                P_budget = int(round(P_orig * (1 - config.compression_ratio / 100)))
                
                denom = sum(math.sqrt(si * wi) for si, wi in zip(s_list, w_list)) or 1e-12
                k_cont = [math.sqrt(si / wi) * (P_budget / denom) for si, wi in zip(s_list, w_list)]
                # num = [math.sqrt( (s_i * float(m_i)) / (w_i + 1e-12) ) for s_i, m_i, w_i in zip(s_list, m_entropy, w_list)]
                # den = sum(num) or 1e-12
                # k_cont = [ (P_budget / den) * x for x in num ]


                max_rank = min(nx, nf)
                print(max_rank)
                dynamic_k_list = [max(1, min(max_rank, int(round(ci)))) for ci in k_cont]


                # 步骤A: 计算每个组的“可压缩性”分数 S_i = 1 / Log(L_min)
                # 我们的 s_list 已经是 L_min^2，为简化和稳定，我们直接使用 s_list 的倒数作为可压缩性分数
                # （s_list越高 -> 越重要 -> 可压缩性越低）
                # compressibility_scores = [score for score in s_list]

                # # 步骤B: 计算分数总和
                # total_compressibility = sum(compressibility_scores)

                # # 步骤C: 按“可压缩性”比例，分配总的目标参数预算 P_target_module
                # P_orig = sum(nx * len(g) * nf for g in base_groups)
                # P_budget = P_orig * (1 - config.compression_ratio / 100)

                # k_list = []
                # if total_compressibility > 0:
                #     for i in range(len(base_groups)):
                #         # 1. 计算这个组应该分到多少参数预算
                #         proportion = compressibility_scores[i] / total_compressibility
                #         target_params_for_group = P_budget * proportion
                        
                #         # 2. 将参数预算反算回k值 (参数 = k * 单个秩的成本)
                #         cost_per_rank = w_list[i]
                #         k_val = target_params_for_group / cost_per_rank
                        
                #         k_list.append(max(1, min(max_rank, int(round(k_val)))))
                # else:
                #     # 预防性代码：如果所有分数都为0，则按参数量均匀分配k值
                #     total_w = sum(w_list)
                #     k_list = [max(1, min(max_rank, int(round(P_budget * (w / total_w) / w)))) for w in w_list]

                # dynamic_k_list = k_list

                static_k = compute_num_basis(nx, nf, gs, config.compression_ratio)
                print(f"  Static k baseline: {static_k}")
                print(f"  Initial dynamic k_list proposal: {dynamic_k_list}")




                total_error_dynamic = 0
                total_error_static = 0
                for i, singular_values in enumerate(singular_values_list):
                    k_dyn = dynamic_k_list[i]
                    error_dyn = torch.sum(singular_values[k_dyn:]**2)
                    total_error_dynamic += error_dyn.item()
                    
                    k_stat = static_k
                    error_stat = torch.sum(singular_values[k_stat:]**2)
                    total_error_static += error_stat.item()

                print(f"  Total Dynamic Reconstruction Error (Squared F-Norm): {total_error_dynamic:.4f}")
                print(f"  Total Static Reconstruction Error (Squared F-Norm):  {total_error_static:.4f}")

                if total_error_dynamic < total_error_static and 'gate' not in name:
                    print("  Decision: Dynamic k_list has lower reconstruction error. Adopting.")
                    k_list = dynamic_k_list


                    # protection_floor = int(static_k * 0.99)
                    # print(f"  Protection floor: {protection_floor}")

                    #     # 2. 抬高低于下限的k值，并计算总参数“赤字” (params_deficit)
                    # params_deficit = 0
                    # adjusted_k_list = list(k_list) # 创建一个副本
                    # for i, k in enumerate(adjusted_k_list):
                    #     if k < protection_floor:
                    #         params_deficit += (protection_floor - k) * w_list[i]
                    #         adjusted_k_list[i] = protection_floor

                    # print(f"  Initial parameter deficit after raising floors: {params_deficit}")

                    #     # 3. 从高于静态k值的“富裕”层中，按比例扣除参数赤字
                    # if params_deficit > 0:
                    #     donors = [(i, k) for i, k in enumerate(adjusted_k_list) if k > static_k]
                            
                    #         # 计算总的“参数盈余”
                    #     total_params_surplus = sum((k - static_k) * w_list[i] for i, k in donors)

                    #     if total_params_surplus > 0:
                    #         print(f"  Found {len(donors)} donor(s) with a total parameter surplus of {total_params_surplus} to cover the deficit.")
                                
                    #             # 按比例扣除
                    #         for i, k_donor in donors:
                    #             params_surplus_i = (k_donor - static_k) * w_list[i]
                    #             params_to_deduct = params_deficit * (params_surplus_i / total_params_surplus)
                                    
                    #                 # 将要扣除的参数量转换回k值的减少量
                    #             k_deduction = params_to_deduct / w_list[i]
                                    
                    #                 # 扣除，并确保不低于静态k值
                    #             adjusted_k_list[i] = max(static_k, int(round(k_donor - k_deduction)))
                    #     else:
                    #         print("  Warning: No donors with surplus found to cover the deficit. Final compression ratio might be lower.")
                        
                    # k_list = adjusted_k_list # 使用调整后的k_list
                    # print(f"  k_list after reallocation: {k_list}")
                else:
                    if 'gate' in name or 'up' in name or 'down' in name or 'o' in name:
                        print("  Decision: Static k has lower or equal error. Falling back to static k.")
                        k_list = [static_k] * len(base_groups)
                    else:
                        k_list = dynamic_k_list
                    # print("  Decision: Static k has lower or equal error. Falling back to static k.")
                    # k_list = [static_k] * len(base_groups)
                
                def total_params(ks): return sum(ki * wi for ki, wi in zip(ks, w_list))
                for _ in range(len(k_list) * 2):
                    diff = total_params(k_list) - P_budget
                    if diff == 0: break
                    if diff > 0:
                        candidates = [i for i, k in enumerate(k_list) if k > 1]
                        if not candidates: break
                        idx_to_reduce = min(candidates, key=lambda i: s_list[i] / (w_list[i] + 1e-9))
                        k_list[idx_to_reduce] -= 1
                    else:
                        # --- 【核心修正】: 只在那些k值还没有达到最大秩的组里选择 ---
                        candidates = [i for i, k in enumerate(k_list) if k < max_rank]
                        if not candidates: 
                            break # 如果所有组都已满，无法再增加，则退出
                        idx_to_increase = max(range(len(k_list)), key=lambda i: s_list[i] / (w_list[i] + 1e-9))
                        k_list[idx_to_increase] += 1
                
                setattr(model_config, name + "_groups", base_groups)
                setattr(model_config, f"dynamic_basis_{name}_proj", k_list)
                print(f"  Final selected k_list for {name}: {k_list}")

        

        model_config = reallocate_k_budget(model_config, tax_rate=0.2, min_k=1, max_rank=4096) #taxrate can be adjusted
  





        # 创建 basis 模块
        k_basis = build_dynamic_basis_collection(model_config.k_groups, model_config.dynamic_basis_k_proj, model_config.hidden_size)
        q_basis = build_dynamic_basis_collection(model_config.q_groups, model_config.dynamic_basis_q_proj, model_config.hidden_size)
        v_basis = build_dynamic_basis_collection(model_config.v_groups, model_config.dynamic_basis_v_proj, model_config.hidden_size)
        o_basis = build_dynamic_basis_collection(model_config.o_groups, model_config.dynamic_basis_o_proj, model_config.hidden_size)
        up_basis = build_dynamic_basis_collection(model_config.up_groups, model_config.dynamic_basis_up_proj, model_config.hidden_size)
        gate_basis = build_dynamic_basis_collection(model_config.gate_groups, model_config.dynamic_basis_gate_proj, model_config.hidden_size)
        down_basis = build_dynamic_basis_collection(model_config.down_groups, model_config.dynamic_basis_down_proj, model_config.intermediate_size)

        # 实例化模型
        if config.model_type == "llama2":
            model = ShareLlamaForCausalLM(model_config, k_basis, q_basis, v_basis, o_basis, up_basis, gate_basis, down_basis)
        elif config.model_type == "gpt2":
            model = ShareGPT2LMHeadModel(model_config) # Note: You'll need to adapt GPT2 model to accept basis modules
        elif config.model_type == "opt":
            model = ShareOPTForCausalLM(model_config) # Note: You'll need to adapt OPT model to accept basis modules
        elif config.model_type == "mistral":
            model = ShareMistralForCausalLM(model_config) # Note: You'll need to adapt Mistral model to accept basis modules
        else:
            raise NotImplementedError

        print("Model init finished!")
        if not hasattr(config, "tfs"):
            matched_state_dict, _ = match_state_dict(model.state_dict(), std_model.state_dict())
            model.load_state_dict(matched_state_dict, strict=False)

            # 初始化 Share Part 权重
            for name in config.share_part:
                groups = getattr(model_config, name + "_groups")
                ks = getattr(model_config, f"dynamic_basis_{name}_proj")
                for idx, group in enumerate(groups):
                    ki = ks[idx]
                    print(f"Change {name}, group {idx}, k = {ki}")
                    model = change_model(std_model, model, config.model_type, [group], getattr(config, name + "_name"), ShareConfig.weight_info[short_model_name][getattr(config, name + "_name")][1], ki, name + "_basis", config.calib_path)

            # 初始化 Private Part 权重
            for name in config.private_part:
                groups = getattr(model_config, name + "_groups")
                ks = getattr(model_config, f"dynamic_basis_{name}_proj")
                for idx, group in enumerate(groups):
                    ki = ks[idx]
                    print(f"Change {name}, group {idx}, k = {ki}")
                    model = change_model(std_model, model, config.model_type, [group], getattr(config, name + "_name"), ShareConfig.weight_info[short_model_name][getattr(config, name + "_name")][1], ki, name + "_basis", config.calib_path)



            print("Releasing std_model from memory...")
            del std_model
            torch.cuda.empty_cache()




            
            if config.save_untrained_model:
                model.save_pretrained(config.untrained_model_path, safe_serialization=False)
                tokenizer.save_pretrained(config.untrained_model_path)
                print(config.untrained_model_path)




    return model


