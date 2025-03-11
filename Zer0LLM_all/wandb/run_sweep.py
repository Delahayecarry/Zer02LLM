#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Weights & Biases超参数搜索启动脚本
用于启动和运行wandb sweep进行超参数优化
"""

import os
import sys
import argparse
import subprocess
import yaml
import wandb

def main():
    parser = argparse.ArgumentParser(description="启动Weights & Biases超参数搜索")
    parser.add_argument("--config", type=str, default="sweep_config.yaml", 
                        help="超参数搜索配置文件路径")
    parser.add_argument("--count", type=int, default=10, 
                        help="要运行的超参数组合数量")
    parser.add_argument("--project", type=str, default="Zer02LLM_Sweep", 
                        help="Wandb项目名称")
    parser.add_argument("--entity", type=str, default=None, 
                        help="Wandb实体名称（用户名或组织名）")
    parser.add_argument("--mode", type=str, choices=["pretrain", "sft", "dpo"], default=None,
                        help="训练模式，如果指定则覆盖配置文件中的设置")
    parser.add_argument("--gpu", type=str, default=None,
                        help="指定使用的GPU，例如 '0,1'")
    parser.add_argument("--wandb_host", type=str, default=None,
                        help="Wandb主机地址，用于私有化部署")
    parser.add_argument("--wandb_base_url", type=str, default=None,
                        help="Wandb基础URL，用于私有化部署")
    args = parser.parse_args()
    
    # 读取配置文件
    with open(args.config, "r") as f:
        sweep_config = yaml.safe_load(f)
    
    # 如果命令行指定了mode，则覆盖配置文件中的设置
    if args.mode:
        sweep_config["parameters"]["mode"]["value"] = args.mode
        
    # 如果指定了项目名称，则覆盖配置文件中的设置
    if args.project:
        sweep_config["parameters"]["wandb_project"]["value"] = args.project
    
    # 设置环境变量
    env = os.environ.copy()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # 设置wandb私有化部署配置
    if args.wandb_host:
        os.environ["WANDB_HOST"] = args.wandb_host
        os.environ["WANDB_BASE_URL"] = args.wandb_base_url if args.wandb_base_url else f"http://{args.wandb_host}"
    
    # 初始化wandb
    wandb.login()
    
    # 创建sweep
    sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
    print(f"创建了新的sweep: {sweep_id}")
    
    # 运行sweep agent
    print(f"启动{args.count}次运行...")
    
    # 构建agent命令
    cmd = [
        "python", "-m", "wandb", "agent",
        f"{args.entity or wandb.api.default_entity}/{args.project}/{sweep_id}",
        "--count", str(args.count)
    ]
    
    # 运行agent
    subprocess.call(cmd, env=env)
    
    print("超参数搜索完成！")
    print(f"请访问 https://wandb.ai/{args.entity or wandb.api.default_entity}/{args.project}/sweeps/{sweep_id} 查看结果")

if __name__ == "__main__":
    main() 