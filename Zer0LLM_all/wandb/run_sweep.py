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
    parser.add_argument("--sweep_id", type=str, default=None,
                        help="使用已存在的sweep ID，而不是创建新的sweep")
    parser.add_argument("--agent_format", type=str, default=None, choices=["full", "id_only"],
                        help="agent命令的格式: full=entity/project/sweep_id, id_only=sweep_id")
    parser.add_argument("--create_only", action="store_true",
                        help="只创建sweep，不运行agent")
    args = parser.parse_args()
    
    # 读取配置文件
    with open(args.config, "r", encoding="utf-8") as f:
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
        print(f"使用私有化wandb服务器: {os.environ['WANDB_HOST']}")
        print(f"Wandb基础URL: {os.environ['WANDB_BASE_URL']}")
    
    # 初始化wandb
    try:
        wandb.login()
        print(f"成功登录wandb，用户: {wandb.api.viewer()['entity']}")
    except Exception as e:
        print(f"警告: wandb登录失败: {e}")
        print("尝试继续执行...")
    
    # 确保有正确的实体名称
    entity = args.entity
    if not entity:
        try:
            entity = wandb.api.viewer()['entity']
            print(f"使用当前登录用户作为实体名称: {entity}")
        except:
            print("警告: 无法获取当前用户实体名称")
    
    # 使用已存在的sweep ID或创建新的sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"使用已存在的sweep ID: {sweep_id}")
    else:
        # 创建sweep
        try:
            print(f"创建sweep，项目: {args.project}, 实体: {entity}")
            sweep_id = wandb.sweep(sweep_config, project=args.project, entity=entity)
            print(f"创建了新的sweep: {sweep_id}")
        except Exception as e:
            print(f"创建sweep失败: {e}")
            sys.exit(1)
    
    # 运行sweep agent
    print(f"启动{args.count}次运行...")
    
    # 构建agent命令
    cmd = [
        "python", "-m", "wandb", "agent"
    ]
    
    # 根据agent_format参数或环境决定使用哪种格式
    agent_format = args.agent_format
    if not agent_format:
        # 如果未指定，则根据是否使用私有化部署来决定
        agent_format = "id_only" if args.wandb_host else "full"
    
    if agent_format == "id_only":
        # 仅使用sweep_id
        print("使用仅ID格式的agent命令")
        cmd.append(f"{sweep_id}")
    else:
        # 使用完整路径
        print("使用完整路径格式的agent命令")
        cmd.append(f"{entity}/{args.project}/{sweep_id}")
    
    # 添加count参数
    cmd.extend(["--count", str(args.count)])
    
    # 显示完整命令
    print(f"运行agent命令: {' '.join(cmd)}")
    
    # 运行agent
    try:
        subprocess.call(cmd, env=env)
    except Exception as e:
        print(f"运行agent失败: {e}")
        sys.exit(1)
    
    print("超参数搜索完成！")
    
    # 使用正确的wandb_base_url构建结果URL
    base_url = args.wandb_base_url if args.wandb_base_url else "https://wandb.ai"
    # 移除URL末尾的斜杠（如果有）
    base_url = base_url.rstrip('/')
    
    # 确保有正确的实体名称用于URL
    url_entity = entity or "user"
    
    print(f"请访问 {base_url}/{url_entity}/{args.project}/sweeps/{sweep_id} 查看结果")

if __name__ == "__main__":
    main() 