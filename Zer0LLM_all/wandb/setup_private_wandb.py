#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Zer02LLM 私有化wandb部署配置脚本
用于设置和测试私有化部署的wandb服务器连接
"""

import os
import sys
import argparse
import subprocess
import json
import yaml
import time
import wandb
from pathlib import Path

def setup_wandb_env(host, base_url=None, insecure=False):
    """设置wandb环境变量"""
    os.environ["WANDB_HOST"] = host
    os.environ["WANDB_BASE_URL"] = base_url if base_url else f"http://{host}"
    
    if insecure:
        os.environ["WANDB_INSECURE"] = "true"
    
    print(f"已设置wandb环境变量:")
    print(f"  WANDB_HOST = {os.environ['WANDB_HOST']}")
    print(f"  WANDB_BASE_URL = {os.environ['WANDB_BASE_URL']}")
    if insecure:
        print(f"  WANDB_INSECURE = {os.environ['WANDB_INSECURE']}")

def test_wandb_connection(api_key=None):
    """测试wandb连接"""
    print("\n正在测试wandb连接...")
    
    try:
        if api_key:
            wandb.login(key=api_key)
        else:
            wandb.login()
        
        api = wandb.Api()
        print(f"连接成功! 当前用户: {api.viewer()['entity']}")
        return True
    except Exception as e:
        print(f"连接失败: {e}")
        return False

def create_config_file(host, base_url=None, insecure=False, api_key=None):
    """创建wandb配置文件"""
    config_dir = Path.home() / ".config" / "wandb"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = config_dir / "settings"
    
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        except:
            pass
    
    # 更新配置
    config["base_url"] = base_url if base_url else f"http://{host}"
    config["host"] = host
    
    if insecure:
        config["insecure"] = True
    
    if api_key:
        config["api_key"] = api_key
    
    # 保存配置
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"\nwandb配置已保存到: {config_path}")

def create_env_file(host, base_url=None, insecure=False, api_key=None):
    """创建环境变量设置脚本"""
    # 创建bash脚本
    bash_script = "#!/bin/bash\n\n"
    bash_script += f"export WANDB_HOST={host}\n"
    bash_script += f"export WANDB_BASE_URL={base_url if base_url else 'http://' + host}\n"
    
    if insecure:
        bash_script += "export WANDB_INSECURE=true\n"
    
    if api_key:
        bash_script += f"export WANDB_API_KEY={api_key}\n"
    
    with open("wandb_env.sh", "w") as f:
        f.write(bash_script)
    
    # 创建PowerShell脚本
    ps_script = ""
    ps_script += f"$env:WANDB_HOST = '{host}'\n"
    ps_script += f"$env:WANDB_BASE_URL = '{base_url if base_url else 'http://' + host}'\n"
    
    if insecure:
        ps_script += "$env:WANDB_INSECURE = 'true'\n"
    
    if api_key:
        ps_script += f"$env:WANDB_API_KEY = '{api_key}'\n"
    
    with open("wandb_env.ps1", "w") as f:
        f.write(ps_script)
    
    print("\n环境变量设置脚本已创建:")
    print("  - wandb_env.sh (Linux/macOS)")
    print("  - wandb_env.ps1 (Windows)")
    print("\n使用方法:")
    print("  Linux/macOS: source wandb_env.sh")
    print("  Windows: . .\\wandb_env.ps1")

def update_workflow_config(host, base_url=None, insecure=False):
    """更新工作流配置示例"""
    examples = {
        "workflow.py": f"python workflow.py --mode pretrain --project \"Zer02LLM_Workflow\" --all --wandb_host \"{host}\" --wandb_base_url \"{base_url if base_url else 'http://' + host}\"",
        "run_sweep.py": f"python run_sweep.py --config sweep_config.yaml --count 5 --wandb_host \"{host}\" --wandb_base_url \"{base_url if base_url else 'http://' + host}\"",
        "analyze_sweep.py": f"python analyze_sweep.py --sweep_id \"your-sweep-id\" --project \"Zer02LLM_Workflow\" --wandb_host \"{host}\" --wandb_base_url \"{base_url if base_url else 'http://' + host}\""
    }
    
    print("\n工作流命令示例:")
    for script, cmd in examples.items():
        print(f"\n{script}:")
        print(f"  {cmd}")

def main():
    parser = argparse.ArgumentParser(description="Zer02LLM 私有化wandb部署配置")
    parser.add_argument("--host", type=str, required=True,
                        help="wandb主机地址")
    parser.add_argument("--base_url", type=str, default=None,
                        help="wandb基础URL (默认为 http://<host>)")
    parser.add_argument("--insecure", action="store_true",
                        help="允许不安全的连接 (自签名证书)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="wandb API密钥")
    parser.add_argument("--test", action="store_true",
                        help="测试连接")
    parser.add_argument("--create_config", action="store_true",
                        help="创建wandb配置文件")
    parser.add_argument("--create_env", action="store_true",
                        help="创建环境变量设置脚本")
    parser.add_argument("--all", action="store_true",
                        help="执行所有操作")
    
    args = parser.parse_args()
    
    # 设置环境变量
    setup_wandb_env(args.host, args.base_url, args.insecure)
    
    # 测试连接
    if args.test or args.all:
        test_wandb_connection(args.api_key)
    
    # 创建配置文件
    if args.create_config or args.all:
        create_config_file(args.host, args.base_url, args.insecure, args.api_key)
    
    # 创建环境变量设置脚本
    if args.create_env or args.all:
        create_env_file(args.host, args.base_url, args.insecure, args.api_key)
    
    # 更新工作流配置示例
    update_workflow_config(args.host, args.base_url, args.insecure)

if __name__ == "__main__":
    main() 