#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Zer02LLM 工作流启动脚本
简化工作流的使用，提供交互式配置和一键启动功能
"""

import os
import sys
import argparse
import json
import yaml
import subprocess
import time
from pathlib import Path
import inquirer
from colorama import init, Fore, Style

# 初始化colorama
init()

def print_header(text):
    """打印带颜色的标题"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * len(text)}{Style.RESET_ALL}")

def print_success(text):
    """打印成功信息"""
    print(f"{Fore.GREEN}{text}{Style.RESET_ALL}")

def print_warning(text):
    """打印警告信息"""
    print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")

def print_error(text):
    """打印错误信息"""
    print(f"{Fore.RED}{text}{Style.RESET_ALL}")

def check_dependencies():
    """检查依赖项"""
    try:
        import inquirer
        import colorama
        import wandb
        return True
    except ImportError as e:
        print_error(f"缺少依赖项: {e}")
        print("请安装所需依赖: pip install inquirer colorama wandb")
        return False

def check_wandb_login():
    """检查wandb登录状态"""
    try:
        import wandb
        api = wandb.Api()
        return True, api.viewer()['entity']
    except Exception as e:
        return False, str(e)

def interactive_config():
    """交互式配置工作流"""
    print_header("Zer02LLM 工作流配置")
    
    # 训练模式选择
    mode_choices = [
        ('预训练 (Pretrain)', 'pretrain'),
        ('监督微调 (SFT)', 'sft'),
        ('DPO训练', 'dpo'),
        ('PPO训练', 'ppo')
    ]
    
    mode_question = [
        inquirer.List('mode',
                      message="选择训练模式",
                      choices=[choice[0] for choice in mode_choices],
                      default=mode_choices[0][0])
    ]
    mode_answer = inquirer.prompt(mode_question)
    selected_mode = next(choice[1] for choice in mode_choices if choice[0] == mode_answer['mode'])
    
    # 项目名称
    project_question = [
        inquirer.Text('project',
                     message="输入Wandb项目名称",
                     default=f"Zer02LLM_{selected_mode.upper()}")
    ]
    project_answer = inquirer.prompt(project_question)
    project_name = project_answer['project']
    
    # GPU选择
    gpu_question = [
        inquirer.Text('gpu',
                     message="指定使用的GPU (例如: 0,1,2,3，留空使用所有可用GPU)",
                     default="")
    ]
    gpu_answer = inquirer.prompt(gpu_question)
    gpu = gpu_answer['gpu']
    
    # 超参数搜索次数
    sweep_count_question = [
        inquirer.Text('sweep_count',
                     message="设置超参数搜索次数",
                     default="5",
                     validate=lambda _, x: x.isdigit())
    ]
    sweep_count_answer = inquirer.prompt(sweep_count_question)
    sweep_count = int(sweep_count_answer['sweep_count'])
    
    # 输出目录
    output_dir_question = [
        inquirer.Text('output_dir',
                     message="设置输出目录",
                     default=f"workflow_output_{selected_mode}")
    ]
    output_dir_answer = inquirer.prompt(output_dir_question)
    output_dir = output_dir_answer['output_dir']
    
    # 私有化wandb配置
    use_private_wandb_question = [
        inquirer.Confirm('use_private_wandb',
                        message="是否使用私有化部署的wandb服务器?",
                        default=False)
    ]
    use_private_wandb_answer = inquirer.prompt(use_private_wandb_question)
    use_private_wandb = use_private_wandb_answer['use_private_wandb']
    
    wandb_host = None
    wandb_base_url = None
    
    if use_private_wandb:
        wandb_host_question = [
            inquirer.Text('wandb_host',
                         message="输入wandb主机地址",
                         default="localhost:8080")
        ]
        wandb_host_answer = inquirer.prompt(wandb_host_question)
        wandb_host = wandb_host_answer['wandb_host']
        
        wandb_base_url_question = [
            inquirer.Text('wandb_base_url',
                         message="输入wandb基础URL",
                         default=f"http://{wandb_host}")
        ]
        wandb_base_url_answer = inquirer.prompt(wandb_base_url_question)
        wandb_base_url = wandb_base_url_answer['wandb_base_url']
    
    # 运行阶段选择
    stage_choices = [
        ('设置 (Setup)', 'setup'),
        ('超参数搜索 (Sweep)', 'sweep'),
        ('训练 (Train)', 'train'),
        ('评估 (Evaluate)', 'evaluate'),
        ('分析 (Analyze)', 'analyze'),
        ('部署 (Deploy)', 'deploy'),
        ('全部阶段 (All)', 'all')
    ]
    
    stage_question = [
        inquirer.List('stage',
                      message="选择要运行的工作流阶段",
                      choices=[choice[0] for choice in stage_choices],
                      default=stage_choices[-1][0])
    ]
    stage_answer = inquirer.prompt(stage_question)
    selected_stage = next(choice[1] for choice in stage_choices if choice[0] == stage_answer['stage'])
    
    # 确认配置
    print_header("配置摘要")
    print(f"训练模式: {Fore.YELLOW}{selected_mode}{Style.RESET_ALL}")
    print(f"项目名称: {Fore.YELLOW}{project_name}{Style.RESET_ALL}")
    print(f"GPU设置: {Fore.YELLOW}{gpu if gpu else '所有可用GPU'}{Style.RESET_ALL}")
    print(f"超参数搜索次数: {Fore.YELLOW}{sweep_count}{Style.RESET_ALL}")
    print(f"输出目录: {Fore.YELLOW}{output_dir}{Style.RESET_ALL}")
    if use_private_wandb:
        print(f"Wandb主机: {Fore.YELLOW}{wandb_host}{Style.RESET_ALL}")
        print(f"Wandb基础URL: {Fore.YELLOW}{wandb_base_url}{Style.RESET_ALL}")
    print(f"运行阶段: {Fore.YELLOW}{selected_stage}{Style.RESET_ALL}")
    
    confirm_question = [
        inquirer.Confirm('confirm',
                        message="确认以上配置并开始运行?",
                        default=True)
    ]
    confirm_answer = inquirer.prompt(confirm_question)
    
    if not confirm_answer['confirm']:
        print_warning("已取消运行")
        return None
    
    # 构建配置
    config = {
        'mode': selected_mode,
        'project': project_name,
        'gpu': gpu,
        'sweep_count': sweep_count,
        'output_dir': output_dir,
        'stage': selected_stage,
        'wandb_host': wandb_host,
        'wandb_base_url': wandb_base_url
    }
    
    return config

def build_command(config):
    """构建工作流命令"""
    cmd = ["python", "workflow.py"]
    
    # 添加基本参数
    cmd.extend(["--mode", config['mode']])
    cmd.extend(["--project", config['project']])
    
    if config['gpu']:
        cmd.extend(["--gpu", config['gpu']])
    
    if config['output_dir']:
        cmd.extend(["--output_dir", config['output_dir']])
    
    if config['sweep_count']:
        cmd.extend(["--sweep_count", str(config['sweep_count'])])
    
    # 添加私有化wandb参数
    if config.get('wandb_host'):
        cmd.extend(["--wandb_host", config['wandb_host']])
    
    if config.get('wandb_base_url'):
        cmd.extend(["--wandb_base_url", config['wandb_base_url']])
    
    # 添加阶段参数
    if config['stage'] == 'all':
        cmd.append("--all")
    else:
        cmd.extend(["--stage", config['stage']])
    
    return cmd

def run_workflow(cmd):
    """运行工作流"""
    print_header("启动工作流")
    print(f"执行命令: {Fore.YELLOW}{' '.join(cmd)}{Style.RESET_ALL}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时输出日志
        for line in iter(process.stdout.readline, ''):
            if "error" in line.lower() or "exception" in line.lower():
                print_error(line.strip())
            elif "warning" in line.lower():
                print_warning(line.strip())
            elif "success" in line.lower() or "completed" in line.lower():
                print_success(line.strip())
            else:
                print(line.strip())
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code == 0:
            print_success("\n工作流执行成功!")
        else:
            print_error(f"\n工作流执行失败，返回代码: {return_code}")
        
        return return_code
    except Exception as e:
        print_error(f"执行工作流时出错: {e}")
        return 1

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Zer02LLM 工作流启动脚本")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--interactive", action="store_true", help="交互式配置")
    args = parser.parse_args()
    
    # 检查依赖项
    if not check_dependencies():
        return 1
    
    # 检查wandb登录状态
    is_logged_in, entity = check_wandb_login()
    if is_logged_in:
        print_success(f"已登录wandb，用户: {entity}")
    else:
        print_warning(f"未登录wandb: {entity}")
        print("请先运行 'wandb login' 登录wandb")
        return 1
    
    # 获取配置
    config = None
    
    if args.config:
        # 从文件加载配置
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            print_success(f"已从 {args.config} 加载配置")
        except Exception as e:
            print_error(f"加载配置文件失败: {e}")
            return 1
    elif args.interactive or True:  # 默认使用交互式配置
        # 交互式配置
        config = interactive_config()
        if not config:
            return 0
        
        # 保存配置
        save_config_question = [
            inquirer.Confirm('save_config',
                            message="是否保存此配置以便将来使用?",
                            default=True)
        ]
        save_config_answer = inquirer.prompt(save_config_question)
        
        if save_config_answer['save_config']:
            config_dir = Path("saved_configs")
            config_dir.mkdir(exist_ok=True)
            
            config_name_question = [
                inquirer.Text('config_name',
                             message="输入配置名称",
                             default=f"{config['mode']}_{int(time.time())}")
            ]
            config_name_answer = inquirer.prompt(config_name_question)
            config_name = config_name_answer['config_name']
            
            config_path = config_dir / f"{config_name}.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print_success(f"配置已保存到 {config_path}")
    
    if not config:
        print_error("未提供配置")
        return 1
    
    # 构建命令
    cmd = build_command(config)
    
    # 运行工作流
    return run_workflow(cmd)

if __name__ == "__main__":
    sys.exit(main()) 