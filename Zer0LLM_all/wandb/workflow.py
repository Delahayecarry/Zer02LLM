#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Zer02LLM Wandb工作流脚本
用于管理整个模型训练过程，包括超参数搜索、模型训练、结果分析和可视化
"""

import os
import sys
import argparse
import subprocess
import yaml
import json
import time
import datetime
import wandb
from typing import Dict, List, Optional, Any, Tuple, Union
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 工作流阶段
class WorkflowStage:
    SETUP = "setup"
    SWEEP = "sweep"
    TRAIN = "train"
    EVALUATE = "evaluate"
    ANALYZE = "analyze"
    DEPLOY = "deploy"

class Zer02LLMWorkflow:
    """Zer02LLM Wandb工作流管理器"""
    
    def __init__(self, args):
        self.args = args
        self.project_name = args.project
        self.entity = args.entity
        self.mode = args.mode
        self.config_path = args.config
        self.output_dir = args.output_dir
        self.sweep_id = None
        self.best_run_id = None
        self.best_model_path = None
        
        # 私有化wandb服务器配置
        self.wandb_host = args.wandb_host
        self.wandb_base_url = args.wandb_base_url
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化wandb
        if args.stage != WorkflowStage.SETUP:
            self._setup_wandb()
    
    def _setup_wandb(self) -> None:
        """设置wandb配置"""
        print("=== 设置wandb配置 ===")
        
        # 设置wandb主机和基础URL
        wandb.login(host=self.wandb_host, base_url=self.wandb_base_url)
    
    def setup(self) -> None:
        """设置工作流环境和配置"""
        print("=== 设置工作流环境和配置 ===")
        
        # 创建配置目录
        config_dir = os.path.join(self.output_dir, "configs")
        os.makedirs(config_dir, exist_ok=True)
        
        # 根据训练模式选择基础配置模板
        if self.mode == "pretrain":
            template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_config.yaml")
        elif self.mode == "dpo":
            template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_config_dpo.yaml")
        else:  # sft
            template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_config.yaml")
            # 修改SFT特定参数
            with open(template_path, "r") as f:
                config = yaml.safe_load(f)
                config["parameters"]["mode"]["value"] = "sft"
                config["parameters"]["learning_rate"]["min"] = 1e-6
                config["parameters"]["learning_rate"]["max"] = 1e-4
        
        # 如果没有指定配置文件，则使用模板创建
        if not self.config_path or not os.path.exists(self.config_path):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config_path = os.path.join(config_dir, f"sweep_config_{self.mode}_{timestamp}.yaml")
            
            if self.mode in ["pretrain", "dpo"]:
                # 直接复制模板
                with open(template_path, "r") as src, open(self.config_path, "w") as dst:
                    dst.write(src.read())
            else:
                # 写入修改后的SFT配置
                with open(self.config_path, "w") as f:
                    yaml.dump(config, f)
        
        print(f"配置文件已保存到: {self.config_path}")
        
        # 创建工作流配置文件
        workflow_config = {
            "project": self.project_name,
            "entity": self.entity,
            "mode": self.mode,
            "sweep_config": self.config_path,
            "created_at": datetime.datetime.now().isoformat(),
        }
        
        workflow_config_path = os.path.join(self.output_dir, "workflow_config.json")
        with open(workflow_config_path, "w") as f:
            json.dump(workflow_config, f, indent=2)
        
        print(f"工作流配置已保存到: {workflow_config_path}")
    
    def run_sweep(self) -> str:
        """运行超参数搜索"""
        print("=== 运行超参数搜索 ===")
        
        # 读取配置文件
        with open(self.config_path, "r") as f:
            sweep_config = yaml.safe_load(f)
        
        # 确保配置中的项目名称与当前项目一致
        sweep_config["parameters"]["wandb_project"]["value"] = self.project_name
        
        # 确保配置中的训练模式与当前模式一致
        sweep_config["parameters"]["mode"]["value"] = self.mode
        
        # 创建sweep
        self.sweep_id = wandb.sweep(sweep_config, project=self.project_name, entity=self.entity)
        print(f"创建了新的sweep: {self.sweep_id}")
        
        # 保存sweep ID到工作流配置
        workflow_config_path = os.path.join(self.output_dir, "workflow_config.json")
        if os.path.exists(workflow_config_path):
            with open(workflow_config_path, "r") as f:
                workflow_config = json.load(f)
            
            workflow_config["sweep_id"] = self.sweep_id
            workflow_config["sweep_started_at"] = datetime.datetime.now().isoformat()
            
            with open(workflow_config_path, "w") as f:
                json.dump(workflow_config, f, indent=2)
        
        # 运行sweep agent
        count = self.args.sweep_count
        print(f"启动{count}次运行...")
        
        # 设置环境变量
        env = os.environ.copy()
        if self.args.gpu:
            env["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        
        # 为私有化wandb设置环境变量
        if self.wandb_host:
            env["WANDB_BASE_URL"] = self.wandb_base_url if self.wandb_base_url else f"http://{self.wandb_host}"
            env["WANDB_HOST"] = self.wandb_host
        
        # 使用subprocess运行sweep agent
        cmd = [
            "python", "-m", "wandb", "agent", 
            f"{self.entity}/{self.project_name}/{self.sweep_id}",
            "--count", str(count)
        ]
        
        try:
            subprocess.run(cmd, env=env, check=True)
            print("超参数搜索完成！")
        except subprocess.CalledProcessError as e:
            print(f"运行sweep agent时出错: {e}")
        
        return self.sweep_id
    
    def find_best_run(self, sweep_id: str) -> Dict[str, Any]:
        """找到超参数搜索中的最佳运行"""
        print("=== 查找最佳运行 ===")
        
        # 确保wandb API配置正确
        if self.wandb_host:
            os.environ["WANDB_BASE_URL"] = self.wandb_base_url if self.wandb_base_url else f"http://{self.wandb_host}"
            os.environ["WANDB_HOST"] = self.wandb_host
        
        api = wandb.Api()
        sweep = api.sweep(f"{self.entity}/{self.project_name}/{sweep_id}")
        
        best_run = None
        best_metric = float('inf')  # 假设我们在最小化指标
        metric_name = "train/loss"  # 默认指标
        
        # 从sweep配置中获取目标指标和目标（最小化/最大化）
        try:
            sweep_config = sweep.config
            metric_name = sweep_config.get("metric", {}).get("name", "train/loss")
            goal = sweep_config.get("metric", {}).get("goal", "minimize")
            minimize = (goal == "minimize")
        except:
            print("无法从sweep配置中获取指标信息，使用默认值")
            minimize = True
        
        print(f"查找指标 '{metric_name}' 的{'最小' if minimize else '最大'}值...")
        
        # 遍历所有运行
        for run in sweep.runs:
            if run.state == "finished":
                try:
                    metric_value = run.summary.get(metric_name)
                    if metric_value is not None:
                        if minimize and metric_value < best_metric:
                            best_metric = metric_value
                            best_run = run
                        elif not minimize and metric_value > best_metric:
                            best_metric = metric_value
                            best_run = run
                except:
                    continue
        
        if best_run is None:
            print("未找到有效的完成运行")
            return None
        
        self.best_run_id = best_run.id
        print(f"找到最佳运行: {best_run.name} (ID: {best_run.id})")
        print(f"最佳 {metric_name}: {best_metric}")
        
        # 获取最佳配置
        best_config = {k: v for k, v in best_run.config.items() if not k.startswith('_')}
        
        # 保存最佳配置
        best_config_path = os.path.join(self.output_dir, "best_config.yaml")
        with open(best_config_path, "w") as f:
            yaml.dump(best_config, f)
        
        print(f"最佳配置已保存到: {best_config_path}")
        
        # 更新工作流配置
        workflow_config_path = os.path.join(self.output_dir, "workflow_config.json")
        if os.path.exists(workflow_config_path):
            with open(workflow_config_path, "r") as f:
                workflow_config = json.load(f)
            
            workflow_config["best_run_id"] = best_run.id
            workflow_config["best_run_name"] = best_run.name
            workflow_config["best_metric"] = {metric_name: best_metric}
            
            with open(workflow_config_path, "w") as f:
                json.dump(workflow_config, f, indent=2)
        
        return best_config
    
    def train_with_best_config(self, best_config: Dict[str, Any]) -> str:
        """使用最佳配置进行完整训练"""
        print("=== 使用最佳配置进行完整训练 ===")
        
        # 准备命令行参数
        cmd = ["python", "../pretrain_sft_lora_rlhf.py"]
        
        # 添加配置参数
        for key, value in best_config.items():
            if key == "wandb_project":
                # 使用当前项目名称
                cmd.extend(["--wandb_project", self.project_name])
            elif isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
        
        # 添加wandb相关参数
        if "--use_wandb" not in cmd:
            cmd.append("--use_wandb")
        
        # 添加wandb运行名称
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"best_config_{self.mode}_{timestamp}"
        cmd.extend(["--wandb_run_name", run_name])
        
        # 添加私有化wandb服务器配置
        if self.wandb_host:
            cmd.extend(["--wandb_host", self.wandb_host])
            if self.wandb_base_url:
                cmd.extend(["--wandb_base_url", self.wandb_base_url])
        
        # 设置环境变量
        env = os.environ.copy()
        if self.args.gpu:
            env["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        
        # 为私有化wandb设置环境变量
        if self.wandb_host:
            env["WANDB_BASE_URL"] = self.wandb_base_url if self.wandb_base_url else f"http://{self.wandb_host}"
            env["WANDB_HOST"] = self.wandb_host
        
        # 运行训练
        print("启动训练...")
        print(f"命令: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, env=env, check=True)
            print("训练完成！")
            
            # 查找最佳模型路径
            out_dir = next((cmd[i+1] for i, arg in enumerate(cmd) if arg == "--out_dir"), "out")
            best_model_path = os.path.join(out_dir, "best.pt")
            if os.path.exists(best_model_path):
                self.best_model_path = best_model_path
                print(f"最佳模型保存在: {best_model_path}")
                
                # 更新工作流配置
                workflow_config_path = os.path.join(self.output_dir, "workflow_config.json")
                if os.path.exists(workflow_config_path):
                    with open(workflow_config_path, "r") as f:
                        workflow_config = json.load(f)
                    
                    workflow_config["best_model_path"] = best_model_path
                    workflow_config["training_completed_at"] = datetime.datetime.now().isoformat()
                    
                    with open(workflow_config_path, "w") as f:
                        json.dump(workflow_config, f, indent=2)
            
            return best_model_path
            
        except subprocess.CalledProcessError as e:
            print(f"训练过程中出错: {e}")
            return None
    
    def evaluate_model(self, model_path: str) -> Dict[str, float]:
        """评估训练好的模型"""
        print("=== 评估模型 ===")
        
        if not model_path or not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return None
        
        # 从最佳配置中获取模型参数
        best_config_path = os.path.join(self.output_dir, "best_config.yaml")
        if not os.path.exists(best_config_path):
            print(f"最佳配置文件不存在: {best_config_path}")
            return None
        
        with open(best_config_path, "r") as f:
            best_config = yaml.safe_load(f)
        
        # 准备评估命令
        cmd = ["python", "../eval_model.py"]
        
        # 添加模型参数
        model_mode = 0 if self.mode == "pretrain" else 1  # 0: pretrain, 1: sft/dpo
        cmd.extend(["--model_mode", str(model_mode)])
        cmd.extend(["--model_path", model_path])
        
        # 添加模型结构参数
        for param in ["dim", "n_layers", "n_heads", "max_seq_len"]:
            if param in best_config:
                cmd.extend([f"--{param}", str(best_config[param])])
        
        # 添加其他评估参数
        cmd.extend(["--stream", "True"])  # 启用流式输出
        
        # 设置环境变量
        env = os.environ.copy()
        if self.args.gpu:
            env["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        
        # 运行评估
        print("启动评估...")
        print(f"命令: {' '.join(cmd)}")
        
        try:
            # 创建评估结果目录
            eval_dir = os.path.join(self.output_dir, "evaluation")
            os.makedirs(eval_dir, exist_ok=True)
            
            # 运行评估并将输出保存到文件
            eval_output_path = os.path.join(eval_dir, f"eval_results_{self.mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(eval_output_path, "w") as f:
                subprocess.run(cmd, env=env, check=True, stdout=f, stderr=subprocess.STDOUT)
            
            print(f"评估完成！结果保存在: {eval_output_path}")
            
            # 解析评估结果（这里需要根据eval_model.py的输出格式进行调整）
            metrics = self._parse_eval_results(eval_output_path)
            
            # 更新工作流配置
            workflow_config_path = os.path.join(self.output_dir, "workflow_config.json")
            if os.path.exists(workflow_config_path):
                with open(workflow_config_path, "r") as f:
                    workflow_config = json.load(f)
                
                workflow_config["evaluation_results"] = metrics
                workflow_config["evaluation_completed_at"] = datetime.datetime.now().isoformat()
                
                with open(workflow_config_path, "w") as f:
                    json.dump(workflow_config, f, indent=2)
            
            return metrics
            
        except subprocess.CalledProcessError as e:
            print(f"评估过程中出错: {e}")
            return None
    
    def _parse_eval_results(self, eval_output_path: str) -> Dict[str, float]:
        """解析评估结果文件"""
        metrics = {}
        
        try:
            with open(eval_output_path, "r") as f:
                lines = f.readlines()
            
            # 这里需要根据eval_model.py的输出格式进行调整
            for line in lines:
                if ":" in line:
                    parts = line.split(":", 1)
                    key = parts[0].strip()
                    try:
                        value = float(parts[1].strip())
                        metrics[key] = value
                    except:
                        continue
        except:
            print(f"解析评估结果时出错")
        
        return metrics
    
    def analyze_results(self, sweep_id: str) -> None:
        """分析训练结果"""
        print("=== 分析训练结果 ===")
        
        if not sweep_id:
            print("未提供sweep ID，无法分析结果")
            return
        
        # 创建分析结果目录
        analysis_dir = os.path.join(self.output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 运行分析脚本
        cmd = [
            "python", "../wandb/analyze_sweep.py",
            "--sweep_id", sweep_id,
            "--project", self.project_name,
            "--output_dir", analysis_dir
        ]
        
        if self.entity:
            cmd.extend(["--entity", self.entity])
        
        # 添加私有化wandb服务器配置
        if self.wandb_host:
            cmd.extend(["--wandb_host", self.wandb_host])
            if self.wandb_base_url:
                cmd.extend(["--wandb_base_url", self.wandb_base_url])
        
        print("启动分析...")
        print(f"命令: {' '.join(cmd)}")
        
        # 设置环境变量
        env = os.environ.copy()
        if self.wandb_host:
            env["WANDB_BASE_URL"] = self.wandb_base_url if self.wandb_base_url else f"http://{self.wandb_host}"
            env["WANDB_HOST"] = self.wandb_host
        
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"分析完成！结果保存在: {analysis_dir}")
            
            # 更新工作流配置
            workflow_config_path = os.path.join(self.output_dir, "workflow_config.json")
            if os.path.exists(workflow_config_path):
                with open(workflow_config_path, "r") as f:
                    workflow_config = json.load(f)
                
                workflow_config["analysis_completed_at"] = datetime.datetime.now().isoformat()
                workflow_config["analysis_dir"] = analysis_dir
                
                with open(workflow_config_path, "w") as f:
                    json.dump(workflow_config, f, indent=2)
            
        except subprocess.CalledProcessError as e:
            print(f"分析过程中出错: {e}")
    
    def deploy_model(self, model_path: str) -> None:
        """部署模型（示例）"""
        print("=== 部署模型 ===")
        
        if not model_path or not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return
        
        # 创建部署目录
        deploy_dir = os.path.join(self.output_dir, "deploy")
        os.makedirs(deploy_dir, exist_ok=True)
        
        # 复制模型文件到部署目录
        deploy_model_path = os.path.join(deploy_dir, f"model_{self.mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        try:
            import shutil
            shutil.copy2(model_path, deploy_model_path)
            print(f"模型已复制到部署目录: {deploy_model_path}")
            
            # 更新工作流配置
            workflow_config_path = os.path.join(self.output_dir, "workflow_config.json")
            if os.path.exists(workflow_config_path):
                with open(workflow_config_path, "r") as f:
                    workflow_config = json.load(f)
                
                workflow_config["deployed_model_path"] = deploy_model_path
                workflow_config["deployment_completed_at"] = datetime.datetime.now().isoformat()
                
                with open(workflow_config_path, "w") as f:
                    json.dump(workflow_config, f, indent=2)
            
        except Exception as e:
            print(f"部署模型时出错: {e}")
    
    def run_workflow(self) -> None:
        """运行完整工作流"""
        # 根据指定的阶段运行工作流
        if self.args.stage == WorkflowStage.SETUP or self.args.all:
            self.setup()
        
        if self.args.stage == WorkflowStage.SWEEP or self.args.all:
            sweep_id = self.run_sweep()
            self.sweep_id = sweep_id
        
        if self.args.stage == WorkflowStage.TRAIN or self.args.all:
            # 如果没有运行超参数搜索，则从配置中加载sweep_id
            if not self.sweep_id and os.path.exists(os.path.join(self.output_dir, "workflow_config.json")):
                with open(os.path.join(self.output_dir, "workflow_config.json"), "r") as f:
                    workflow_config = json.load(f)
                    self.sweep_id = workflow_config.get("sweep_id")
            
            if not self.sweep_id:
                print("未找到sweep ID，无法继续训练阶段")
                return
            
            best_config = self.find_best_run(self.sweep_id)
            if best_config:
                self.best_model_path = self.train_with_best_config(best_config)
        
        if self.args.stage == WorkflowStage.EVALUATE or self.args.all:
            # 如果没有训练，则从配置中加载最佳模型路径
            if not self.best_model_path and os.path.exists(os.path.join(self.output_dir, "workflow_config.json")):
                with open(os.path.join(self.output_dir, "workflow_config.json"), "r") as f:
                    workflow_config = json.load(f)
                    self.best_model_path = workflow_config.get("best_model_path")
            
            if not self.best_model_path:
                print("未找到最佳模型路径，无法继续评估阶段")
                return
            
            self.evaluate_model(self.best_model_path)
        
        if self.args.stage == WorkflowStage.ANALYZE or self.args.all:
            # 如果没有运行超参数搜索，则从配置中加载sweep_id
            if not self.sweep_id and os.path.exists(os.path.join(self.output_dir, "workflow_config.json")):
                with open(os.path.join(self.output_dir, "workflow_config.json"), "r") as f:
                    workflow_config = json.load(f)
                    self.sweep_id = workflow_config.get("sweep_id")
            
            if not self.sweep_id:
                print("未找到sweep ID，无法继续分析阶段")
                return
            
            self.analyze_results(self.sweep_id)
        
        if self.args.stage == WorkflowStage.DEPLOY or self.args.all:
            # 如果没有训练，则从配置中加载最佳模型路径
            if not self.best_model_path and os.path.exists(os.path.join(self.output_dir, "workflow_config.json")):
                with open(os.path.join(self.output_dir, "workflow_config.json"), "r") as f:
                    workflow_config = json.load(f)
                    self.best_model_path = workflow_config.get("best_model_path")
            
            if not self.best_model_path:
                print("未找到最佳模型路径，无法继续部署阶段")
                return
            
            self.deploy_model(self.best_model_path)
        
        print("工作流完成！")

def main():
    parser = argparse.ArgumentParser(description="Zer02LLM Wandb工作流")
    parser.add_argument("--mode", type=str, choices=["pretrain", "sft", "dpo"], default="pretrain",
                        help="训练模式")
    parser.add_argument("--project", type=str, default="Zer02LLM_Workflow",
                        help="Wandb项目名称")
    parser.add_argument("--entity", type=str, default=None,
                        help="Wandb实体名称（用户名或组织名）")
    parser.add_argument("--config", type=str, default=None,
                        help="超参数搜索配置文件路径")
    parser.add_argument("--output_dir", type=str, default="workflow_output",
                        help="工作流输出目录")
    parser.add_argument("--sweep_count", type=int, default=5,
                        help="超参数搜索运行次数")
    parser.add_argument("--gpu", type=str, default=None,
                        help="指定使用的GPU，例如 '0,1'")
    parser.add_argument("--stage", type=str, 
                        choices=[WorkflowStage.SETUP, WorkflowStage.SWEEP, WorkflowStage.TRAIN, 
                                WorkflowStage.EVALUATE, WorkflowStage.ANALYZE, WorkflowStage.DEPLOY],
                        default=WorkflowStage.SETUP,
                        help="要运行的工作流阶段")
    parser.add_argument("--all", action="store_true",
                        help="运行所有工作流阶段")
    parser.add_argument("--wandb_host", type=str, default=None,
                        help="wandb主机地址")
    parser.add_argument("--wandb_base_url", type=str, default=None,
                        help="wandb基础URL")
    
    args = parser.parse_args()
    
    # 创建并运行工作流
    workflow = Zer02LLMWorkflow(args)
    workflow.run_workflow()

if __name__ == "__main__":
    main() 