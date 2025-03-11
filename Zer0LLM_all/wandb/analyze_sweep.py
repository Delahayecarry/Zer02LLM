#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Weights & Biases超参数搜索结果分析脚本
用于分析和可视化超参数搜索的结果
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from typing import List, Dict, Any, Optional

def fetch_sweep_runs(sweep_id: str, project: str, entity: Optional[str] = None, 
                    wandb_host: Optional[str] = None, wandb_base_url: Optional[str] = None) -> pd.DataFrame:
    """获取指定sweep的所有运行数据"""
    # 设置wandb API配置
    if wandb_host:
        os.environ["WANDB_HOST"] = wandb_host
        os.environ["WANDB_BASE_URL"] = wandb_base_url if wandb_base_url else f"http://{wandb_host}"
    
    api = wandb.Api()
    sweep = api.sweep(f"{entity or wandb.api.default_entity}/{project}/{sweep_id}")
    
    runs = []
    for run in sweep.runs:
        if run.state == "finished" or run.state == "crashed" or run.state == "failed":
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            summary = {k: v for k, v in run.summary.items() if not k.startswith('_')}
            
            # 合并配置和摘要
            run_data = {**config, **summary}
            run_data['run_id'] = run.id
            run_data['run_name'] = run.name
            run_data['state'] = run.state
            runs.append(run_data)
    
    return pd.DataFrame(runs)

def plot_parameter_importance(df: pd.DataFrame, target_metric: str = 'train/loss', 
                             top_n: int = 10, output_file: Optional[str] = None) -> None:
    """绘制超参数重要性图"""
    # 过滤掉非数值列和目标指标
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_metric in numeric_cols:
        numeric_cols.remove(target_metric)
    
    # 计算每个参数与目标指标的相关性
    correlations = []
    for col in numeric_cols:
        if df[col].nunique() > 1:  # 只考虑有多个不同值的参数
            corr = df[col].corr(df[target_metric])
            if not np.isnan(corr):
                correlations.append((col, abs(corr)))
    
    # 按相关性绝对值排序
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # 取前N个参数
    top_params = correlations[:min(top_n, len(correlations))]
    
    # 绘制条形图
    plt.figure(figsize=(10, 6))
    params = [p[0] for p in top_params]
    corrs = [p[1] for p in top_params]
    
    plt.barh(params, corrs, color='skyblue')
    plt.xlabel('相关性绝对值')
    plt.ylabel('超参数')
    plt.title(f'超参数与{target_metric}的相关性')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
    plt.show()

def plot_parallel_coordinates(df: pd.DataFrame, target_metric: str = 'train/loss', 
                             top_n: int = 5, output_file: Optional[str] = None) -> None:
    """绘制平行坐标图"""
    # 过滤掉非数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 计算每个参数与目标指标的相关性
    correlations = []
    for col in numeric_cols:
        if col != target_metric and df[col].nunique() > 1:
            corr = df[col].corr(df[target_metric])
            if not np.isnan(corr):
                correlations.append((col, abs(corr)))
    
    # 按相关性绝对值排序
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # 取前N个参数
    top_params = [p[0] for p in correlations[:min(top_n, len(correlations))]]
    
    if target_metric not in top_params and target_metric in df.columns:
        plot_cols = top_params + [target_metric]
    else:
        plot_cols = top_params
    
    if len(plot_cols) < 2:
        print("没有足够的数值参数进行平行坐标图绘制")
        return
    
    # 绘制平行坐标图
    plt.figure(figsize=(12, 6))
    pd.plotting.parallel_coordinates(
        df, class_column=target_metric, cols=top_params, 
        colormap=plt.cm.coolwarm, alpha=0.5
    )
    plt.title(f'超参数平行坐标图 (按{target_metric}着色)')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
    plt.show()

def print_best_configs(df: pd.DataFrame, target_metric: str = 'train/loss', 
                      top_n: int = 3, minimize: bool = True) -> None:
    """打印最佳超参数配置"""
    if target_metric not in df.columns:
        print(f"目标指标 {target_metric} 不在数据中")
        return
    
    # 按目标指标排序
    if minimize:
        sorted_df = df.sort_values(by=target_metric)
    else:
        sorted_df = df.sort_values(by=target_metric, ascending=False)
    
    # 打印前N个配置
    print(f"\n最佳{top_n}个超参数配置 (按{target_metric} {'最小化' if minimize else '最大化'}):")
    for i, (_, row) in enumerate(sorted_df.head(top_n).iterrows()):
        print(f"\n第{i+1}名配置:")
        print(f"  运行ID: {row.get('run_id', 'N/A')}")
        print(f"  运行名称: {row.get('run_name', 'N/A')}")
        print(f"  {target_metric}: {row.get(target_metric, 'N/A')}")
        
        # 打印主要超参数
        for param in sorted([col for col in row.index if not col.startswith(('_', 'run_', 'state')) and col != target_metric]):
            print(f"  {param}: {row[param]}")

def main():
    parser = argparse.ArgumentParser(description="分析Weights & Biases超参数搜索结果")
    parser.add_argument("--sweep_id", type=str, required=True, 
                        help="Sweep ID")
    parser.add_argument("--project", type=str, required=True, 
                        help="Wandb项目名称")
    parser.add_argument("--entity", type=str, default=None, 
                        help="Wandb实体名称（用户名或组织名）")
    parser.add_argument("--metric", type=str, default="train/loss", 
                        help="要分析的目标指标")
    parser.add_argument("--minimize", action="store_true", default=True,
                        help="是否最小化目标指标（默认为True）")
    parser.add_argument("--top_n", type=int, default=5,
                        help="显示前N个最佳配置和参数")
    parser.add_argument("--output_dir", type=str, default="sweep_analysis",
                        help="输出目录")
    parser.add_argument("--wandb_host", type=str, default=None,
                        help="Wandb主机地址")
    parser.add_argument("--wandb_base_url", type=str, default=None,
                        help="Wandb基础URL")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取sweep运行数据
    print(f"正在获取sweep {args.sweep_id} 的运行数据...")
    df = fetch_sweep_runs(args.sweep_id, args.project, args.entity, args.wandb_host, args.wandb_base_url)
    
    if df.empty:
        print("没有找到任何运行数据")
        return
    
    print(f"找到 {len(df)} 个运行")
    
    # 打印最佳配置
    print_best_configs(df, args.metric, args.top_n, args.minimize)
    
    # 绘制参数重要性图
    print("\n绘制参数重要性图...")
    plot_parameter_importance(
        df, args.metric, args.top_n, 
        os.path.join(args.output_dir, f"parameter_importance_{args.sweep_id}.png")
    )
    
    # 绘制平行坐标图
    print("\n绘制平行坐标图...")
    try:
        plot_parallel_coordinates(
            df, args.metric, args.top_n,
            os.path.join(args.output_dir, f"parallel_coordinates_{args.sweep_id}.png")
        )
    except Exception as e:
        print(f"绘制平行坐标图时出错: {e}")
    
    print(f"\n分析完成！图表已保存到 {args.output_dir} 目录")

if __name__ == "__main__":
    main() 