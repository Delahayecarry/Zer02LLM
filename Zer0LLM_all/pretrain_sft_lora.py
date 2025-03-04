import os
import argparse
import time
import math
import warnings
import unittest
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

from model.model import LLM
from model.LLMconfig import LLMconfig
from datasets import PretrainDataset, SFTDataset

warnings.filterwarnings('ignore')

class TrainingMode(Enum):
    PRETRAIN = "pretrain"
    SFT = "sft"
    TEST = "test"

@dataclass
class TrainingConfig:
    """统一的训练配置类"""
    mode: TrainingMode
    # 基础训练参数
    out_dir: str = "out"
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 5e-4
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    
    # 日志和监控
    use_wandb: bool = False
    wandb_project: str = "Zer02LLM"
    log_interval: int = 100
    save_interval: int = 100
    
    # 分布式训练
    num_workers: int = 1
    ddp: bool = False
    local_rank: int = -1
    
    # 优化器参数
    accumulation_steps: int = 8
    grad_clip: float = 1.0
    warmup_iters: int = 0
    
    # 模型参数 (从LLMconfig同步)
    dim: int = 512
    n_layers: int = 8
    max_seq_len: int = 512
    vocab_size: int = 4000
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    rope_theta: float = 10000.0
    dropout: float = 0.1
    norm_eps: float = 1e-8
    multiple_of: int = 64
    flash_attn: bool = False
    
    # MoE相关参数
    use_moe: bool = False
    n_routed_experts: int = 8
    num_experts_per_tok: int = 2
    n_shared_experts: Optional[int] = 0
    scoring_func: str = 'softmax'
    aux_loss_alpha: float = 0.1
    seq_aux: bool = True
    norm_topk_prob: bool = True
    
    # 数据路径
    data_path: str = "./dataset/pretrain_hq.jsonl"

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_environment()
        self.model, self.tokenizer = self.init_model()
        self.setup_training()

    def setup_environment(self) -> None:
        """设置训练环境"""
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            self.init_distributed_mode()
        
        os.makedirs(self.config.out_dir, exist_ok=True)
        torch.manual_seed(1337)
        
        device_type = 'cpu' if 'cpu' in self.config.device else 'cuda'
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=getattr(torch, self.config.dtype))
        
        if self.config.use_wandb and (not self.ddp or dist.get_rank() == 0):
            import wandb
            self.wandb = wandb
            wandb.init(project=self.config.wandb_project)
        else:
            self.wandb = None

    def init_distributed_mode(self):
        """初始化分布式训练环境"""
        dist.init_process_group(backend="nccl")
        self.ddp_rank = int(os.environ["RANK"])
        self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
        self.ddp_world_size = int(os.environ["WORLD_SIZE"])
        self.device = f"cuda:{self.ddp_local_rank}"
        torch.cuda.set_device(self.device)

    def init_model(self) -> Tuple[LLM, AutoTokenizer]:
        """初始化模型和分词器"""
        lm_config = LLMconfig(
            dim=self.config.dim,
            n_layers=self.config.n_layers,
            max_seq_len=self.config.max_seq_len,
            vocab_size=self.config.vocab_size,
            n_heads=self.config.n_heads,
            n_kv_heads=self.config.n_kv_heads,
            rope_theta=self.config.rope_theta,
            dropout=self.config.dropout,
            norm_eps=self.config.norm_eps,
            multiple_of=self.config.multiple_of,
            flash_attn=self.config.flash_attn,
            use_moe=self.config.use_moe,
            n_routed_experts=self.config.n_routed_experts,
            num_experts_per_tok=self.config.num_experts_per_tok,
            n_shared_experts=self.config.n_shared_experts,
            scoring_func=self.config.scoring_func,
            aux_loss_alpha=self.config.aux_loss_alpha,
            seq_aux=self.config.seq_aux,
            norm_topk_prob=self.config.norm_topk_prob
        )
        
        tokenizer_path = os.path.join(os.path.dirname(__file__), 'tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True
        )
        # 更新vocab_size以匹配tokenizer的词表大小
        lm_config.vocab_size = len(tokenizer)
        
        model = LLM(lm_config).to(self.config.device)
        
        # 如果是SFT模式，加载预训练权重
        if self.config.mode == TrainingMode.SFT:
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'
            state_dict = torch.load(ckp, map_location=self.config.device)
            model.load_state_dict(state_dict, strict=False)
            
        self.log(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
        return model, tokenizer

    def setup_training(self) -> None:
        """设置训练相关组件"""
        # 设置数据集
        Dataset = PretrainDataset if self.config.mode == TrainingMode.PRETRAIN else SFTDataset
        
        # 验证数据文件是否存在
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.config.data_path}")
            
        try:
            train_ds = Dataset(self.config.data_path, self.tokenizer, max_length=self.config.max_seq_len)
            # 验证数据集大小
            if len(train_ds) == 0:
                raise ValueError("数据集为空")
            self.log(f"数据集大小: {len(train_ds)}")
            
            # 验证第一个样本
            sample = train_ds[0]
            self.log(f"样本输入形状: {sample[0].shape}")
            self.log(f"样本标签形状: {sample[1].shape}")
            self.log(f"样本掩码形状: {sample[2].shape}")
            
        except Exception as e:
            raise RuntimeError(f"数据集加载失败: {str(e)}")
        
        # 设置数据加载器
        train_sampler = DistributedSampler(train_ds) if self.ddp else None
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=(train_sampler is None),
            num_workers=self.config.num_workers,
            sampler=train_sampler
        )
        
        # 设置优化器和梯度缩放器
        self.scaler = torch.amp.GradScaler(enabled=(self.config.dtype in ['float16', 'bfloat16']))
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1
        )
        
        if self.ddp:
            self.model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
            self.model = DistributedDataParallel(self.model, device_ids=[self.ddp_local_rank])

    def get_lr(self, current_step: int, total_steps: int) -> float:
        """计算学习率"""
        return self.config.learning_rate / 10 + 0.5 * self.config.learning_rate * (
            1 + math.cos(math.pi * current_step / total_steps)
        )

    def train(self) -> None:
        """训练循环"""
        # 清理缓存并重置
        if torch.cuda.is_available() and 'cuda' in self.config.device:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # 重置 CUDA 设备
            device = torch.device(self.config.device)
            torch.cuda.set_device(device)
            
            # 确保 CUDA 初始化正确
            try:
                torch.cuda.init()
            except RuntimeError:
                self.log("CUDA 初始化失败，尝试重新初始化...")
                torch.cuda.empty_cache()
                torch.cuda.init()
        
        # 确保模型在正确的设备上
        self.model = self.model.to(self.config.device)
        
        # 打印当前设备信息
        self.log(f"训练设备: {self.config.device}")
        if torch.cuda.is_available() and 'cuda' in self.config.device:
            self.log(f"当前GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            self.log(f"最大GPU内存使用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
            self.log(f"当前GPU设备索引: {torch.cuda.current_device()}")
        
        self.iter_per_epoch = len(self.train_loader)
        for epoch in range(self.config.epochs):
            try:
                self.train_epoch(epoch)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    self.log("CUDA 错误，尝试恢复训练...")
                    torch.cuda.empty_cache()
                    continue
                raise e

    def train_epoch(self, epoch: int) -> None:
        """训练一个epoch"""
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        start_time = time.time()
        
        for step, (X, Y, loss_mask) in enumerate(self.train_loader):
            try:
                # 确保输入数据维度正确
                if X.dim() != 2:
                    X = X.view(X.size(0), -1)
                if Y.dim() != 2:
                    Y = Y.view(Y.size(0), -1)
                
                # 将数据移动到设备
                X = X.to(self.config.device)
                Y = Y.to(self.config.device)
                loss_mask = loss_mask.to(self.config.device)

                # 更新学习率
                lr = self.get_lr(epoch * self.iter_per_epoch + step, self.config.epochs * self.iter_per_epoch)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                # 使用 autocast 进行混合精度训练
                with self.ctx:
                    # 同步 GPU
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    res = self.model(X)
                    loss = loss_fct(
                        res.logits.view(-1, res.logits.size(-1)),
                        Y.view(-1)
                    ).view(Y.size())
                    loss = (loss * loss_mask).sum() / loss_mask.sum()
                    loss += res.aux_loss
                    loss = loss / self.config.accumulation_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.config.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # 同步 GPU
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                if step % self.config.log_interval == 0:
                    self.log_progress(epoch, step, loss, start_time)

                if (step + 1) % self.config.save_interval == 0:
                    self.save_checkpoint()
                    
            except RuntimeError as e:
                if "CUDA" in str(e):
                    self.log(f"步骤 {step} CUDA 错误: {str(e)}")
                    torch.cuda.empty_cache()
                    continue
                raise e

    def log_progress(self, epoch: int, step: int, loss: torch.Tensor, start_time: float) -> None:
        """记录训练进度"""
        spend_time = time.time() - start_time
        self.log(
            'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                epoch + 1,
                self.config.epochs,
                step,
                self.iter_per_epoch,
                loss.item() * self.config.accumulation_steps,
                self.optimizer.param_groups[-1]['lr'],
                spend_time / (step + 1) * self.iter_per_epoch // 60 - spend_time // 60
            )
        )

        if self.wandb is not None and (not self.ddp or dist.get_rank() == 0):
            self.wandb.log({
                "loss": loss.item() * self.config.accumulation_steps,
                "lr": self.optimizer.param_groups[-1]['lr'],
                "epoch_Time": spend_time / (step + 1) * self.iter_per_epoch // 60 - spend_time // 60
            })

    def save_checkpoint(self) -> None:
        """保存模型检查点"""
        if not self.ddp or dist.get_rank() == 0:
            self.model.eval()
            moe_path = '_moe' if self.config.use_moe else ''
            mode_prefix = "pretrain" if self.config.mode == TrainingMode.PRETRAIN else "sft"
            ckp = f'{self.config.out_dir}/{mode_prefix}_{self.config.dim}{moe_path}.pth'
            
            if isinstance(self.model, DistributedDataParallel):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
                
            torch.save(state_dict, ckp)
            self.model.train()

    def log(self, content: str) -> None:
        """日志记录"""
        if not self.ddp or dist.get_rank() == 0:
            print(content)

class ModelTester:
    """模型测试类"""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model, self.tokenizer = self.init_model()

    def init_model(self) -> Tuple[LLM, AutoTokenizer]:
        """初始化测试模型"""
        return Trainer(self.config).init_model()

    def run_tests(self) -> None:
        """运行测试用例"""
        test_suite = unittest.TestLoader().loadTestsFromTestCase(ModelTests)
        unittest.TextTestRunner(verbosity=2).run(test_suite)

class ModelTests(unittest.TestCase):
    """模型测试用例"""
    def setUp(self):
        self.config = TrainingConfig(mode=TrainingMode.TEST)
        self.trainer = Trainer(self.config)

    def test_model_output(self):
        """测试模型输出"""
        # 添加具体的测试用例
        pass

def main():
    parser = argparse.ArgumentParser(description="Zer02LLM Training Framework")
    parser.add_argument("--mode", type=str, choices=["pretrain", "sft", "test"], default="pretrain",
                      help="Training mode: pretrain, sft, or test")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Zer02LLM")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--local_rank", type=int, default=-1)
    
    # 模型参数
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=2048)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=None)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--norm_eps", type=float, default=1e-8)
    parser.add_argument("--multiple_of", type=int, default=64)
    parser.add_argument("--flash_attn", action="store_true")
    
    # MoE参数
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--n_routed_experts", type=int, default=8)
    parser.add_argument("--num_experts_per_tok", type=int, default=2)
    parser.add_argument("--n_shared_experts", type=int, default=0)
    parser.add_argument("--scoring_func", type=str, default="softmax")
    parser.add_argument("--aux_loss_alpha", type=float, default=0.1)
    parser.add_argument("--seq_aux", action="store_true")
    parser.add_argument("--norm_topk_prob", action="store_true")
    
    # 数据路径
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        mode=TrainingMode(args.mode),
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        dtype=args.dtype,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        num_workers=args.num_workers,
        ddp=args.ddp,
        accumulation_steps=args.accumulation_steps,
        grad_clip=args.grad_clip,
        warmup_iters=args.warmup_iters,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        local_rank=args.local_rank,
        dim=args.dim,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        rope_theta=args.rope_theta,
        dropout=args.dropout,
        norm_eps=args.norm_eps,
        multiple_of=args.multiple_of,
        flash_attn=args.flash_attn,
        use_moe=args.use_moe,
        n_routed_experts=args.n_routed_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        n_shared_experts=args.n_shared_experts,
        scoring_func=args.scoring_func,
        aux_loss_alpha=args.aux_loss_alpha,
        seq_aux=args.seq_aux,
        norm_topk_prob=args.norm_topk_prob,
        data_path=args.data_path
    )
    
    if config.mode == TrainingMode.TEST:
        tester = ModelTester(config)
        tester.run_tests()
    else:
        trainer = Trainer(config)
        trainer.train()

if __name__ == "__main__":
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    print(f"当前CUDA版本: {torch.version.cuda}")
    print(f"GPU设备数量: {torch.cuda.device_count()}")
    print(f"当前GPU设备: {torch.cuda.current_device()}")
    print(f"GPU设备名称: {torch.cuda.get_device_name(0)}")
    
    main()