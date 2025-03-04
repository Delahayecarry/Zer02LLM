import os 
import torch    
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
from cankao import GPT, GPTconfig
from data import TextDataset

# 创建保存目录
os.makedirs('checkpoints', exist_ok=True)

# 设置随机种子，确保结果可复现
torch.manual_seed(1024)
# 对于分布式训练，需要设置不同的随机种子
# torch.cuda.manual_seed_all(1024)
# torch.backends.cudnn.deterministic = True

# 早停类
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        """
        初始化早停
        
        参数:
            patience: 容忍的epoch数
            min_delta: 最小改善量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 训练函数
def train(model, device, train_loader, optimizer, scheduler, epoch):
    """
    训练一个epoch
    
    参数:
        model: 模型
        device: 设备
        train_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        
    返回:
        平均训练损失
    """
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, target=y)
        loss.backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    return total_loss / len(train_loader)

# 评估函数
def evaluate(model, device, val_loader):
    """
    评估模型
    
    参数:
        model: 模型
        device: 设备
        val_loader: 验证数据加载器
        
    返回:
        平均验证损失
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, target=y)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def main():
    # 加载数据
    data_path = r'E:\xiangmu\AAHOMEWORK\Transformer\GPTmodel\datasets\pretrain_hq.jsonl'  # 请替换为您的数据路径
    dataset = TextDataset(data_path)
    
    # 分割数据集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    config = GPTconfig()
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 初始化模型
    model = GPT(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f} M")
    
    # 设置优化器和学习率调度器
    num_epochs = 10
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = num_training_steps // 10  # 10%的步数用于warmup
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 初始化早停
    early_stopping = EarlyStopping(patience=3)
    best_val_loss = float('inf')
    
    # 检查是否存在检查点
    start_epoch = 0
    if os.path.exists('checkpoints/latest.pt'):
        print("正在加载检查点...")
        checkpoint = torch.load('checkpoints/latest.pt')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"从 {start_epoch} epoch恢复训练...")
    
    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        train_loss = train(model, device, train_loader, optimizer, scheduler, epoch)
        val_loss = evaluate(model, device, val_loader)
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoints = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoints, 'checkpoints/best_model.pt')
        
        # 保存最新检查点
        checkpoints = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoints, 'checkpoints/latest.pt')
        
        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

if __name__ == "__main__":
    main()
    
