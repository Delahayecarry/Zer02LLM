import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"cuDNN版本: {torch.backends.cudnn.version()}")
print(f"设备数量: {torch.cuda.device_count()}")
print(f"当前设备: {torch.cuda.current_device()}")
print(f"设备名称: {torch.cuda.get_device_name(0)}")

# 测试基本张量运算
x = torch.randn(2,2).cuda()
y = torch.randn(2,2).cuda()
z = x @ y
print("矩阵乘法测试通过:", z)

