# Zer02LLM 私有化wandb部署方案

本目录包含用于部署私有化Weights & Biases (wandb)服务器的Docker配置文件，支持Zer02LLM模型训练工作流。

## 部署架构

基本部署架构包括：

1. **wandb-local服务**：核心wandb服务，提供实验跟踪、可视化和协作功能
2. **MySQL数据库**（可选）：用于生产环境中存储wandb数据
3. **Nginx反向代理**（可选）：提供SSL加密和域名配置

## 快速开始

### 基本部署（开发环境）

1. 复制环境变量模板并根据需要修改：

```bash
cp .env.example .env
```

2. 启动wandb服务：

```bash
docker-compose up -d wandb
```

3. 访问wandb界面：

```
http://localhost:8080
```

### 生产环境部署

1. 修改`.env`文件，配置管理员账户和数据库密码

2. 编辑`docker-compose.yml`，取消MySQL和Nginx服务的注释

3. 配置域名和SSL证书：
   - 将您的域名证书放在`nginx/ssl/`目录下
   - 编辑`nginx/conf.d/wandb.conf`，将`wandb.example.com`替换为您的实际域名

4. 启动所有服务：

```bash
docker-compose up -d
```

5. 访问wandb界面：

```
https://your-domain.com
```

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|------|------|--------|
| `WANDB_ADMIN_USERNAME` | 管理员用户名 | admin |
| `WANDB_ADMIN_PASSWORD` | 管理员密码 | admin |
| `WANDB_ADMIN_EMAIL` | 管理员邮箱 | admin@example.com |
| `MYSQL_ROOT_PASSWORD` | MySQL root密码 | wandb_root_password |
| `MYSQL_PASSWORD` | MySQL wandb用户密码 | wandb_password |
| `WANDB_HOST` | wandb主机地址 | wandb.example.com |
| `WANDB_BASE_URL` | wandb基础URL | https://wandb.example.com |

### 持久化存储

数据持久化通过Docker卷实现：

- `wandb_data`：存储实验数据、模型文件等
- `wandb_logs`：存储服务日志
- `mysql_data`：存储数据库文件（生产环境）

## 安全配置

### SSL证书

1. 获取SSL证书（可以使用Let's Encrypt）：

```bash
mkdir -p nginx/ssl
# 使用certbot获取证书
certbot certonly --standalone -d your-domain.com
# 复制证书到nginx目录
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/wandb.crt
cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/wandb.key
```

2. 配置Nginx使用SSL证书（已在配置文件中设置）

### 防火墙配置

确保以下端口开放：

- 80/TCP（HTTP，用于重定向到HTTPS）
- 443/TCP（HTTPS，用于安全访问）

## 与Zer02LLM工作流集成

部署完成后，可以使用`setup_private_wandb.py`脚本配置Zer02LLM工作流：

```bash
cd ..
python setup_private_wandb.py --host your-domain.com --base_url https://your-domain.com --all
```

## 维护操作

### 备份数据

```bash
# 备份wandb数据
docker run --rm -v wandb_data:/source -v $(pwd)/backups:/target alpine tar -czf /target/wandb_data_$(date +%Y%m%d).tar.gz -C /source .

# 备份MySQL数据（如果使用）
docker exec wandb-mysql mysqldump -u root -p${MYSQL_ROOT_PASSWORD} --all-databases > backups/mysql_$(date +%Y%m%d).sql
```

### 更新wandb版本

```bash
docker-compose pull
docker-compose up -d
```

### 查看日志

```bash
# 查看wandb服务日志
docker-compose logs -f wandb

# 查看Nginx日志
docker-compose logs -f nginx
```

## 故障排除

### 连接问题

- 确保防火墙配置正确
- 检查Nginx配置是否正确
- 验证SSL证书是否有效

### 数据库问题

- 检查MySQL服务是否正常运行
- 验证数据库凭据是否正确

### 权限问题

- 确保Docker卷权限设置正确
- 检查Nginx用户是否有权访问SSL证书

## 参考资源

- [Weights & Biases 文档](https://docs.wandb.ai/)
- [Docker 文档](https://docs.docker.com/)
- [Nginx 文档](https://nginx.org/en/docs/) 