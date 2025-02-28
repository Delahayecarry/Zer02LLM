import torch
import unittest
from model.model import LLM, LLMconfig, FFN, Attention, RMSnorm, MOEgate, MoEFFN, LLMBlock
import torch.nn as nn

class TestLLMComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 设置基本配置
        cls.config = LLMconfig(
            dim=512,  # 模型维度
            n_layers=4,  # 层数
            n_heads=8,  # 注意力头数
            vocab_size=32000,  # 词表大小
            max_seq_len=2048,  # 最大序列长度
            dropout=0.1,  # dropout率
            norm_eps=1e-5,  # 归一化epsilon
            rope_theta=10000,  # RoPE theta参数
            n_kv_heads=None,  # KV头数（如果为None则等于n_heads）
            multiple_of=256,  # FFN隐藏层维度的倍数
            hidden_dim=None,  # FFN隐藏层维度（如果为None则自动计算）
            use_moe=True,  # 是否使用MoE
            num_experts=8,  # 专家数量
            num_experts_per_tok=2,  # 每个token使用的专家数量
            n_routed_experts=8,  # 路由专家数量
            n_shared_experts=None,  # 共享专家数量
            norm_topk_experts=True,  # 是否归一化top-k专家权重
            seq_aux=True,  # 是否使用序列辅助损失
            aux_loss_alpha=0.01,  # 辅助损失权重
            scoring_func='softmax'  # 专家选择评分函数
        )
        cls.batch_size = 4
        cls.seq_len = 32
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n使用设备: {cls.device}")
        print(f"配置信息: dim={cls.config.dim}, n_heads={cls.config.n_heads}, n_layers={cls.config.n_layers}")

    def test_rmsnorm(self):
        """测试RMSNorm层"""
        print("\n测试RMSNorm层...")
        norm = RMSnorm(self.config.dim, self.config.norm_eps)
        x = torch.randn(self.batch_size, self.seq_len, self.config.dim)
        print(f"输入张量形状: {x.shape}")
        out = norm(x)
        print(f"输出张量形状: {out.shape}")
        print(f"权重形状: {norm.weight.shape}")
        self.assertEqual(out.shape, x.shape)
        rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
        print(f"RMS值形状: {rms.shape}, 均值: {rms.mean().item():.4f}")
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-5))

    def test_attention(self):
        """测试注意力层"""
        print("\n测试注意力层...")
        attn = Attention(self.config)
        x = torch.randn(self.batch_size, self.seq_len, self.config.dim)
        pos_cis = torch.randn(self.seq_len, self.config.dim // self.config.n_heads // 2, dtype=torch.complex64)
        print(f"输入张量形状: {x.shape}")
        print(f"位置编码形状: {pos_cis.shape}")
        print(f"注意力头数: {self.config.n_heads}, 每头维度: {self.config.dim // self.config.n_heads}")
        out, past_kv = attn(x, pos_cis)
        print(f"输出张量形状: {out.shape}")
        if past_kv is not None:
            print(f"past_kv形状: k={past_kv[0].shape}, v={past_kv[1].shape}")
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.config.dim))

    def test_ffn(self):
        """测试FFN层"""
        print("\n测试FFN层...")
        ffn = FFN(self.config)
        x = torch.randn(self.batch_size, self.seq_len, self.config.dim)
        print(f"输入张量形状: {x.shape}")
        print(f"FFN隐藏层维度: {ffn.w1.out_features}")
        out = ffn(x)
        print(f"输出张量形状: {out.shape}")
        self.assertEqual(out.shape, x.shape)

    def test_moe_gate(self):
        """测试MoE门控层"""
        print("\n测试MoE门控层...")
        gate = MOEgate(self.config)
        x = torch.randn(self.batch_size, self.seq_len, self.config.dim)
        print(f"输入张量形状: {x.shape}")
        print(f"门控权重形状: {gate.weights.shape}")
        top_k_idx, top_k_weights, aux_loss = gate(x)
        print(f"专家索引形状: {top_k_idx.shape}")
        print(f"专家权重形状: {top_k_weights.shape}")
        print(f"辅助损失值: {aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss}")
        self.assertEqual(top_k_idx.shape, (self.batch_size * self.seq_len, self.config.num_experts_per_tok))
        self.assertEqual(top_k_weights.shape, (self.batch_size * self.seq_len, self.config.num_experts_per_tok))

    def test_moe_ffn(self):
        """测试MoE FFN层"""
        print("\n测试MoE FFN层...")
        moe_ffn = MoEFFN(self.config)
        x = torch.randn(self.batch_size, self.seq_len, self.config.dim)
        print(f"输入张量形状: {x.shape}")
        print(f"专家数量: {len(moe_ffn.experts)}")
        out = moe_ffn(x)
        print(f"输出张量形状: {out.shape}")
        print(f"辅助损失值: {moe_ffn.aux_loss.item() if hasattr(moe_ffn, 'aux_loss') else None}")
        self.assertEqual(out.shape, x.shape)

    def test_llm_block(self):
        """测试LLM Block"""
        print("\n测试LLM Block...")
        block = LLMBlock(self.config, layer_id=0)
        x = torch.randn(self.batch_size, self.seq_len, self.config.dim)
        pos_cis = torch.randn(self.seq_len, self.config.dim // self.config.n_heads // 2, dtype=torch.complex64)
        print(f"输入张量形状: {x.shape}")
        print(f"位置编码形状: {pos_cis.shape}")
        out, past_kv = block(x, pos_cis)
        print(f"输出张量形状: {out.shape}")
        if past_kv is not None:
            print(f"past_kv形状: k={past_kv[0].shape}, v={past_kv[1].shape}")
        self.assertEqual(out.shape, x.shape)

    def test_full_model(self):
        """测试完整的LLM模型"""
        print("\n测试完整的LLM模型...")
        model = LLM(self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        print(f"输入ID形状: {input_ids.shape}")
        print(f"词表大小: {self.config.vocab_size}")
        output = model(input_ids)
        print(f"输出logits形状: {output.logits.shape}")
        print(f"辅助损失值: {output.aux_loss.item() if hasattr(output, 'aux_loss') else None}")
        if output.past_key_values:
            print(f"past_key_values数量: {len(output.past_key_values)}")
            print(f"第一层past_kv形状: k={output.past_key_values[0][0].shape}, v={output.past_key_values[0][1].shape}")
        self.assertEqual(output.logits.shape, (self.batch_size, self.seq_len, self.config.vocab_size))

    def test_model_generation(self):
        """测试模型生成功能"""
        print("\n测试模型生成功能...")
        model = LLM(self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (1, 10))
        print(f"输入ID形状: {input_ids.shape}")
        generated = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.9,
            stream=False
        )
        print(f"生成序列形状: {generated.shape}")
        print(f"生成的新token数: {generated.shape[1] - input_ids.shape[1]}")
        self.assertTrue(generated.shape[1] > input_ids.shape[1])

    def test_model_stream_generation(self):
        """测试模型流式生成功能"""
        print("\n测试模型流式生成功能...")
        model = LLM(self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (1, 10))
        print(f"输入ID形状: {input_ids.shape}")
        generator = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.9,
            stream=True
        )
        tokens = []
        for i, gen_ids in enumerate(generator):
            tokens.append(gen_ids)
            print(f"第{i+1}步生成的序列形状: {gen_ids.shape}")
            self.assertTrue(gen_ids.shape[1] > 0)
        print(f"总共生成了{len(tokens)}个token")

def run_tests():
    """运行所有测试"""
    unittest.main(argv=[''], verbosity=2)

if __name__ == '__main__':
    run_tests() 