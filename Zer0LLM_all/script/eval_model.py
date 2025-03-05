import argparse
import random
import time
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer
from model.model import LLM
from model.LLMconfig import LLMconfig

warnings.filterwarnings('ignore')


def init_model(args):
    """初始化模型和tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(
        './tokenizer',
        trust_remote_code=True,
        local_files_only=True
    )
    
    moe_path = '_moe' if args.use_moe else ''
    modes = {0: 'pretrain', 1: 'sft'}
    ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{args.dim}{moe_path}.pth'

    model = LLM(LLMconfig(
        dim=args.dim,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        vocab_size=len(tokenizer),
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
        norm_topk_prob=args.norm_topk_prob
    ))

    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=True)
    
    print(f'LLM总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M')
    return model.eval().to(args.device), tokenizer


def get_prompt_datas(args):
    """获取测试prompt"""
    if args.model_mode == 0:
        # 预训练模型的文本续写能力
        prompt_datas = [
            '人工智能的发展历程',
            '深度学习的基本原理',
            '计算机视觉技术主要包括',
            '自然语言处理的应用场景',
            '机器学习算法可以分为',
            '神经网络的基本组成',
            '强化学习的核心思想'
        ]
    else:
        # 对话测试问题
        prompt_datas = [
            '请介绍一下自己。',
            '你的训练数据来自哪里？',
            '解释一下什么是深度学习。',
            '如何评价大语言模型的能力？',
            '详细介绍Transformer架构。',
            '解释一下注意力机制的原理。',
            '什么是MoE（混合专家）模型？',
            'Explain what is machine learning.',
            '如何优化神经网络的训练过程？'
        ]
    return prompt_datas


def setup_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Zer02LLM Evaluation")
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    
    # 模型参数
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=2048, type=int)
    parser.add_argument('--n_heads', default=8, type=int)
    parser.add_argument('--n_kv_heads', default=None, type=int)
    parser.add_argument('--rope_theta', default=10000.0, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--norm_eps', default=1e-8, type=float)
    parser.add_argument('--multiple_of', default=64, type=int)
    parser.add_argument('--flash_attn', action='store_true')
    
    # MoE参数
    parser.add_argument('--use_moe', action='store_true')
    parser.add_argument('--n_routed_experts', default=8, type=int)
    parser.add_argument('--num_experts_per_tok', default=2, type=int)
    parser.add_argument('--n_shared_experts', default=0, type=int)
    parser.add_argument('--scoring_func', default='softmax', type=str)
    parser.add_argument('--aux_loss_alpha', default=0.1, type=float)
    parser.add_argument('--seq_aux', action='store_true')
    parser.add_argument('--norm_topk_prob', action='store_true')
    
    # 生成参数
    parser.add_argument('--history_cnt', default=0, type=int, help="携带历史对话上下文条数")
    parser.add_argument('--stream', default=True, type=bool, help="是否启用流式输出")
    parser.add_argument('--model_mode', default=1, type=int, help="0: 预训练模型，1: SFT模型")
    
    args = parser.parse_args()

    model, tokenizer = init_model(args)

    prompts = get_prompt_datas(args)
    test_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    messages = []
    
    for idx, prompt in enumerate(prompts if test_mode == 0 else iter(lambda: input('User: '), '')):
        setup_seed(random.randint(0, 2048))
        if test_mode == 0: 
            print(f'User: {prompt}')

        messages = messages[-args.history_cnt:] if args.history_cnt else []
        messages.append({"role": "user", "content": prompt})

        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-args.max_seq_len + 1:] if args.model_mode != 0 else (tokenizer.bos_token + prompt)

        answer = new_prompt
        with torch.no_grad():
            x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=args.device).unsqueeze(0)
            outputs = model.generate(
                x,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_seq_len,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=True,
                pad_token_id=tokenizer.pad_token_id
            )

            print('Assistant: ', end='')
            try:
                if not args.stream:
                    print(tokenizer.decode(outputs.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True), end='')
                else:
                    history_idx = 0
                    for y in outputs:
                        answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                        if (answer and answer[-1] == '') or not answer:
                            continue
                        print(answer[history_idx:], end='', flush=True)
                        history_idx = len(answer)
            except StopIteration:
                print("No answer")
            print('\n')

        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
