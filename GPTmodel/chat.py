import torch
import argparse
from GPTmodel.model import GPT, GPTconfig
from GPTmodel.data import get_tokenizer

def load_model(model_path):
    """
    加载预训练模型
    
    参数:
        model_path: 模型路径
        
    返回:
        加载好的模型
    """
    config = GPTconfig()
    model = GPT(config)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model' in checkpoint:
        # 如果是训练过程中保存的检查点
        model.load_state_dict(checkpoint['model'])
    else:
        # 如果只是模型权重
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8):
    """
    生成对话回答
    
    参数:
        model: GPT模型
        tokenizer: 分词器
        prompt: 用户输入的文本
        max_new_tokens: 最大生成长度
        temperature: 采样温度
        
    返回:
        str: 生成的回答
    """
    # 对输入进行编码
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    
    try:
        # 生成回答
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
        
        # 解码输出
        response = tokenizer.decode(output_ids[0].tolist())
        
        # 只返回新生成的内容
        original_prompt_len = len(prompt)
        response = response[original_prompt_len:]
        
        return response.strip()
    
    except Exception as e:
        print(f"生成回答时发生错误: {e}")
        return "抱歉，生成回答时出现了错误。"

def chat(model_path):
    """
    启动聊天界面
    
    参数:
        model_path: 模型路径
    """
    print("正在加载模型...")
    model = load_model(model_path)
    tokenizer = get_tokenizer()
    
    print("\n模型已加载，开始聊天! (输入 'quit' 退出)")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n你: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n结束聊天!")
                break
                
            if not user_input.strip():
                continue
                
            response = generate_response(model, tokenizer, user_input)
            print(f"\n机器人: {response}")
            
        except KeyboardInterrupt:
            print("\n结束聊天!")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")

def main():
    parser = argparse.ArgumentParser(description='GPT聊天机器人')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pt', 
                        help='模型路径')
    args = parser.parse_args()
    
    chat(args.model_path)

if __name__ == "__main__":
    main() 