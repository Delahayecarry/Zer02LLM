import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import numpy as np
import re

# 设置页面配置
st.set_page_config(
    page_title="Zero2LLM Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 设置样式
st.markdown("""
    <style>
        /* 添加操作按钮样式 */
        .stButton button {
            border-radius: 50% !important;
            width: 32px !important;
            height: 32px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;
            margin: 5px 10px 5px 0 !important;
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }
        
        /* 重置按钮基础样式 */
        .stButton > button {
            all: unset !important;
            box-sizing: border-box !important;
            border-radius: 50% !important;
            width: 18px !important;
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            max-width: 18px !important;
            max-height: 18px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #888 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            margin: 0 2px !important;
        }
    </style>
""", unsafe_allow_html=True)

# 全局变量
system_prompt = []
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu" 

# 模型路径映射
MODEL_PATHS = {
    "Zero2LLM-v1-pretrain-0.02B": ["../Zer02LLM-v1-pretrain-0.02B", "Zero2LLM-v1-pretrain-0.02B"],
    "Zero2LLM-v1-sft-0.02B": ["../Zer02LLM-v1-sft-0.02B", "Zero2LLM-v1-sft-0.02B"],
    "Zero2LLM-v1-dpo-0.02B": ["../Zer02LLM-v1-dpo-0.02B", "Zero2LLM-v1-dpo-0.02B"]
}

def setup_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@st.cache_resource
def load_model_tokenizer(model_path):
    """加载模型和分词器"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    model = model.eval().to(device)
    return model, tokenizer

def clear_chat_messages():
    """清空聊天记录"""
    del st.session_state.messages
    del st.session_state.chat_messages

def init_chat_messages():
    """初始化聊天消息"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

def main():
    # 侧边栏配置
    st.sidebar.title("模型设定调整")
    st.sidebar.text("【注】可以根据需要调整以下参数")
    
    # 模型参数设置
    st.session_state.history_chat_num = st.sidebar.slider("历史对话轮数", 0, 6, 0, step=2)
    st.session_state.max_new_tokens = st.sidebar.slider("最大序列长度", 256, 8192, 2048, step=256)
    st.session_state.top_p = st.sidebar.slider("Top-P", 0.8, 0.99, 0.9, step=0.01)
    st.session_state.temperature = st.sidebar.slider("Temperature", 0.6, 1.2, 0.7, step=0.01)

    # 选择模型
    selected_model = st.sidebar.selectbox('选择模型', list(MODEL_PATHS.keys()), index=0)
    model_path = MODEL_PATHS[selected_model][0]
    
    # 加载模型
    model, tokenizer = load_model_tokenizer(model_path)

    # 页面标题
    st.title(f"🤖 {MODEL_PATHS[selected_model][1]} Chat")
    st.markdown('<p style="color: #666; font-style: italic;">内容完全由AI生成，请务必仔细甄别</p>', unsafe_allow_html=True)

    # 初始化消息
    init_chat_messages()
    messages = st.session_state.messages

    # 显示历史消息
    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message["content"], unsafe_allow_html=True)
                if st.button("×", key=f"delete_{i}"):
                    st.session_state.messages = st.session_state.messages[:i-1]
                    st.session_state.chat_messages = st.session_state.chat_messages[:i-1]
                    st.rerun()
        else:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: gray; border-radius: 10px; color: white;">{message["content"]}</div></div>',
                unsafe_allow_html=True
            )

    # 用户输入
    user_input = st.chat_input("输入消息...")

    if user_input:
        # 显示用户消息
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: gray; border-radius: 10px; color: white;">{user_input}</div></div>',
            unsafe_allow_html=True
        )
        
        # 添加用户消息到历史记录
        messages.append({"role": "user", "content": user_input})
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        # 生成回复
        with st.chat_message("assistant"):
            placeholder = st.empty()
            
            # 设置随机种子
            random_seed = random.randint(0, 2**32 - 1)
            setup_seed(random_seed)

            # 准备输入
            st.session_state.chat_messages = system_prompt + st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):]
            new_prompt = tokenizer.apply_chat_template(
                st.session_state.chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )[-(st.session_state.max_new_tokens - 1):]

            # 生成回复
            x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=device).unsqueeze(0)
            with torch.no_grad():
                try:
                    for y in model.generate(
                        x,
                        tokenizer.eos_token_id,
                        max_new_tokens=st.session_state.max_new_tokens,
                        temperature=st.session_state.temperature,
                        top_p=st.session_state.top_p,
                        stream=True
                    ):
                        answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                        if (answer and answer[-1] == '') or not answer:
                            continue
                        placeholder.markdown(answer, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"生成回复时发生错误: {str(e)}")
                    return

            # 保存助手回复
            assistant_answer = answer.replace(new_prompt, "")
            messages.append({"role": "assistant", "content": assistant_answer})
            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_answer})

if __name__ == "__main__":
    main() 