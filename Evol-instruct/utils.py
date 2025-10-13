import os
from dotenv import load_dotenv
import pandas as pd
from raganything import RAGAnything
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import asyncio
import re

# 加载环境变量文件
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

from openai import OpenAI


def generate(source, prompt, model,enable_thinking=False,enable_search=False):

    """
    调用API生成内容
    
    Args:
        source (str): API源，"ali"表示使用阿里源，其他值使用默认源
        prompt (str): 输入提示词
        enable_thinking (bool): 是否启用思考模式（仅在使用阿里源时有效）
        
    
    Returns:
        str: API返回的结果
    """
    if source == "ali":
        # 使用阿里源
        api_key = os.getenv("Ali_API_KEY")
        base_url = os.getenv("Ali_BASE_URL")
        
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 如果启用思考模式，则添加相应参数
        if enable_thinking:
            extra_body={"enable_thinking": True}
        else:
            extra_body={"enable_thinking": False}
        if enable_search:
            extra_body.update({"enable_search": True})
        else:
            extra_body.update({"enable_search": False})
        
        response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_body=extra_body,
    )
    else:
        # 使用默认源（如OpenAI）
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    
    return response.choices[0].message.content


def evol_instruct(mode,given_prompt,model,enable_thinking=False):
    """
    指令进化函数
    """
    df = pd.read_csv("./prompt.csv")
    if mode == "deep":
        gen_prompt = df["prompt"][0]
    elif mode == "con":
        gen_prompt = df["prompt"][1]
    elif mode == "reason":
        gen_prompt = df["prompt"][2]
    
    gen_prompt = gen_prompt.replace("$Given_Prompt$",given_prompt)
    return generate("ali", gen_prompt, model,enable_thinking)

def instruct_checker(previous_instruct,after_instruct):
    df = pd.read_csv("./prompt.csv")
    prompt = df["prompt"][3]
    prompt = prompt.replace("$previous$",previous_instruct)
    prompt = prompt.replace("$after$",after_instruct)

    output = generate("openai",prompt,"gpt-4o")

    return output

def answer(instruction,model):
    #addition_info = asyncio.run(load_existing_lightrag(instruction,"global"))
    pass

def instruct_classifier(instruction):#TODO:训练个简易分类器或者使用快速模型进行判断
    pass

def answer_checker():
    pass

def output_selecter():
    pass

async def load_existing_lightrag(work_dir,query:str,mode):
    # 设置 API 配置
    api_key = ""
    base_url = ""  # 可选

    # 首先，创建或加载已存在的 LightRAG 实例
    lightrag_working_dir = "../data_gen/storage/"
    lightrag_working_dir = os.path.join(lightrag_working_dir, work_dir)

    # 检查是否存在之前的 LightRAG 实例
    if os.path.exists(lightrag_working_dir) and os.listdir(lightrag_working_dir):
        print("✅ 发现已存在的 LightRAG 实例，正在加载...")
    else:
        print("❌ 未找到已存在的 LightRAG 实例，将创建新实例")

    # 使用您的配置创建/加载 LightRAG 实例
    lightrag_instance = LightRAG(
        working_dir=lightrag_working_dir,
        llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
            "gpt-4o",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        ),
        embedding_func=EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model="text-embedding-3-large",
                api_key=api_key,
                base_url=base_url,
            ),
        )
    )

    # 初始化存储（如果有现有数据，这将加载它们）
    await lightrag_instance.initialize_storages()
    #await initialize_pipeline_status()


    # 现在使用已存在的 LightRAG 实例初始化 RAGAnything
    rag = RAGAnything(
        lightrag=lightrag_instance,  # 传入已存在的 LightRAG 实例
        
        # 注意：working_dir、llm_model_func、embedding_func 等都从 lightrag_instance 继承
    )
    
    


    # 查询已存在的知识库
    result =  await rag.aquery(
        query,
        mode
    )
    print("查询结果:", result)

    return result   
    