import asyncio
from raganything import RAGAnything
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import os
from openai import OpenAI
import pandas as pd
import re
import json

df = pd.read_csv("./prompt_qgen.csv")
gen_prompt = df["prompt"][0]
client = OpenAI(
    api_key="",
    base_url=""
)
async def load_existing_lightrag(query:str,mode):
    # 设置 API 配置
    api_key = ""
    base_url = ""  # 可选

    # 首先，创建或加载已存在的 LightRAG 实例
    lightrag_working_dir = "./storage/household"

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
    
instruct_data = []
previous_question = ""
message = asyncio.run(load_existing_lightrag("关于家居补贴消费者补贴流程和参与原则，商家的申报补贴流程以及资讯方式","global"))
gen_prompt = gen_prompt.replace("$message",message)
for i in range(15):
    user_prompt = gen_prompt.replace("$previous_question",previous_question)

    output = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {"role": "system", "content": "你是一个助手,擅长帮我根据信息构建问题-回答对"},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
    )
    
    # 提取生成的问题和回答
    response_text = output.choices[0].message.content
    print(response_text)
    # 使用正则表达式提取问题和回答
    
    question_match = re.search(r"<问题>:\s*(.*?)\s*<回答>", response_text, re.DOTALL)
    answer_match = re.search(r"<回答>:\s*(.*)", response_text, re.DOTALL)
    
    if question_match and answer_match:
        question = question_match.group(1).strip()
        answer = answer_match.group(1).strip()
        
        # 将问题-回答对加入instruction_data
        instruct_data.append({
            "instruction": question,
            "input": "",
            "output": answer,
            "system":"你是一个擅长以旧换新政策解答的助手，你能够用清晰自然的语言解答政策相关问题的疑惑"
        }) 
        
        # 将当前问题添加到previous_question中
        if previous_question:
            previous_question +=  f"{i}:" + question + "\n"
        else:
            previous_question = question


input = input("请输入是否选择接受：")
if input == "是":
    print("接受")

    with open("./instruction_dataset.json", "a+", encoding="utf-8") as f:
        # 检查文件是否为空
        f.seek(0)
        content = f.read().strip()
        
        if content:  # 如果文件中有内容
            # 将文件指针移到开头，读取现有数据
            f.seek(0)
            try:
                existing_data = json.load(f)
                # 合并现有数据和新生成的数据
                combined_data = existing_data + instruct_data
            except json.JSONDecodeError:
                # 如果JSON解析失败，则只使用新数据
                combined_data = instruct_data
        else:
            # 如果文件为空，则直接使用新数据
            combined_data = instruct_data
        
        # 清空文件内容并写入合并后的数据
        f.seek(0)
        f.truncate()
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
