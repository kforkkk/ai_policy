import os
from dotenv import load_dotenv
import pandas as pd
from raganything import RAGAnything
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import asyncio
import re
import logging
from datetime import datetime

def setup_simple_logger(log_file=None, level="INFO"):
    """
    设置简单的日志记录器
    
    Args:
        log_file: 日志文件路径，None表示仅输出到控制台
        level: 日志级别，可选：DEBUG/INFO/WARNING/ERROR
    
    Returns:
        logging.Logger: 配置好的logger实例
    """
    # 配置基本日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 设置日志级别
    logging.basicConfig(
        level=getattr(logging, level),
        format=log_format,
        datefmt=date_format
    )
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(
            f"{log_file}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logging.getLogger().addHandler(file_handler)
    
    return logging.getLogger()

# 全局logger实例
logger = setup_simple_logger(
    log_file="./evol.log",
)
# 加载环境变量文件
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

from openai import OpenAI

csv_path = os.path.join(os.path.dirname(__file__), "prompt.csv")
df = pd.read_csv(csv_path)

# 预加载所有知识向量库实例
lightrag_instances = {}
lightrag_working_dir =  os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_gen/storage')
print("数据存储目录:", lightrag_working_dir)

# 支持的知识库列表
supported_knowledge_bases = ["car", "digital_household", "elec_car", "food", "household"]

# 初始化所有知识库实例
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

for kb_name in supported_knowledge_bases:
    kb_working_dir = os.path.join(lightrag_working_dir, kb_name)
    
    if os.path.exists(kb_working_dir) and os.listdir(kb_working_dir):
        print(f"✅ 发现已存在的 {kb_name} LightRAG 实例，正在加载...")
    else:
        print(f"❌ 未找到已存在的 {kb_name} LightRAG 实例，将创建新实例")
    
    # 创建/加载 LightRAG 实例
    lightrag_instances[kb_name] = LightRAG(
        working_dir=kb_working_dir,
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
    # 初始化存储
    asyncio.run(lightrag_instances[kb_name].initialize_storages())

def generate(source, prompt, model,enable_thinking=False,enable_search=False,system_prompt=None):

    """
    调用API生成内容
    
    Args:
        source (str): API源，"ali"表示使用阿里源，其他值使用默认源
        prompt (str): 输入提示词
        enable_thinking (bool): 是否启用思考模式（仅在使用阿里源时有效）
        
    
    Returns:
        str: API返回的结果
    """
    message = [{"role": "user", "content": prompt}]
    if system_prompt:
        message.insert(0, {"role": "system", "content": system_prompt})
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
        messages=message,
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
            messages=message,
        )
    
    return response.choices[0].message.content


def evol_instruct(mode,given_prompt,model,enable_thinking=False):
    """
    指令进化函数
    """
    
    if mode == "deep":
        gen_prompt = df["prompt"][0]
    elif mode == "con":
        gen_prompt = df["prompt"][1]
    elif mode == "reason":
        gen_prompt = df["prompt"][2]
    
    gen_prompt = gen_prompt.replace("$Given_Prompt$",given_prompt)
    output = generate("ali", gen_prompt, model,enable_thinking)
    output = output_selecter(output,mode)
    logger.info(f"原始指令：{given_prompt}")
    logger.info(f"指令进化结果：{output}")
    return output

def instruct_checker(previous_instruct,after_instruct):#use flash model
    #df = pd.read_csv("./prompt.csv")
    prompt = df["prompt"][3]
    prompt = prompt.replace("$previous$",previous_instruct)
    prompt = prompt.replace("$after$",after_instruct)

    output = generate("ali",prompt,"qwen-flash")
    output = output_selecter(output,'instruct_check')
    logger.info(f"指令检查结果：{output}")
    return output

async def answer(instruction,system,output,model):
    cls = instruct_classifier(instruction)
    print(cls)
    if cls in ["car","digital_household","elec_car","food","household"]:

        addition_info = await load_existing_lightrag(cls,instruction,"global")
    else:
        raise Exception("Invalid work_dir")
    prompt = df["prompt"][5]
    prompt = prompt.replace("$question$",instruction)
    prompt = prompt.replace("$field$",addition_info)
    prompt = prompt.replace("$answer$",output)
    output = generate("ali",prompt,model,system_prompt=system)
    output = output_selecter(output,'answer')
    logger.info(f"答案生成结果：{output}")
    return output
    

def instruct_classifier(instruction):#TODO:训练个简易分类器或者使用快速模型进行判断
    prompt = df["prompt"][4]
    prompt = prompt.replace("$question$",instruction)
    output = generate("ali",prompt,"qwen-flash")
    output = output_selecter(output,'field_check')
    logger.info(f"指令分类结果：{output}")
    return output


def answer_checker(answer):#use flash model
    prompt = df["prompt"][6]
    prompt = prompt.replace("$answer$",answer)
    output = generate("ali",prompt,"qwen-flash")
    output = output_selecter(output,'answer_check')
    logger.info(f"答案检查结果：{output}")
    return output
    



async def load_existing_lightrag(work_dir,query:str,mode):
    # 使用预加载的 LightRAG 实例
    if work_dir not in lightrag_instances:
        raise Exception(f"Unsupported work_dir: {work_dir}")
    
    lightrag_instance = lightrag_instances[work_dir]
    
    # 使用已存在的 LightRAG 实例初始化 RAGAnything
    rag = RAGAnything(
        lightrag=lightrag_instance,
    )
    
    # 查询已存在的知识库
    result =  await rag.aquery(
        query,
        mode
    )
    print("查询结果:", result)

    return result

def output_selecter(output_text, prompt_type):
    """
    根据prompt类型使用正则表达式提取输出内容
    
    Args:
        output_text (str): 模型的原始输出文本
        prompt_type (str): prompt类型，对应csv中的type列
    
    Returns:
        str: 提取的内容，如果未匹配到则返回原输出
    """
    patterns = {
        'deep': r'<Rewritten Prompt>:\s*(.+)',
        'con': r'<Rewritten Prompt>:\s*(.+)',
        'reason': r'<Rewritten Prompt>:\s*(.+)',
        'instruct_check': r'<final>[：:]\s*(是|否)(?:\n\s*\Z|$)',
        'field_check':r'<final>[：:]\s*(.+?)(?:\n\s*\Z|$)',
        'answer':r'<回答>[：:]\s*(.+?)(?:\n\s*\Z|$)',
        'answer_check':r'<输出>[：:]\s*(是|否)(?:\n\s*\Z|$)'
    }
    
    if prompt_type in patterns:
        match = re.search(patterns[prompt_type], output_text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # 如果没有匹配到模式，返回原始输出
    return output_text
