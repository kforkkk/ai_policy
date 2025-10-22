import utils
from utils import evol_instruct,instruct_checker,answer,answer_checker
import asyncio
import random
import json
import os

modes = ["deep","con","reason"]
json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), './data_gen/instruction_dataset.json')
async def evol_policy(iteration:int,check_point):
    with open(json_path,'r',encoding='utf-8') as f:
            dataset = json.load(f)
    new_dataset = dataset
    print("成功加载数据集")
    for i in range(iteration):
        rand = random.randint(0,2)
        mode = modes[rand]##决定当前进化模式
        accept = 0
        print("----------------------------------------\n")
        print(f"当前轮次为：{i}, 当前模式为：{mode}\n")
        
        for data in dataset:
            print(f"当前进度为：{dataset.index(data)/len(dataset)*100:.2f},模式为：{mode}\n")
            instruct = evol_instruct(mode,data["instruction"],"qwen3-max")#增强后的指令
            inst_check = instruct_checker(data["instruction"],instruct)#检查
            #print(inst_check)
            if inst_check=="是":
                ans = await answer(instruct,data["system"],data["output"],"qwen3-max")
                ans_check = answer_checker(ans)
                if ans_check=="是":
                    new_dataset.append({"instruction":instruct,"output":ans,"system":"你是一个擅长以旧换新政策解答的助手，你能够用清晰自然的语言解答政策相关问题的疑惑"})
                    accept += 1
                else:
                    continue
            else:
                continue
        print(f"此轮结束,接受的数据量为：{accept}，占比为：{accept/len(dataset)*100:.2f}\n")

    print("数据集生成完毕")
    manul_accept = input("是否手动选择数据？(y/n)")
    if manul_accept == "y":
        
        with open(check_point,'w',encoding='utf-8') as f:
            json.dump(new_dataset,f,ensure_ascii=False)
    print("数据集保存完毕")

if __name__ == "__main__":
    asyncio.run(evol_policy(2,"./instruction_dataset_evol.json"))
       
        



        

