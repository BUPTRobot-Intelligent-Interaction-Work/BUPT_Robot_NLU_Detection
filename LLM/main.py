from test_class import LLM_model
from test_class import messages

if __name__ == '__main__':

    model = LLM_model(api_key='sk-0725a223a78d4c64a15dc46e77298400', messages=messages,
                      random_seed=42, model_name='qwen2-0.5b-instruct', max_tokens=120)
    answer = model.call_with_message('你好，你是谁？')
    print(answer)
    answer = model.call_with_message('给我简单介绍一下这幅叫蒙娜丽莎的画作吧。')
    print(answer)
    answer = model.call_with_message('华沙的双塔是谁的作品？')
    print(answer)
    answer = model.call_with_message('来聊聊最近的好看电影吧。')
    print(answer)
    answer = model.call_with_message('再见')
    print(answer)
    # messages = model.get_messages()
    # print(messages)
