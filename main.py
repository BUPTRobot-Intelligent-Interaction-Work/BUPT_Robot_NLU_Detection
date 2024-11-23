from flask import Flask, request, jsonify
from flask_cors import CORS
from urllib.parse import unquote
from LLM.test_class import LLM_model, messages
from scentence_similarity import SentenceSimilarity
import requests
from utils import detect_language
from translate import translate

model_name = 'quora-distilbert-multilingual'
sm = SentenceSimilarity(model_name)
model = LLM_model(api_key='xxx', messages=messages,
                      random_seed=42, model_name='qwen2-0.5b-instruct', max_tokens=120)

start_mode = False
chat_mode = False

now_place = '充电桩'

exhibit_sentence = ['蒙娜丽莎', '星夜', '向日葵', '呐喊']

exhibit_numbers = ['一号展品', '二号展品', '三号展品', '四号展品']

exhibit_intents = ['author', 'expression', 'classify', 'ask_for_way']

last_explain = ''

rough_intents = {
    'exhibit' : ['表达', '意义', '内涵', '主题', '分类', '种类', '类型', '形式', '导览', 
                 '参观', '指引', '路线',   '讲解', '展示', '展览', '展览品', '展览物',
                 ],
    
    'product_explain': ['功能','型号','生日','生产日期','品牌','能做什么','有什么用','机器人', '制造商'],
    
    'chat': ['聊天', '闲聊', '你好','天气','日期','星期','风景','酒店','笑话','吃','游戏','电影',
             '音乐','旅行','新闻','科技','摇滚','电影','博物馆','流行','好看','自我','名字'],
    
    'detect_human': ['有多少人', '人数'], # 人数检测
    
    'dance':['跳舞', '请你跳个舞'], # 跳舞
    
    'downhead': ['低头', '低下头', '低头一点', '低下头一点', '低头看看', '低下头看看', '低头一点看看', '低下头一点看看'], # 低头
    
    'uphead': ['抬头', '抬起头', '抬头一点', '抬起头一点', '抬头看看', '抬起头看看', '抬头一点看看', '抬起头一点看看'], # 抬头
    
    'detect_exhibit': ['检测展品', '识别展品', '展品识别', '展品检测' ], # 展品检测
    
    'detect_face': ['检测人脸', '识别人脸', '人脸识别', '人脸检测', '我是谁', '我是哪位', '我是谁呢', '我是哪位呢'], # 人脸检测
    
    'detect_action': ['检测动作', '识别动作', '动作识别', '动作检测', '我在做什么', '我在干什么', '我在做什么呢', '我在干什么呢'], # 动作检测
    
    'detect_hand': ['检测手势', '识别手势', '手势识别', '手势检测', '我在比什么', '我在比什么呢', '我在比什么呢', '我在比什么呢'], # 手势检测
    
    'detect_emotion': ['检测情绪', '识别情绪', '情绪识别', '情绪检测', '我现在的情绪', '我现在的心情', '我现在的情绪是什么', '我现在的心情是什么'], # 情绪检测
    
    'move': ['带我去','带我看','带我参观', '到','去','看看','参观'], # 导航
    
    'stop_talk': ['停止讲解', '停止说话', '不要说了', '不要讲了', '不要讲了', '不要讲了', '不要讲了', '不要讲了'],
    
    'start': ['开始', '启动'],
    'stop': ['停下','停止', '拜拜', '太吵了', '不要讲了'],
}




chat_start_list = ['聊聊', '来聊天吧', '和我聊聊', '和我说说', '和我闲聊', '和我聊聊天', '和我聊聊天吧']


advanced_intents = {
    'more_exhibit_information':['能告诉我更多关于这幅画的信息吗？','告诉我更多关于这幅画的背景故事。','这幅画的创作背景是什么？', '这幅画的历史意义是什么？'],
    'recommend_exhibit': ['能给我提供一个推荐的展品吗？'],
}


move_map = {
    '出发点':'充电桩',
    '门口':'充电桩',
    '入口':'充电桩',
    '出口':'充电桩',
    '蒙娜丽莎' : '一号展品',
    '星夜' : '二号展品',
    '向日葵' : '三号展品',
    '呐喊' : '四号展品'
}


face_map = {
    'face': '普通客户',
    'kezundi': '柯尊迪',
    'wuyuhao' : '吴宇浩',
    'xiaguoyang' : '夏国洋',
}


def exhibit_number_convert_name(question):
    global exhibit_numbers
    for i in range(len(exhibit_numbers)):
        if exhibit_numbers[i] in question:
            # 用展品名替换对应部分
            question = question.replace(exhibit_numbers[i], exhibit_sentence[i])
            return question
    return question


def rough_intents_detect(question):
    global rough_intents
    
    for key in rough_intents:
        for word in rough_intents[key]:
            if word in question:
                return key
    return 'chat'


def advanced_intents_detect(question):
    global advanced_intents
    # 使用sentence_transformers计算相似度
    similarities = {}
    results = {}
    for key in advanced_intents:
        results[key] = sm.get_most_similar_sentence(question, advanced_intents[key])
        similarities[key] = sm.get_similarity(question, results[key])
        
    max_similarity = max(similarities.values())
    print(max_similarity)
    if max_similarity > 0.9:
        for key in similarities:
            if similarities[key] == max_similarity:
                return key
    else:
        return 'exhibit'



ExplanatoryWords = {
    '蒙娜丽莎': '这幅肖像画展示了一位神秘微笑的女性，背景是意大利的风景。蒙娜丽莎的微笑被认为是神秘的，令人难以捉摸，象征了人类情感的复杂性和不可测性。达·芬奇通过细致入微的笔触和独特的明暗对比技术，使画作充满了生命力。',
    '星夜': '梵高的这幅画描绘了一个充满动感和涡旋的夜空，星星闪烁，天空似乎在旋转。作品表达了艺术家内心的挣扎和对自然力量的敬畏，同时也传递出一种超越现实的精神力量，象征着孤独与对未知的探索。',
    '向日葵': '这幅画描绘了一束向日葵，花瓣金黄，花蕊黑色，充满了生命力和活力。达·芬奇通过细腻的笔触和鲜艳的色彩，展现了向日葵的美丽和神秘，同时也表达了对自然的热爱和敬畏。',
    '呐喊': '《呐喊》是挪威艺术家爱德华·蒙克的代表作之一。这幅画描绘了一个面部扭曲、手捂耳朵的人物，背景是一片血红的晚霞。作品表达了艺术家内心的恐惧和绝望，同时也反映了现代社会的压力和焦虑。'
}


AuthorWords = {
    "蒙娜丽莎": "由达·芬奇在16世纪创作。",
    "星夜": "由梵高在19世纪末创作。",
    "向日葵": "由梵高在19世纪末创作。",
    "呐喊": "由爱德华·蒙克在19世纪末创作。",
}


ClassificationWords = {
    '蒙娜丽莎':'蒙娜丽莎的微笑是肖像画。',
    '星夜':'星夜是一幅油画。',
    '向日葵':'向日葵是一幅静物画。',
    '呐喊':'呐喊是一幅表现主义作品。',
}


visiting_status = {
    '蒙娜丽莎': False,
    '星夜': False,
    '向日葵': False,
    '呐喊': False,
}

exhibit_rasa_url = 'http://10.29.49.247:5000/model/parse'
product_explain_url = 'http://10.29.56.78:5000/webhooks/rest/webhook'  # 产品解释的url

human_url = 'http://10.29.49.247:8002/human_detection'

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)


def exhibit_question_process(question):
    global sm, exhibit_sentence, ExplanatoryWords, visiting_status, last_explain
    
    # ad_intent = advanced_intents_detect(question)
    # if ad_intent == 'more_exhibit_information' and last_explain != '':
    #     question = '请基于这个展品' + last_explain + question
    #     return chat_question_process(question)
    # elif ad_intent == 'recommend_exhibit':
    #     recommendation = ''
    #     for exhibit in exhibit_sentence:
    #         if not visiting_status[exhibit]:
    #             recommendation = exhibit
    #             break
    #     visiting_status[recommendation] = True
    #     return '我建议您先去看看' + recommendation + '。' + chat_question_process("请介绍下" + recommendation + "。")
    
    try:
        rasa_need = {
            'text': question
        }
        print(rasa_need)
        
        response = requests.post(exhibit_rasa_url, json=rasa_need)
        response = response.json()
        print(response)
        intent = response['intent']['name']
        
        if len(response['entities']) == 0:
            entity = question
        else:
            entity = response['entities'][0]['value']
    except:
        entity = question
        most_similar = sm.get_most_similar_sentence(entity, exhibit_sentence)
        similarity = sm.get_similarity(entity, most_similar)
        visiting_status[most_similar] = True
        last_explain = most_similar
        # return chat_question_process(question)
        intent = 'expression'
    
    most_similar = sm.get_most_similar_sentence(entity, exhibit_sentence)
    similarity = sm.get_similarity(entity, most_similar)
    
    if similarity < 0.2:
        return chat_question_process(question)
    
    visiting_status[most_similar] = True
    last_explain = most_similar
    response_words = ''
    
    if intent == 'expression':
        response_words = ExplanatoryWords[most_similar]
    elif intent == 'author':
        response_words = AuthorWords[most_similar]
    elif intent == 'classify':
        response_words = ClassificationWords[most_similar]
    elif intent == 'ask_for_way':
        response_words = '您好，请跟我来，我带您去看看。'
    
    # 检查是否所有展品都已经参观，若有未参观展品，则返回最靠前的未参观展品
    if intent != 'ask_for_way':
        recommendation = ''
        for exhibit in exhibit_sentence:
            if not visiting_status[exhibit]:
                recommendation = exhibit
                break
            
    if recommendation != '':
        response_words = response_words + '您仍有未参观的展品，我非常建议您先去看看' + recommendation + '。'
    else:
        response_words = response_words + '您已经参观完所有展品，祝您有一个愉快的参观体验！如果您还有其他问题或需要帮助，请随时告诉我。'

    print(most_similar)
    print(similarity)

    return response_words


def move_question_process(question):
    global move_map
    move_list = list(move_map.keys()) + list(move_map.values())
    query_list = ['带我去'+ x for x in move_list]
    most_similar = sm.get_most_similar_sentence(question, query_list)
    print(most_similar)
    
    
    simiilar_index = query_list.index(most_similar)
    most_similar = move_list[simiilar_index]
    print(most_similar)
    
    # similarity = sm.get_similarity(question, most_similar)
    # print(similarity)
    
    if most_similar in move_map:
        target = move_map[most_similar]
        if target != '充电桩':
            visiting_status[most_similar] = True
        return target
    else:
        return '请问您要去哪里？'




def chat_question_process(question):
    print(question)
    global model
    answer = model.call_with_message(question=question)
    print(answer)
    data = {
        "question": question,
        "answer": answer
    }
    return answer


def product_explain_question_process(question):
    
    data = {
        "message": question
    }
    
    response = requests.post(product_explain_url, json=data)
    response = response.json()
    print(response)
    return response[0]['text']


@app.route('/emotion', methods=['POST'])
def emotion_detect():
    question = request.json['question']
    # print(question)
    # response = chat_question_process(question)
    return question

@app.route('/head', methods=['POST'])
def head_detect():
    question = request.json['question']
    # print(question)
    # response = chat_question_process(question)
    return question


# @app.route('/human', methods=['POST'])
def human_detect():
    response = requests.post(url=human_url)
    response = response.json()
    num = response['num']
    return num

face_url = 'http://10.29.49.247:8002/face_detection'
action_url = 'http://10.29.49.247:8002/action_detection'
hand_url = 'http://10.29.49.247:8002/gesture_detection'
emotion_url = 'http://10.29.49.247:8002/emotion_detection'
exhibit_url = 'http://10.29.49.247:8002/exhibit_detection'

def face_detect():
    response = requests.post(url=face_url)
    response = response.json()
    if response['detection']:
        label = response['label']
        if label in face_map:
            response = face_map[label]
            return response

    return '我无法识别您的身份，请再试一次'

action_map = {
    'hugging': '拥抱',
    "sitting": '坐着',
    'using_laptop': '使用笔记本电脑',
    'clapping': '鼓掌',
    'drinking': '喝水',
    'calling': '打电话',
}


def action_detect():
    response = requests.post(url=action_url)
    response = response.json()
    if response['detection']:
        label = response['label']
        if label in action_map:
            response = action_map[label]
            return response
    
    
    return '您的动作我无法识别，请再试一次'


hand_map = {
    'one': '一号展品',
    'two': '二号展品',
    'three': '三号展品',
    'four': '四号展品',
    'like' : '充电桩'
}

def hand_detect():
    response = requests.post(url=hand_url)
    response = response.json()
    
    if response['detection']:
        label = response['label']
        
        if label in hand_map:
            response = hand_map[label]
            return response
    return '您的手势我无法识别，请再试一次'


emotion_map = {
    'happy': '开心',
    'sad': '伤心',
    'neutral': '平淡',
}

def emotion_detect():
    response = requests.post(url=emotion_url)
    response = response.json()
    if response['detection']:
        label = response['label']
        if label in emotion_map:
            response = emotion_map[label]
            if response == '开心':
                return '您看起来很开心' + chat_question_process('我很开心')
            elif response == '伤心':
                return '您看起来有点伤心' + chat_question_process('我很伤心')
            elif response == '平淡':
                return '您看起来没什么特别的情绪，需要我为你做什么吗'
    
    return '我无法识别您的情绪，请再试一次'

exhibit_map = {
    'monalisa' : '一号展品',
    'star_night' : '二号展品',
    'sunflower' : '三号展品',
    'scream' : '四号展品'
}


def exhibit_detect():
    response = requests.post(url=exhibit_url)
    response = response.json()
    
    if response['detection']:
        if response['label'] in exhibit_map:
            response = exhibit_map[response['label']]
            return response
    return '我无法识别展品，请再试一次'
    


@app.route('/qa', methods=['POST'])
def QA():
    question = request.json['question']
    
    question = exhibit_number_convert_name(question)
    print(question)
    lang = detect_language(question)
    print(lang)
    
    if lang == 'English':
        question = translate(question, 'en', 'zh')
    
    rough_intent = rough_intents_detect(question)
    print(rough_intent)
    global start_mode, chat_mode
    chat_words = sm.get_most_similar_sentence(question, chat_start_list)
    if sm.get_similarity(question, chat_words) > 0.9:
        # global chat_mode
        chat_mode = True
    
    
    response = ''
    if rough_intent == 'start':
        start_mode = True
        response = '已进入讲解模式'
        # return response
    elif rough_intent == 'stop':
        start_mode = False
        chat_mode = False
        response = '您好，已停止讲解'
        # return response
    global now_place
    if start_mode:
        if rough_intent == 'exhibit':
            response = exhibit_question_process(question)
        elif rough_intent == 'product_explain':
            response = product_explain_question_process(question)
        elif rough_intent == 'move':
            response = move_question_process(question)
            # if now_place == response:
            #     response = '我们已经在' + now_place + '了'
            #     return response + 'C'
            # else:
            #     now_place = response
        elif rough_intent == 'detect_hand':
            response = hand_detect()
            if response == '您的手势我无法识别，请再试一次':
                return response + 'C'
            return response + 'D'
        elif rough_intent == 'detect_exhibit':
            response = exhibit_detect()
            if response == '我无法识别展品，请再试一次':
                return response + 'C'
            return response + 'D'
        elif rough_intent == 'detect_action':
            response = action_detect()
            if response == '您的动作我无法识别，请再试一次':
                return response + 'C'
            return f'您正在{response}'+ 'C'
        elif rough_intent == 'detect_face':
            response = face_detect()
            if response == '我无法识别您的身份，请再试一次':
                return response + 'C'
            if response == '普通客户':
                return f'您好，由于您的身份是{response}，未在我们的数据库中，我不能得知您的信息' + 'C'
            return f'您是最贵的VIP客户：{response}， 来自北京邮电大学，有什么我可以帮您' + 'C'
        elif rough_intent == 'detect_emotion':
            response = emotion_detect()
            if response == '我无法识别您的情绪，请再试一次':
                return response + 'C'
            return response + 'C'
        
        elif rough_intent == 'dance':
            response = '那我就给大家跳支舞吧'
        elif rough_intent == 'detect_human':
            num = human_detect()
            response = f'这里有{num}名旅客'
        elif rough_intent == 'downhead':
            response = '好的，我低下头看看'
        elif rough_intent == 'uphead':
            response = '好的，我抬起头看看'
        elif rough_intent != 'start' and rough_intent != 'stop':
            if chat_mode:
                response = chat_question_process(question)
            
    if lang == 'English':
        response = translate(response, 'zh', 'en')
        
    if rough_intent == 'exhibit':
        response += 'A'
    elif rough_intent == 'product_explain':
        response += 'B'
    elif rough_intent == 'chat':
        response += 'C'
    elif rough_intent == 'move':
        response += 'D'
    elif rough_intent == 'detect_human':
        response += 'C'
    elif rough_intent == 'dance':
        response += 'E'
    elif rough_intent == 'downhead':
        response += 'F'
    elif rough_intent == 'uphead':
        response += 'G'    
    elif rough_intent == 'start' or rough_intent == 'stop':
        response += 'C'
    
    return response



if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False,processes=True)