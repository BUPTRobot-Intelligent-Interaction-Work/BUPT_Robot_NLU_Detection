import requests
import random
import json
from hashlib import md5

class Translate(object):
    def __init__(self, appid: str, appkey: str, support_langs: list = ['en', 'zh'], auto_lang: bool = True) -> None:
        self.appid = appid
        self.appkey = appkey
        self.support_langs = support_langs
        self.auto_lang = auto_lang
        self.url = "http://api.fanyi.baidu.com/api/trans/vip/translate"
        self.headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    def _get_payload(self, query: str, from_lang: str, to_lang: str) -> dict:
        salt = random.randint(32768, 65536)
        sign = md5((self.appid + query + str(salt) + self.appkey).encode('utf-8')).hexdigest()
        return {'appid': self.appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}
    
    def translate(self, query: str, from_lang: str, to_lang: str) -> str:
        if from_lang not in self.support_langs and from_lang != 'auto':
            return None
        if to_lang not in self.support_langs:
            return None
        
        payload = self._get_payload(query, from_lang, to_lang)
        r = requests.post(self.url, params=payload, headers=self.headers)
        result = r.json()
        return "".join([i["dst"] for i in result["trans_result"]])


def translate(question, fr, to):
    appid = '20240305001983277'
    appkey = 'S84pQi6Ttex96vINzLIz'
    t = Translate(appid, appkey)
    return t.translate(question, fr, to)

# if __name__ == '__main__':
#     main()
