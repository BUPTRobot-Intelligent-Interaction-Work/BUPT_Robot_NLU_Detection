# Intelligent Robot Interaction Lab 
# It's just a homework code!
# 北邮智能机器人交互实验作业 

## Report Docs
* [NLU Parts](https://bupt-robotteam.feishu.cn/docx/IypSdIzxXo6Q2gxLwMWcue6Bnrd?from=from_copylink) the second task
* [Detection Parts](https://bupt-robotteam.feishu.cn/docx/LUHwd8ptZoy5Ttxnlsmcg7t4nJg?from=from_copylink)  the third task

## Code
  This main branch is the core backend code of the entire project. It includes the two tasks related code.
  
  It provides basic functions as follows:
  * simple coarse-grain intention detection based on key-words
  * simple fine-grain intention detection based on sentence vectors extracted by bert models (compute their cosine similarity)
  * chat functions based on qwen api
  * calling interfaces of other 'edge-backends' which provide rasa framework and detection functions (you could see these in other branchs)

You could install the enviroments by (This is merely a reference!)
``` shell
pip install torch sentence-transformers flask flask-core
```

You could run the core backend by
```shell
 python main.py
```

May god bless you! 😘

## TODO

  
