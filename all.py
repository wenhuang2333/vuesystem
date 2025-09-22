
from  ENGtext import text
from sound import sound
from pic import picture
from flask_cors import CORS
from flask import Flask,request,send_file,render_template_string,jsonify
import re
from predict import brain
from CNtext import text_text
from datetime import date
import os
import glob
import asyncio
from googletrans import Translator, LANGUAGES
import json
import requests
from werkzeug.utils import secure_filename
os.makedirs('./daily/pic', exist_ok=True)
os.makedirs('./daily/sound', exist_ok=True)
app=Flask(__name__)
CORS(app)
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:5500"}})

@app.route('/text',methods=['POST'])
def text_predict():
    text_file=request.files['text_file']
    text_file.save("./source/text.txt")
    text_result=text("./source/text.txt")
    print(text_result)
    return render_template_string(text_result)

@app.route('/sound',methods=['POST'])
def sound_predict():
    sound_file=request.files['sound_file']
    sound_file.save("./source/sound.wav")
    sound_result=sound("./source/sound.wav")
    return render_template_string(sound_result)

@app.route('/pic',methods=['POST'])
def pic_predict():
    pic_file = request.files['pic_file']
    pic_result = picture(pic_file)
    return render_template_string(pic_result)

@app.route('/brain',methods=['POST'])
def brain_predict():
    brain_file = request.files['brain_file']
    brain_file.save("./source/brain.mat")
    brain_result = brain("./source/brain.mat")
    return render_template_string(brain_result)
def extract_sentences(data_text):
    content_without_tags = re.sub(r'<[^>]*>', '', data_text)
    sentences = re.split(r'[，。、？！；：,\.\?!:;]+', content_without_tags)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences


def check_file_in_pic(file_name):
    """
    检查在 './daily/pic' 文件夹中是否存在文件名（不含扩展名）与给定字符串相等的文件。
    如果存在，返回文件的相对路径；否则返回 None。
    """
    folder = './daily/pic'
    return check_file_in_folder(folder, file_name)

def check_file_in_sound(file_name):
    """
    检查在 './daily/sound' 文件夹中是否存在文件名（不含扩展名）与给定字符串相等的文件。
    如果存在，返回文件的相对路径；否则返回 None。
    """
    folder = './daily/sound'
    return check_file_in_folder(folder, file_name)

def check_file_in_folder(folder, file_name):
    """
    辅助函数，用于在指定文件夹中检查文件名（不含扩展名）是否与给定字符串相等。
    如果存在，返回文件的相对路径；否则返回 None。
    """
    try:
        files = os.listdir(folder)
    except FileNotFoundError:
        print(f"文件夹 {folder} 不存在")
        return None

    for file in files:
        name_without_extension = os.path.splitext(file)[0]
        if name_without_extension == file_name:
            return os.path.join(folder, file)

    return None
async def translator(arr):
    # 创建翻译器对象
    translator = Translator()
    allthings=''
    for i in range(len(arr)):
        allthings+=arr[i]
        allthings+='\n'
    # 翻译为英文
    translated = await translator.translate(allthings, src='zh-cn', dest='en')

    print(f"原文: {allthings}")
    print(f"翻译: {translated.text}")
    re=allthings.split('\n')
    k=re.pop()
    print("结果数组：",re)
    return re
@app.route('/daily', methods=['POST'])
def upload_daily():
    # 获取日期参数
    result = [0, 0, 0, 0, 0, 0, 0]
    real_pic_re = [0, 0, 0, 0, 0, 0, 0]
    text_re = [0, 0, 0, 0, 0, 0, 0]

    sound_num = {'平静': 6, '高兴': 0, '悲伤': 1, '愤怒': 2, '恐惧': 5, '厌恶': 4, '惊讶': 3}
    data_text = request.form.get('text')
    data_text = extract_sentences(data_text)
    print("文字的参数", data_text)
    if data_text:
        text_re = text_text(data_text)
    print("文字的结果", text_re)
    date_str = request.form.get('date')
    date_str = date_str.strip('"')
    soundpath = '' if check_file_in_sound(date_str)==None else check_file_in_sound(date_str)  # 注意这里调用的是 check_file_in_sound
    picpath = '' if check_file_in_pic(date_str)==None else  check_file_in_pic(date_str)    # 注意这里调用的是 check_file_in_pic
    if not date_str:
        return jsonify({'message': '未提供日期参数'}), 400

    files = request.files
    saved_files = {}

    # 定义允许的文件类型
    image_exts = ['.jpg', '.jpeg', '.png', '.gif']
    audio_exts = ['.mp3', '.wav']

    for file_key in files:
        file = files[file_key]
        print("当前文件", file.filename)
        if not file:
            continue

        _, file_extension = os.path.splitext(file.filename)
        file_extension = file_extension.lower()
        file_filename = secure_filename(file.filename)

        if file_extension in image_exts:
            # 目标目录
            target_dir = os.path.join('daily', 'pic')
            os.makedirs(target_dir, exist_ok=True)

            # 删除该日期下的所有图片（不管什么格式）
            for old_file in glob.glob(os.path.join(target_dir, f"{date_str}.*")):
                try:
                    os.remove(old_file)
                except Exception as e:
                    app.logger.error(f"删除文件时出错: {e}")

            # 保存新文件
            file_path = os.path.join(target_dir, f"{date_str}{file_extension}")
            file.save(file_path)
            saved_files[file_key] = os.path.abspath(file_path)
            picpath = file_path

        elif file_extension in audio_exts:
            target_dir = os.path.join('daily', 'sound')
            os.makedirs(target_dir, exist_ok=True)

            # 删除该日期下的所有音频（不管什么格式）
            for old_file in glob.glob(os.path.join(target_dir, f"{date_str}.*")):
                try:
                    os.remove(old_file)
                except Exception as e:
                    app.logger.error(f"删除文件时出错: {e}")

            # 保存新文件
            file_path = os.path.join(target_dir, f"{date_str}{file_extension}")
            soundpath = file_path
            print("音频的位置", file_path)
            saved_files[file_key] = os.path.abspath(file_path)
            file.save(file_path)

        else:
            return jsonify({'message': f'不支持的文件类型: {file_extension}'}), 400

    if os.path.exists(picpath):
        print("图片测试开始", picpath)
        pic_re = picture(picpath)
        if isinstance(pic_re, str):
            pic_re = json.loads(pic_re)
        print("pic_re如下", pic_re)
        real_pic_re[0] += pic_re['happy']
        real_pic_re[1] += pic_re['sad']
        real_pic_re[2] += pic_re['angry']
        real_pic_re[3] += pic_re['surprise']
        real_pic_re[4] += pic_re['disgust']
        real_pic_re[5] += pic_re['fear']
        real_pic_re[6] += pic_re['neutral']
        print("图片结果", real_pic_re)
        print("图片测试结束")

    if os.path.exists(soundpath):
        print("音频测试开始", soundpath)
        sound_re = sound(soundpath)
        result[sound_num[sound_re]] += 31.25
        print("音频后的re", result)
        print("音频测试结束")

    for i in range(len(result)):
        result[i] += text_re[i]
        result[i]+=real_pic_re[i]
        print(result[i])
    result=[round(i, 2) for i in result]
    print("result", result)
    return jsonify({'message': '文件上传成功', 'saved_files': saved_files, 'result': result})
@app.route('/clear', methods=['POST'])
def clear_files():
    # 获取前端发送的文件基本名称
    base_name = request.form.get('file_name')
    if not base_name:
        return jsonify({"error": "No file name provided"}), 400
    print(base_name)
    # 定义文件夹路径
    pic_folder = "./daily/pic"
    sound_folder = "./daily/sound"

    # 检查并删除文件
    try:
        # 检查图片文件夹
        for file in os.listdir(pic_folder):
            file_name, _ = os.path.splitext(file)
            if file_name==base_name:
                print("yes",file)
                file_path = os.path.join(pic_folder, file)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

        # 检查声音文件夹
        for file in os.listdir(sound_folder):
            file_name, _ = os.path.splitext(file)
            if file_name==base_name:
                file_path = os.path.join(sound_folder, file)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

        return jsonify({"message": "Files cleared successfully"}), 200

    except Exception as e:
        print("hello")
BASE_DIR_pic='./daily/pic'
@app.route('/getimg',methods=['POST'])
def get_imgfile():
    data = request.json
    date_str = data.get("date")  # 前端传过来的 "20250917" 之类的字符串
    # 匹配任意扩展名的文件，例如 20250917.png / 20250917.jpg / 20250917.gif ...
    pattern = os.path.join(BASE_DIR_pic, f"{date_str}.*")
    files = glob.glob(pattern)

    if files:
        file_path = files[0]  # 只取第一个匹配到的文件
        return send_file(file_path)
    else:
        return jsonify({"success": False})

BASE_DIR_sound='./daily/sound'
@app.route('/getsound',methods=['POST'])
def get_soundfile():
    data = request.json
    date_str = data.get("date")  # 前端传过来的 "20250917" 之类的字符串

    # 匹配任意扩展名的文件，例如 20250917.png / 20250917.jpg / 20250917.gif ...
    pattern = os.path.join(BASE_DIR_sound, f"{date_str}.*")
    files = glob.glob(pattern)

    if files:
        file_path = files[0]  # 只取第一个匹配到的文件
        return send_file(file_path)
    else:
        return jsonify({"success": False})




#AI
api_key = "Bearer kLCsxOpEfaZxCAiRmmmi:zAAQSsnkzQgVlpJhyIva"
url = "https://spark-api-open.xf-yun.com/v2/chat/completions"

# ---------------- 工具函数 ----------------
def get_answer(message):
    headers = {
        'Authorization': api_key,
        'content-type': "application/json"
    }
    body = {
        "model": "x1",
        "user": "user_id",
        "messages": message,
        "stream": True,
        "tools": [
            {
                "type": "web_search",
                "web_search": {
                    "enable": True,
                    "search_mode": "deep"
                }
            }
        ]
    }

    full_response = ""   # 存储最终AI回答
    reasoning_all = ""   # 存储完整思维链
    isFirstContent = True

    response = requests.post(url=url, json=body, headers=headers, stream=True)

    for chunks in response.iter_lines():
        if chunks and b'[DONE]' not in chunks:
            data_org = chunks[6:]  # 去掉 "data: " 前缀
            chunk = json.loads(data_org)
            text = chunk['choices'][0]['delta']

            # 拼接思维链
            if 'reasoning_content' in text and text['reasoning_content']:
                reasoning_all += text['reasoning_content']

            # 拼接模型回复
            if 'content' in text and text['content']:
                content = text['content']
                if isFirstContent:
                    isFirstContent = False
                full_response += content

    return full_response, reasoning_all


def getText(text, role, content):
    jsoncon = {"role": role, "content": content}
    text.append(jsoncon)
    return text


def getlength(text):
    return sum(len(c["content"]) for c in text)


def checklen(text):
    while getlength(text) > 11000:
        del text[0]
    return text
chatHistory = []  # 全局对话历史

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "")

        if not user_input:
            return jsonify({"error": "message不能为空"}), 400

        # 记录用户输入
        question = checklen(getText(chatHistory, "user", user_input))

        # 调用AI接口
        reply, reasoning = get_answer(question)

        # 记录AI回复
        getText(chatHistory, "assistant", reply)

        return jsonify({
            "reply": reply,
            "thinking": reasoning
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=5080)
    print("服务器正在运行，监听5080端口……")

