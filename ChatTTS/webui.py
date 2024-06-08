from pywebio.input import input
from pywebio.output import put_text, put_file
from pywebio import start_server
from text_to_chat import text_to_speech
def web_ui():
    # 获取用户输入
    user_input = input("请输入要转换为语音的文本：")
    # 生成 TTS 音频
    # 转换 mp3 为 wav 格式，因为 pywebio 只支持播放 wav 文件

    # 展示文本和音频
    put_text("您输入的文本：")
    put_text(user_input)
    output_file_path = text_to_speech(user_input, "demo.mp4")
    content = open(output_file_path, 'rb').read()
    put_file("demo.mp4",content,"下载")



if __name__ == '__main__':
    start_server(web_ui, port=8080)
