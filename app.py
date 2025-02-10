import os
import csv
import requests
import json
import base64
from datetime import datetime
import gradio as gr
import queue
import threading
from google.generativeai import GenerativeModel
from dotenv import load_dotenv

load_dotenv()

class PurewebuiChatbot:
    def __init__(self,
                 memory_file='chat_memory.csv',
                 upload_dir='uploads'):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        self.model = GenerativeModel('gemini-1.5-pro-latest', api_key=self.gemini_api_key)
        self.memory_file = memory_file
        self.upload_dir = upload_dir
        
        # สร้างโฟลเดอร์สำหรับเก็บไฟล์อัปโหลด (ถ้ายังไม่มี)
        os.makedirs(upload_dir, exist_ok=True)
        
        # ตรวจสอบและสร้างไฟล์เก็บประวัติ (ถ้ายังไม่มี)
        if not os.path.exists(memory_file):
            with open(memory_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'user_input', 'bot_response', 'files'])

    def process_image(self, image_path):
        """อ่านและเข้ารหัสภาพเป็น base64 เพื่อส่งไปประมวลผล"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_response(self, user_input, uploaded_files=None):
        """
        รับข้อความและไฟล์ (ถ้ามี) จากผู้ใช้
        สร้าง payload สำหรับส่งไปยัง API พร้อมกับข้อมูลโมเดล gemini-1.5-pro-latest
        แล้วคืนค่า bot_response
        """
        try:
            contents = []
            if uploaded_files:
                for file_path in uploaded_files:
                    image_data = open(file_path, 'rb').read()
                    contents.append({"mime_type": "image/jpeg", "data": image_data})

            contents.append(user_input)
            response = self.model.generate_content(contents)
            response.resolve()
            return response.text

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def save_interaction(self, user_input, bot_response, files=None):
        """บันทึกประวัติการสนทนาในรูปแบบ CSV"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_paths = ','.join(files) if files else ''
        with open(self.memory_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, user_input, bot_response, file_paths])

class PureAGIAgent:
    """
    ตัวแทน Agent ที่ใช้ queue และ threading
    เพื่อรับข้อความจากผู้ใช้และส่งต่อให้ PurewebuiChatbot ประมวลผล
    """
    def __init__(self, **kwargs):
        self.chatbot = PurewebuiChatbot(**kwargs)
        self.input_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self.agent_loop, daemon=True)
        self.thread.start()

    def agent_loop(self):
        while self.running:
            try:
                # รับข้อมูลจาก input_queue (blocking แบบมี timeout)
                user_input, files = self.input_queue.get(timeout=1)
            except queue.Empty:
                continue
            # ใช้ PurewebuiChatbot ในการประมวลผลข้อความ
            response = self.chatbot.generate_response(user_input, files)
            self.response_queue.put(response)

    def send_message(self, message, files=None):
        """
        ส่งข้อความและไฟล์ (ถ้ามี) ลงใน input_queue
        แล้วรอรับคำตอบจาก response_queue
        """
        self.input_queue.put((message, files))
        response = self.response_queue.get()  # รอคำตอบแบบ blocking
        return response

    def shutdown(self):
        """หยุดการทำงานของ agent และรอให้ thread จบการทำงาน"""
        self.running = False
        self.thread.join()

def launch_interface():
    """
    สร้าง Gradio Interface สำหรับรับส่งข้อความกับ Agent
    โดยใช้ PureAGIAgent ที่ทำงานเบื้องหลัง
    """
    agent = PureAGIAgent()

    def process_message(user_message, uploaded_files):
        file_paths = []
        if uploaded_files:
            for file in uploaded_files:
                # แก้ไขการเข้าถึงข้อมูลไฟล์ให้ใช้ file["name"] และ file["data"]
                file_path = os.path.join(agent.chatbot.upload_dir, file["name"])
                with open(file_path, "wb") as f:
                    f.write(file["data"])
                file_paths.append(file_path)
        response = agent.send_message(user_message, file_paths)
        # Delete uploaded files after processing
        for file_path in file_paths:
            os.remove(file_path)
        return response

    with gr.Blocks(theme=gr.themes.Soft()) as demo:  # เพิ่ม theme
        gr.Markdown("""
        # 🤖 PureAGI Chat Assistant
        Chat with AI and share images!
        """)
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot_ui = gr.Chatbot(
                    label="Chat History",
                    height=500,
                    show_label=True,
                    container=True,
                    show_copy_button=True
                )
                
                with gr.Row():
                    message_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here...",
                        scale=4
                    )
                    btn = gr.Button("Send 🚀", scale=1)
                
                file_upload = gr.File(
                    file_count="multiple",
                    type="file",
                    label="Upload Images 📸"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Chat Settings")
                clear_btn = gr.Button("Clear Chat 🗑️")
                
        def clear_chat(history):
            return []
        
        def on_submit(user_message, uploaded_files, history):
            if not user_message.strip() and not uploaded_files:
                return "", history
            response = process_message(user_message, uploaded_files)
            history.append((user_message, response))
            return "", history

        # Event handlers
        message_input.submit(on_submit, inputs=[message_input, file_upload, chatbot_ui], outputs=[message_input, chatbot_ui])
        btn.click(on_submit, inputs=[message_input, file_upload, chatbot_ui], outputs=[message_input, chatbot_ui])
        clear_btn.click(clear_chat, inputs=[chatbot_ui], outputs=[chatbot_ui])

    return demo

import asyncio
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

async def process_message(agent, user_message, uploaded_files):
    file_paths = []
    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(agent.chatbot.upload_dir, file.filename)
            try:
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                file_paths.append(file_path)
            except Exception as e:
                print(f"Error saving file: {e}")
                return f"Error saving file: {e}"
    response = agent.send_message(user_message, file_paths)
    # Delete uploaded files after processing
    for file_path in file_paths:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file: {e}")
    return response

@app.post("/api/chat")
async def chat_endpoint(request: Request, message: str = "", files: list[UploadFile] = File(None)):
    agent = PureAGIAgent()
    response = await process_message(agent, message, files)
    return JSONResponse({"response": response})

if __name__ == "__main__":
    import uvicorn
    demo = launch_interface()
    # demo.launch(share=True)

    # Run FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
