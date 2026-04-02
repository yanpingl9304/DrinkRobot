import os
import time
import datetime
import threading
import ctypes
import wave
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import message_filters

import pyaudio
from ultralytics import YOLO
from google import genai
from google.cloud import speech, texttospeech
import requests
from pygame import mixer
import torch
from PIL import Image as PILImage, ImageTk

import tkinter as tk
from tkinter import scrolledtext
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import openwakeword

# --- 屏蔽 ALSA 錯誤訊息 ---
ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
def py_error_handler(filename, line, function, err, fmt): pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
try:
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except: pass

# --- 全域參數設定 ---
GOOGLE_GEMINI_API_KEY = "YOUR_API_KEY_HERE"
OS_CREDENTIALS_PATH = "./resource/google_credential.json"
WAKEWORD_MODEL_PATH = "./resource/WakeWord/Aqua.onnx"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = OS_CREDENTIALS_PATH

openwakeword.utils.download_models()
CHINESE_FONT = "Noto Sans CJK TC" 

class DrinkRobotApp(Node):
    def __init__(self, root):
        super().__init__('drink_robot_brain')
        self.root = root
        self.root.title("drink robot")
        self.root.geometry("1600x900")
        
        # 1. 初始化狀態
        self.is_running = False
        self.is_processing = False 
        self.is_delivery = False
        self.last_visual_trigger_time = 0
        self.visual_cooldown = 80

        self.frame_count = 0
        self.detect_every_n_frames = 5
        self.last_ui_update_time = 0
        self.ui_update_interval = 0.05
        
        # 2. 模型初始化
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo_model = YOLO('./resource/yolov10m.pt')
        self.gemini_client = genai.Client(api_key=GOOGLE_GEMINI_API_KEY)
        
        # 3. ROS 2 通訊與影像轉換
        self.bridge = CvBridge()
        self.pub = self.create_publisher(String, '/goal_pose', 10)
        self.sub = self.create_subscription(Bool, '/navigation_complete', self.nav_callback, 10)
        self.subscription = self.create_subscription(String, '/goal_pose', self.goal_callback, 10)

        # 影像訂閱佔位符
        self.color_sub = None
        self.depth_sub = None
        self.ts = None
        
        # 工具對照表
        self.available_functions = {
            "get_weather_internal": self.get_weather_internal,
            "delivery_breakfast": self.delivery_breakfast,
            "select_drink": self.select_drink
        }
        self.tools = [self.get_weather_internal, self.delivery_breakfast, self.select_drink]

        # 4. UI 介面建置
        self._setup_ui()

        # 定時檢查 ROS
        self.root.after(100, self.ros_update)

    def goal_callback(self, msg: String):
        self.get_logger().info(f"Goal received: {msg.data}")

    def _setup_ui(self):
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=4, bg="#cccccc")
        self.paned.pack(fill="both", expand=True)

        self.video_frame = tk.Frame(self.paned, bg="black", width=1280, height=720)
        self.video_frame.pack_propagate(False)
        self.paned.add(self.video_frame, width=1280, stretch="always")
        
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

        self.right_frame = tk.Frame(self.paned, padx=10, pady=10)
        self.paned.add(self.right_frame, width=350, stretch="never")

        status_f = tk.Frame(self.right_frame)
        status_f.pack(fill="x")
        self.canvas = tk.Canvas(status_f, width=40, height=40)
        self.canvas.pack(side="left")
        self.status_light = self.canvas.create_oval(5, 5, 35, 35, fill="red")
        
        self.status_label = tk.Label(status_f, text="系統停止中", font=(CHINESE_FONT, 12))
        self.status_label.pack(side="left", padx=10)

        tk.Label(self.right_frame, text=":: 對話紀錄 ::", font=(CHINESE_FONT, 10, "bold")).pack(anchor="w", pady=(10,0))
        self.chat_display = scrolledtext.ScrolledText(self.right_frame, width=40, height=25, font=(CHINESE_FONT, 10))
        self.chat_display.pack(pady=5, fill="both", expand=True)

        btn_f = tk.Frame(self.right_frame)
        btn_f.pack(pady=10)
        self.start_btn = tk.Button(btn_f, text="啟動系統", command=self.start_all, width=12, height=2, bg="green", fg="white")
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = tk.Button(btn_f, text="停止系統", command=self.stop_all, width=12, height=2, bg="red", fg="white")
        self.stop_btn.pack(side="left", padx=5)

    def stop_all(self):
        self.is_running = False
        self.update_ui("系統已停止", "red")

    def ros_update(self):
        if rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)
        self.root.after(10, self.ros_update)

    def start_all(self):
        if not self.is_running:
            self.is_running = True
            self.update_ui("系統啟動中...", "green")

            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
            
            # 使用 BEST_EFFORT 確保能接收到 RealSense 的數據
            image_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=5
            )

            color_topic = '/arm/camera_right/realsense_camera_right/color/image_raw'
            depth_topic = '/arm/camera_right/realsense_camera_right/aligned_depth_to_color/image_raw'
            
            self.get_logger().info(f"正在啟動手動同步訂閱: {color_topic}")

            # 儲存最新的深度圖供彩色圖回呼使用
            self.latest_depth_img = None

            # 1. 訂閱深度圖 (只負責更新資料快取)
            self.depth_sub = self.create_subscription(
                Image, depth_topic, self.depth_callback, image_qos)

            # 2. 訂閱彩色圖 (主觸發：處理 YOLO、邏輯與 UI 更新)
            self.color_sub = self.create_subscription(
                Image, color_topic, self.video_callback, image_qos)

            threading.Thread(target=self.wakeword_thread, daemon=True).start()

    def depth_callback(self, msg):
        """ 接收深度圖並存入快取 """
        try:
            # 轉換並縮小深度圖，INTER_NEAREST 確保深度數值不失真
            raw_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            self.latest_depth_img = cv2.resize(raw_depth, (1280, 720), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            pass

    def video_callback(self, color_msg):
        """ 主回呼函式：彩色圖一到就跑邏輯 """
        if not self.is_running: return
        self.frame_count += 1
        
        try:
            # 1. 轉換並縮放彩色圖
            full_img = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            img = cv2.resize(full_img, (1280, 720))
            
            # 2. 獲取快取中的深度圖
            depth_img = self.latest_depth_img
            
            # 3. 抽幀執行 YOLO
            closest_dist = 999
            if self.frame_count % self.detect_every_n_frames == 0:
                results = self.yolo_model(img, device=self.device, verbose=False)
                
                for r in results:
                    for box in r.boxes:
                        if int(box.cls[0]) == 0: # Person
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            
                            dist = 0.0
                            if depth_img is not None:
                                if 0 <= cy < depth_img.shape[0] and 0 <= cx < depth_img.shape[1]:
                                    dist = depth_img[cy, cx] / 1000.0

                            if 0 < dist < closest_dist: closest_dist = dist
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, f"{dist:.2f}m", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            

                # 視覺觸發邏輯
                now = time.time()
                if 0 < closest_dist < 1.2 and not self.is_processing and (now - self.last_visual_trigger_time > self.visual_cooldown):
                    self.is_processing = True
                    self.last_visual_trigger_time = now
                    threading.Thread(target=self.process_gemini_and_speak, args=("偵測到有人靠近。請主動溫馨地打招呼，詢問是否需要咖啡、水或茶。嚴禁提到餐點相關。", True), daemon=True).start()

            # 4. 更新 UI
            current_time = time.time()
            if current_time - self.last_ui_update_time > self.ui_update_interval:
                self.last_ui_update_time = current_time
                display_img = img.copy()
                self.root.after(0, self._update_video_ui, display_img)
                
        except Exception as e:
            pass

    def _update_video_ui(self, img):
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = PILImage.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            # 這裡順序很重要
            self.video_label.config(image=img_tk)
            self.video_label.img_tk = img_tk  # 必須保留引用，否則影像會消失
        except Exception:
            pass

    def update_ui(self, text, color):
        self.status_label.config(text=text)
        self.canvas.itemconfig(self.status_light, fill=color)

    def log_chat(self, sender, msg):
        self.chat_display.insert(tk.END, f"[{sender}]: {msg}\n")
        self.chat_display.see(tk.END)

    def wakeword_thread(self):
        oww_model = openwakeword.Model(wakeword_models=[WAKEWORD_MODEL_PATH], inference_framework="onnx")
        pa = pyaudio.PyAudio()
        self.mic_stream = None
        while self.is_running:
            try:
                if self.is_processing:
                    if self.mic_stream:
                        self.mic_stream.stop_stream(); self.mic_stream.close(); self.mic_stream = None
                    time.sleep(0.5); continue

                if self.mic_stream is None:
                    self.mic_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1280)
                    self.root.after(0, lambda: self.update_ui("正在監聽 Aqua...", "green"))

                data = self.mic_stream.read(1280, exception_on_overflow=False)
                audio_frame = np.frombuffer(data, dtype=np.int16)
                print(f"當前音量: {np.abs(audio_frame).max()}", end='\r')
                oww_model.predict(audio_frame)

                for mdl, score in oww_model.prediction_buffer.items():
                    if score[-1] > 0.3:  # 喚醒詞觸發門檻，建議調整到 0.3-0.5 之間，視環境而定
                        self.is_processing = True
                        if self.mic_stream: self.mic_stream.stop_stream(); self.mic_stream.close(); self.mic_stream = None
                        print("Score:", score[-1])
                        self._tts_and_play("在的，請說。")
                        self.root.after(0, lambda: self.update_ui("聽取指令中...", "yellow"))
                        self.handle_voice_interaction()
                        oww_model.reset(); break
            except: time.sleep(0.2)
        pa.terminate()

    def handle_voice_interaction(self):
        threading.Thread(target=self._voice_logic_task).start()

    def _voice_logic_task(self):
        try:
            time.sleep(0.2) # 給硬體一點緩衝時間
            fname = "user_voice.wav"
            self._record_audio(fname, duration=4)
            text = self._stt(fname)
            
            if text:
                print(f"辨識結果: {text}") # Debug 用
                self.root.after(0, lambda: self.log_chat("User", text))
                self.process_gemini_and_speak(text)
            else:
                print("STT 辨識失敗或沒抓到聲音") # 如果跳這行，代表沒錄到音
                self.is_processing = False
                self.root.after(0, lambda: self.update_ui("監聽喚醒詞中", "green"))
        except Exception as e:
            print(f"語音邏輯出錯: {e}")
        self.is_processing = False

    def process_gemini_and_speak(self, prompt, auto_listen=True):
        response_text = self.gemini_brain(prompt)
        
        # 確保 response_text 不是 None 或空值
        if not response_text or response_text.strip() == "":
            response_text = "我聽不太清楚，可以再說一次嗎？"

        self.root.after(0, lambda: self.log_chat("Aqua", response_text))
        self._tts_and_play(response_text)
        
        # 檢查是否為天氣資訊 (報完天氣就結束)
        is_weather = any(k in response_text for k in ["度", "氣溫", "天氣", "下雨", "晴天"])
        
        should_listen = auto_listen or "？" in response_text or "?" in response_text

        if "祝您" in response_text or "請隨時告訴我喔" in response_text:
            should_listen = False

        print("Should Listen:", should_listen)
        
        if should_listen :
            self.root.after(0, lambda: self.update_ui("正在聽取您的回覆...", "yellow"))
            self._voice_logic_task()
        else:
            self.is_processing = False
            self.root.after(0, lambda: self.update_ui("監聽喚醒詞中", "green"))

    def gemini_brain(self, user_input):

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        week_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        week_day = week_list[datetime.datetime.now().weekday()]

        system_time = f"Current time is {current_time}，{week_day}。"
        
        instruction = f"""
        {system_time}
        ## ROLE
        你是一個溫暖貼心的繁體中文早餐機器人與送餐機器人與飲料機器人 Aqua，不負責點餐，僅負責執行指令與客人互動。

        ## STRICT RULES (PRIORITY: CRITICAL)
        1. **FUNCTION CALL FIRST**: 當使用者表達「送餐、送過來、就這樣、麻煩了」或「回到起始點」時，必須先執行 `delivery_service()`。
        2. **WEATHER TRIGGER**: 詢問天氣、氣溫或穿衣建議時，必須先執行 `get_weather_internal()`。
        3. **DRINK SERVICE**: 選擇飲料時執行 `select_drink()`。選完後主動問是否查天氣。
        4. **NO TEXT PREVIEW**: 工具執行前，不可對使用者做出任何承諾。
        5. **STATUS BOUNDARY**: 嚴禁提及「已送達」或「請享用」。接收送餐指令後，統一回覆：「已收到，準備送往 $X$ 號桌。」（$X$ 為桌號，若是 home 則回覆回到起始位置）。
        6. **END OF MISSION**: 報完天氣資訊（包含溫度、氣候）後，請直接給予暖心祝福並【停止詢問任何問題】。

        ## STEP-BY-STEP LOGIC
        Step 1: 偵測使用者意圖。
        Step 2: 涉及送餐、選飲或天氣時，【立即調用相關工具】，不准廢話。
        Step 3: 獲得工具回傳結果後，再根據結果回覆使用者。

        ## TRIGGER KEYWORDS
        - 送餐意圖："送餐"、"幫我送餐"、"送過來"、"點餐完畢"、"就這樣"、"麻煩外送"、"Ok 了"。
        - 天氣意圖："天氣"、"氣溫"、"台北天氣"、"台南氣溫"。
        """

        process_input = user_input
        if any(k in user_input for k in ["天氣", "氣溫", "溫度"]):
            process_input = f"【指令：請立即查詢天氣】{user_input}"

        try:
            # 1. 發送請求給 Gemini
            response = self.gemini_client.models.generate_content(
                model="gemini-3.1-flash-lite-preview" ,
                contents=[process_input],
                config={"tools": self.tools, "system_instruction": instruction}
            )

            return response.text if response.text else "抱歉，我現在無法處理這個請求。"

        except Exception as e:
            print(f"Gemini API Error: {e}")
            return "抱歉，我的大腦連線發生了一點問題。"

    def _record_audio(self, fname, duration):
        p = pyaudio.PyAudio()

        stream = p.open(format=pyaudio.paInt16, 
                        channels=1, 
                        rate=16000, 
                        input=True, 
                        frames_per_buffer=1024)
        
        frames = [stream.read(1024, exception_on_overflow=False) for _ in range(0, int(16000 / 1024 * duration))]
        stream.stop_stream()
        stream.close()
        p.terminate()

        with wave.open(fname, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(frames))

    def _stt(self, fname):
        client = speech.SpeechClient()
        with open(fname, "rb") as f: content = f.read()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                                        sample_rate_hertz=16000, 
                                        language_code="zh-TW")
        
        res = client.recognize(config=config, audio=audio)
        if os.path.exists(fname): os.remove(fname)
        for result in res.results: return result.alternatives[0].transcript
        return None

    def _tts_and_play(self, text):
        client = texttospeech.TextToSpeechClient()
        voice = texttospeech.VoiceSelectionParams(language_code="cmn-TW", name="cmn-TW-Standard-A")
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=1.5)
        res = client.synthesize_speech(input=texttospeech.SynthesisInput(text=text), voice=voice, audio_config=audio_config)
        with open("temp.mp3", "wb") as out: out.write(res.audio_content)
        self.update_ui("播放回應中", "lightblue")
        mixer.init(); mixer.music.load("temp.mp3"); mixer.music.play()
        while mixer.music.get_busy(): time.sleep(0.1)
        mixer.music.unload(); os.remove("temp.mp3")

    # --- 機器人工具函式 ---
    def get_weather_internal(self, location: str):
        """
        獲取指定城市的即時天氣資訊。
        Args:
            location: 城市名稱，例如 'Tainan'
        """
        
        url = f"http://api.weatherapi.com/v1/current.json?key=040e24d691134027aa8115215262102&q={location}&aqi=no"
        try:
            res = requests.get(url).json()
            print(f"* 天氣資訊：{res}，不須再次詢問是否需要飲料")
            return res 
        except Exception as e:
            return {"error": str(e)}
        
    def select_drink(self, drink_type: str):
        """
        選擇飲料 (coffee/tea/water)。
        Args:
            drink_type: 飲料類型，例如 'coffee'
        """
        valid = {"coffee": "咖啡", "tea": "茶", "water": "水"}
        drink = drink_type.lower()
        print(f"* 選擇飲料：{drink}")
        if drink in valid: return f"好的，已為您準備{valid[drink]}。對了，氣溫多變，需要我查天氣嗎？"
        return "抱歉，目前只有咖啡、水和茶。"

    def delivery_breakfast(self, location:str):
        """
        送早餐服務
        Args:
            location: 桌號，例如 5，或起始點：home
        """
        # 在真實情況下，這裡可能會呼叫外部 API 或內部系統來處理訂單
        print(f"* 收到送餐請求，桌號：{location}")


        msg = String()
        msg.data = location
        self.pub.publish(msg)
        self.get_logger().info(f"Navigation to {location}")

        # 4. 立即回傳結果給 Gemini，確保 API 不會逾時
        if location in ["1","2","3","4","5","6"]:
            self.is_delivery = True
            return f"已為 {location} 號桌的客人送上早餐！"
        elif location in ["home"]:
            return f"已為回到起始位置！"

    def nav_callback(self, msg):
        if msg.data and self.is_delivery:
            self.is_delivery = False
            self.process_gemini_and_speak("我已經到達目的地了，請享用您的飲料！")

def main():
    rclpy.init()
    root = tk.Tk()
    
    app = DrinkRobotApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        app.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
