import os
import sys
import io
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QTextEdit, QPushButton
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

import time
import datetime
import threading
import ctypes
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import pyaudio
from pymouth import VTSAdapter, DBAnalyser
import sounddevice as sd
from pydub import AudioSegment

from ultralytics import YOLO
from google import genai
from google.cloud import speech, texttospeech
import requests
import torch

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import openwakeword
from PyQt5.QtCore import QObject, pyqtSignal

# --- 屏蔽 ALSA 錯誤訊息 ---
ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
def py_error_handler(filename, line, function, err, fmt): pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
try:
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except: pass

# --- 全域參數設定 ---
GOOGLE_GEMINI_API_KEY = "api_key"
OS_CREDENTIALS_PATH = "./resource/google_credential.json"
WAKEWORD_MODEL_PATH = "./resource/WakeWord/mei.onnx"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = OS_CREDENTIALS_PATH

openwakeword.utils.download_models()

class UISignals(QObject):
    update_status = pyqtSignal(str, str)
    log_msg = pyqtSignal(str, str)
    update_video = pyqtSignal(object)

class DrinkRobotApp(Node):
    def __init__(self):
        super().__init__('drink_robot_brain')

        # --- 綁定跨執行緒 UI 訊號 ---
        self.signals = UISignals()
        self.signals.update_status.connect(self._real_update_ui)
        self.signals.log_msg.connect(self._real_log_chat)
        self.signals.update_video.connect(self._update_video_ui)
        
        # --- PyQt5 視窗初始化 ---
        self.window = QWidget()
        self.window.setWindowTitle("飲料機器人中控系統")
        self.window.resize(1600, 900)
        self.window.setStyleSheet("background-color: #f0f0f0;")
        
        # 1. 初始化狀態
        self.is_running = False
        self.is_processing = False 
        self.is_delivery = False
        self.last_visual_trigger_time = 0
        self.visual_cooldown = 120

        self.frame_count = 0
        self.detect_every_n_frames = 1
        self.last_ui_update_time = 0
        self.ui_update_interval = 0.05
        
        # 2. 模型初始化
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo_model = YOLO('./resource/yolov10m.pt')
        self.gemini_client = genai.Client(api_key=GOOGLE_GEMINI_API_KEY)
        
        # 3. ROS 2 通訊與影像轉換
        self.bridge = CvBridge()

        # 影像訂閱佔位符
        self.color_sub = None
        self.depth_sub = None
        self.ts = None
        
        # 工具對照表
        self.available_functions = {
            "get_weather_internal": self.get_weather_internal,
            "select_drink": self.select_drink
        }
        self.tools = [self.get_weather_internal, self.select_drink]

        # 4. UI 介面建置
        self._setup_ui()

        # 5. 定時檢查 ROS (使用 Qt 安全的 QTimer，取代 Tkinter 的 after)
        self.ros_timer = QTimer()
        self.ros_timer.timeout.connect(self.ros_update)
        self.ros_timer.start(10) # 10ms 執行一次 spin_once

        # 顯示視窗
        self.window.show()

        print("--- 真正支援播放的設備清單 ---")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                print(f"Index {i}: {dev['name']} | 最大聲道: {dev['max_output_channels']} | 默認採樣率: {dev['default_samplerate']}")

    def goal_callback(self, msg: String):
        self.get_logger().info(f"Goal received: {msg.data}")

    def _setup_ui(self):
        print("DEBUG: Setting up PyQt5 UI...")
        main_layout = QHBoxLayout(self.window)

        # 左側：影像顯示區
        self.video_label = QLabel("等待啟動系統...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 24px;")
        self.video_label.setMinimumSize(1000, 720)
        main_layout.addWidget(self.video_label, stretch=3)

        # 右側：控制面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, stretch=1)

        # 狀態燈號與文字
        status_layout = QHBoxLayout()
        self.status_light = QLabel()
        self.status_light.setFixedSize(30, 30)
        self.status_light.setStyleSheet("background-color: red; border-radius: 15px;") # 圓形燈號
        
        self.status_label = QLabel("系統停止中")
        self.status_label.setStyleSheet("font-size: 22px; font-weight: bold; color: black;")
        
        status_layout.addWidget(self.status_light)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        right_layout.addLayout(status_layout)

        # 對話紀錄
        chat_title = QLabel(":: 對話紀錄 ::")
        chat_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #333; margin-top: 15px;")
        right_layout.addWidget(chat_title)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("font-size: 16px; background-color: white; border: 1px solid #ccc;")
        right_layout.addWidget(self.chat_display)

        # 按鈕區
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("啟動系統")
        self.start_btn.setStyleSheet("font-size: 20px; padding: 15px; background-color: #28a745; color: white; font-weight: bold; border-radius: 5px;")
        self.start_btn.clicked.connect(self.start_all)
        
        self.stop_btn = QPushButton("停止系統")
        self.stop_btn.setStyleSheet("font-size: 20px; padding: 15px; background-color: #dc3545; color: white; font-weight: bold; border-radius: 5px;")
        self.stop_btn.clicked.connect(self.stop_all)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        right_layout.addLayout(btn_layout)

    def update_ui(self, text, color):
        self.signals.update_status.emit(text, color)

    def _real_update_ui(self, text, color):
        self.status_label.setText(text)
        self.status_light.setStyleSheet(f"background-color: {color}; border-radius: 15px;")

    def log_chat(self, sender, msg):
        self.signals.log_msg.emit(sender, msg)

    def _real_log_chat(self, sender, msg):
        self.chat_display.append(f"<b>[{sender}]</b>: {msg}")
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _update_video_ui(self, cv_img):
        """ 將 OpenCV 影像顯示在 PyQt 的 Label 上 """
        try:
            height, width, channel = cv_img.shape
            bytesPerLine = 3 * width
            q_img = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            
            pixmap = QPixmap.fromImage(q_img)
            
            if hasattr(self, 'video_label'):
                self.video_label.setPixmap(pixmap.scaled(
                    self.video_label.width(), 
                    self.video_label.height(), 
                    Qt.KeepAspectRatio
                ))
            else:
                print(f"找不到 label_vision，當前物件成員包含: {[m for m in dir(self) if 'label' in m.lower()]}")
                
        except Exception as e:
            print(f"PyQt 顯示更新失敗: {e}")

    def stop_all(self):
        self.is_running = False
        self.update_ui("系統已停止", "red")

    def ros_update(self):
        if rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)

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
            
            # color_topic = '/camera/realsense_camera_right/color/image_raw'
            # depth_topic = '/camera/realsense_camera_right/aligned_depth_to_color/image_raw'
            
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
            raw_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            self.latest_depth_img = cv2.resize(raw_depth, (1280, 720), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            pass

    def video_callback(self, color_msg):
        """ 
        主回呼函式：完全繞過 cv_bridge，直接從 ROS 原始位元組重建影像 
        """
        # 0. 基礎狀態檢查
        if not self.is_running: 
            return
        self.frame_count += 1
        
        try:
            # --- 1. 手動解碼：將 ROS byte 數據轉為 uint8 numpy 陣列 ---
            raw_data = np.frombuffer(color_msg.data, dtype=np.uint8)
            
            # 檢查數據量是否符合 (高 * 寬 * 通道數)
            expected_size = color_msg.height * color_msg.width * 3
            if raw_data.size != expected_size:
                # 如果資料長度不符，直接跳過，防止 cv2.resize 報錯
                return

            # --- 2. 重建影像矩陣 (Reshape) ---
            # 轉成 (H, W, C) 格式
            full_img = raw_data.reshape((color_msg.height, color_msg.width, 3))

            # --- 3. 顏色轉換 (Color Space Conversion) ---
            # ROS 預設通常是 RGB，OpenCV 運算與顯示則需要 BGR
            if color_msg.encoding == 'rgb8':
                img_bgr = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = full_img

            # --- 4. 縮放影像 (Resize) ---
            # 此時 img_bgr 保證是有效的 numpy array，resize 不會噴 Bad argument
            img = cv2.resize(img_bgr, (1280, 720))
            
            # --- 5. 獲取快取中的深度圖 ---
            depth_img = self.latest_depth_img
            closest_dist = 999.0
            
            # --- 6. 抽幀執行 YOLO 偵測 ---
            if self.frame_count % self.detect_every_n_frames == 0:
                # 執行 YOLO 模型推論
                results = self.yolo_model(img, device=self.device, verbose=False)
                
                for r in results:
                    for box in r.boxes:
                        if int(box.cls[0]) == 0:  # 類別 0 為 Person
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            
                            dist = 0.0
                            if depth_img is not None:
                                try:
                                    h, w = depth_img.shape[:2]
                                    real_cy = int(cy * h / 720)
                                    real_cx = int(cx * w / 1280)
                                    if 0 <= real_cy < h and 0 <= real_cx < w:
                                        dist = depth_img[real_cy, real_cx] / 1000.0
                                except:
                                    pass

                            if 0 < dist < closest_dist:
                                closest_dist = dist
                            
                            # 繪製偵測框與距離
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, f"{dist:.2f}m", (x1, y1-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # --- 7. 視覺觸發邏輯 ---
                now = time.time()
                if 0 < closest_dist < 1.2 and not self.is_processing and (now - self.last_visual_trigger_time > self.visual_cooldown):
                    self.is_processing = True
                    self.last_visual_trigger_time = now
                    threading.Thread(
                        target=self.process_gemini_and_speak, 
                        args=("偵測到有人靠近。請主動溫馨地打招呼，詢問是否需要咖啡、水或茶。嚴禁提到餐點相關。", True), 
                        daemon=True
                    ).start()

            current_time = time.time()
            if current_time - self.last_ui_update_time > self.ui_update_interval:
                self.last_ui_update_time = current_time
            
                self.signals.update_video.emit(img.copy())
                
        except Exception as e:
            if "Assertion failed" not in str(e):
                print(f"影像處理流程異常: {e}")

    def wakeword_thread(self):
        oww_model = openwakeword.Model(wakeword_models=[WAKEWORD_MODEL_PATH], inference_framework="onnx")
        pa = pyaudio.PyAudio()
        self.mic_stream = None
        while self.is_running:
            try:
                if self.is_processing:
                    if self.mic_stream:
                        self.mic_stream.stop_stream(); self.mic_stream.close(); self.mic_stream = None
                    time.sleep(0.5)
                    continue

                if self.mic_stream is None:
                    self.update_ui("嘗試開啟麥克風中...", "yellow") 
                    print("系統：準備呼叫 PyAudio 開啟硬體麥克風...")
                    
                    self.mic_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1280)
                    
                    self.update_ui("監聽喚醒詞中", "green")
                    print("系統：麥克風開啟成功！")

                data = self.mic_stream.read(1280, exception_on_overflow=False)
                
                # 1. 讀取並轉成 float32 進行計算
                audio_float = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                
                # 2. 放大音量 (增益)，讓喚醒詞更靈敏
                audio_float *= 2.0
                
                # 3. 限制數值範圍在 int16 之間（防止破音），再轉回 int16
                audio_frame = np.clip(audio_float, -32768, 32767).astype(np.int16)
                
                # 接下來再餵給模型
                print(f"當前音量 (增益後): {np.abs(audio_frame).max()}", end='\r')
                oww_model.predict(audio_frame)

                for mdl, score in oww_model.prediction_buffer.items():
                    if score[-1] > 0.20:  # 喚醒詞觸發門檻
                        self.is_processing = True
                        if self.mic_stream: self.mic_stream.stop_stream(); self.mic_stream.close(); self.mic_stream = None
                        print("\nScore:", score[-1])
                        self._tts_and_play("在的，請說。")
                        self.update_ui("聽取指令中...", "yellow")
                        self.handle_voice_interaction()
                        oww_model.reset(); break
            except Exception as e:
                print(f"\n🚨 麥克風錯誤原因: {e}") 
                self.update_ui("麥克風錯誤", "red")
                time.sleep(2)
        pa.terminate()

    def handle_voice_interaction(self):
        threading.Thread(target=self._voice_logic_task).start()

    def _voice_logic_task(self):
        try:
            self.update_ui("請說話...", "orange")
            
            # 1. 串流 STT，邊錄音邊回傳結果
            text = self._streaming_stt() 

            if text and text.strip():
                self.log_chat("User", text)
                
                # 2. 進入 Gemini 與 TTS 邏輯
                self.process_gemini_and_speak(text)
            else:
                print("未偵測到有效語音")
                self.update_ui("監聽中", "green")
                
        except Exception as e:
            print(f"語音邏輯出錯: {e}")
        finally:
            # 確保無論成功或失敗，狀態都會重置
            self.is_processing = False
            if not hasattr(self, 'playing_audio') or not self.playing_audio:
                self.update_ui("監聽喚醒詞中", "green")

    def process_gemini_and_speak(self, prompt, auto_listen=True):
        
        now =time.time()
        response_text = self.gemini_brain(prompt)
        print(f"Gemini 回應: {response_text}") # Debug 用
        print(f"處理 Gemini 邏輯耗時: {time.time() - now:.2f} 秒") # Debug 用

        if not response_text or response_text.strip() == "":
            response_text = "我聽不太清楚，可以再說一次嗎？"

        self._tts_and_play(response_text)
        
        if auto_listen:
            self.update_ui("正在聽取您的回覆...", "yellow")
            self._voice_logic_task()
        else:
            self.is_processing = False
            self.update_ui("監聽喚醒詞中", "green")

    def gemini_brain(self, user_input):

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        week_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        week_day = week_list[datetime.datetime.now().weekday()]

        system_time = f"Current time is {current_time}，{week_day}。"
        
        instruction = f"""
        {system_time}
        ## ROLE
        你是一個溫暖貼心的繁體中文飲料機器人 小美，不負責點餐，僅負責執行指令與客人互動。

        ## STRICT RULES (PRIORITY: CRITICAL)
        1. **WEATHER TRIGGER**: 當使用者詢問天氣、氣溫或穿衣建議時，必須先執行 `get_weather_internal()`。
        2. **DRINK SERVICE**: 選擇飲料時執行 `select_drink()`。選完後主動問是否查天氣或是詢問要不要聽笑話。
        3. **NO TEXT PREVIEW**: 工具執行前，不可對使用者做出任何承諾。
        4. **END OF MISSION**: 報完天氣資訊（包含溫度、氣候）後，請直接給予暖心祝福並【停止詢問任何問題】。
        5. **TONE AND STYLE**: 回答要溫暖、貼心，且帶有一點幽默感。嚴禁機械式回覆或提及自己是機器人及使用表情符號。

        ## STEP-BY-STEP LOGIC
        Step 1: 偵測使用者意圖。
        Step 2: 涉及選飲或天氣時，【立即調用相關工具】，不准廢話。
        Step 3: 獲得工具回傳結果後，再根據結果回覆使用者。

        ## TRIGGER KEYWORDS
        - 送餐意圖："送餐"、"幫我送餐"、"送過來"、"點餐完畢"、"就這樣"、"麻煩外送"、"Ok 了"。
        - 天氣意圖："天氣"、"氣溫"、"台北天氣"、"台南氣溫"。
        """

        try:
            # 1. 發送請求給 Gemini
            response = self.gemini_client.models.generate_content(
                model="gemini-3.1-flash-lite-preview" ,
                contents=[user_input],
                config={"tools": self.tools, "system_instruction": instruction}
            )

            return response.text if response.text else "抱歉，我現在無法處理這個請求。"

        except Exception as e:
            print(f"Gemini API Error: {e}")
            return "抱歉，我的大腦連線發生了一點問題。"



    def _streaming_stt(self):
        client = speech.SpeechClient()

        # 1. 設定辨識參數
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="zh-TW",
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config, 
            interim_results=True
        )

        # 2. 設定錄音串流
        CHUNK = 1024
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                        input=True, frames_per_buffer=CHUNK)

        print("請說話...")

        def request_generator():
            last_audio_time = time.time()
            
            for _ in range(0, int(16000 / CHUNK * 4)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                if time.time() - last_audio_time > 2.0:
                    print("🚨 超過2秒未收到有效操作，自動結束...")
                    break 
                    
                yield speech.StreamingRecognizeRequest(audio_content=data)

        # 3. 開始串流辨識
        responses = client.streaming_recognize(config=streaming_config, requests=request_generator())

        final_transcript = ""
        
        # 4. 處理回傳結果
        for response in responses:
            for result in response.results:
                if result.is_final:
                    final_transcript = result.alternatives[0].transcript
                    print(f"辨識結果: {final_transcript}")
                    
        # 關閉資源
        stream.stop_stream()
        stream.close()
        p.terminate()

        return final_transcript if final_transcript else None

    def _tts_and_play(self, text):
        now = time.time()
        client = texttospeech.TextToSpeechClient()
        voice = texttospeech.VoiceSelectionParams(language_code="cmn-TW", name="cmn-TW-Wavenet-A")
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16, 
            sample_rate_hertz=44100,
            speaking_rate=1.2,
            pitch = 5.0
        )
        
        res = client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text), 
            voice=voice, 
            audio_config=audio_config
        )
        
        self.log_chat("小美", text)
        self.update_ui("播放回應中", "lightblue")

        # 2. 在記憶體處理
        audio_data = io.BytesIO(res.audio_content)
        
        # 3. 計算播放時間
        audio_segment = AudioSegment.from_wav(audio_data)
        audio_duration = audio_segment.duration_seconds
        
        wav_path = "temp.wav"
        with open(wav_path, "wb") as f:
            f.write(res.audio_content)

        print(f"TTS 生成總耗時: {time.time() - now:.2f} 秒")
        now = time.time()
        try:
        # 4. 執行播放
            # Vtuber Studio 跑在本地 不用打 ws_uri=target_ws
            # output_device 看自己電腦的播放設備清單
            # target_ws = 'ws://100.79.190.70:8001' 
            # temperature 可以改開口大小
            with VTSAdapter(DBAnalyser(temperature=10), ws_uri=target_ws) as a:
                a.action(audio=wav_path, samplerate=44100, output_device=5)
                time.sleep(audio_duration+0.65)# 緩衝
        except Exception as e:
            print(f"播放錯誤: {e}")

        # 5. 清理臨時檔
        if os.path.exists(wav_path):
            os.remove(wav_path)
        
        print(f"播放總耗時: {time.time() - now:.2f} 秒")
        
        

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

        
        if drink in valid: return f"好的，將為您準備{valid[drink]}。告知使用者準備時間約2分鐘，詢問是否需要查天氣或是詢問要不要聽笑話嗎？嚴禁提到「已經為您準備」等確認完成的詞。"
        return f"抱歉，目前只有咖啡、水和茶，且僅告知中文選項。"

def main():
    rclpy.init()
    app = QApplication(sys.argv)
    
    robot_node = DrinkRobotApp()
    
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass
    finally:
        robot_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
