#!/usr/bin/env python3
"""Voice & AI interface node for the drink robot.

Runs inside the conda environment (drink_robot).
Handles:
  - Wake-word detection (SIAO_MING.onnx via openwakeword)
  - YOLO person detection + depth-based approach trigger (yolov10n_HumanDetect.pt)
  - Speech-to-Text (Google Cloud Speech)
  - Text-to-Speech (Google Cloud TTS + pygame)
  - Gemini LLM brain for intent parsing and dialogue
  - Camera subscriptions (RealSense color + depth)

Publishes /user_command (std_msgs/String) when a command is resolved.
Subscribes /robot_status (std_msgs/String) for state awareness.

export GOOGLE_GEMINI_API_KEY="AIzaSyByHt3zz6UHiBNwcSKTWkni84o_0pY6oJo"
"""

from __future__ import annotations

import ctypes
import datetime
import io
import os
import threading
import time

import cv2
import numpy as np
import torch
import pyaudio
import openwakeword
import openwakeword.utils
import requests
import sounddevice as sd
from google import genai
from google.cloud import speech, texttospeech
from pydub import AudioSegment
from pymouth import VTSAdapter, DBAnalyser
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

# Suppress ALSA error messages
_ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
)


def _py_error_handler(filename, line, function, err, fmt):
    pass


_c_error_handler = _ERROR_HANDLER_FUNC(_py_error_handler)
try:
    _asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    _asound.snd_lib_error_set_handler(_c_error_handler)
except Exception:
    pass


class VoiceInterfaceNode(Node):
    """ROS 2 node providing voice and AI-driven interaction with the drink robot."""

    COLOR_TOPIC = '/arm/camera_right/realsense_camera_right/color/image_raw'
    DEPTH_TOPIC = '/arm/camera_right/realsense_camera_right/aligned_depth_to_color/image_raw'
    # COLOR_TOPIC = '/camera/realsense_camera_right/color/image_raw'
    # DEPTH_TOPIC = '/camera/realsense_camera_right/aligned_depth_to_color/image_raw'
    VISUAL_COOLDOWN = 1200.0
    DETECT_EVERY_N_FRAMES = 1

    def __init__(self):
        super().__init__('voice_interface_node')

        import google.genai as genai
        print(genai.__version__) # 確保新舊檔案版本一致

        pkg_share = get_package_share_directory('drinks_robot')
        wakeword_model = os.path.join(pkg_share, 'resource', 'WakeWord', 'mei.onnx')
        yolo_model_path = os.path.join(pkg_share, 'resource', 'yolov10n_HumanDetect.pt')
        google_cred = os.path.join(pkg_share, 'resource', 'google_credential.json')
        gemini_api_key = os.environ.get('GOOGLE_GEMINI_API_KEY', '')

        os.environ.setdefault('GOOGLE_APPLICATION_CREDENTIALS', google_cred)

        self._wakeword_model_path = wakeword_model
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._yolo = YOLO(yolo_model_path)
        self.gemini_client = genai.Client(api_key=gemini_api_key)

        openwakeword.utils.download_models()

        # State
        self._is_processing = False
        self._latest_depth: np.ndarray | None = None
        self._last_visual_trigger = 0.0
        self._frame_count = 0

        # 工具對照表
        self.available_functions = {
            "get_weather_internal": self.get_weather_internal,
            "select_drink": self.select_drink
        }
        # 在 __init__ 裡面
        self.tools = [
            {
                "function_declarations": [
                    {
                        "name": "get_weather_internal",
                        "description": "獲取指定城市的即時天氣資訊。",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "location": {"type": "STRING", "description": "城市名稱，例如 'Tainan'"}
                            },
                            "required": ["location"]
                        }
                    },
                    {
                        "name": "select_drink",
                        "description": "選擇飲料 (coffee/tea/water)。",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "drink_type": {"type": "STRING", "description": "飲料類型，例如 'coffee'"}
                            },
                            "required": ["drink_type"]
                        }
                    }
                ]
            }
        ]

        # ROS I/O
        self._drink_cmd_pub = self.create_publisher(String, '/drink_command', 10)
        self._status_sub = self.create_subscription(
            String, '/robot_status', self._status_callback, 10
        )

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._color_sub = self.create_subscription(
            Image, self.COLOR_TOPIC, self._color_callback, image_qos
        )
        self._depth_sub = self.create_subscription(
            Image, self.DEPTH_TOPIC, self._depth_callback, image_qos
        )

        # Start wake-word thread
        threading.Thread(target=self._wakeword_loop, daemon=True).start()
        self.get_logger().info('VoiceInterfaceNode started')

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _status_callback(self, msg: String):
        self.get_logger().info(f'robot status: {msg.data}')

    def _depth_callback(self, msg: Image):
        try:
            raw = np.frombuffer(msg.data, dtype=np.uint16)
            depth = raw.reshape((msg.height, msg.width))
            self._latest_depth = cv2.resize(depth, (1280, 720), interpolation=cv2.INTER_NEAREST)
        except Exception:
            pass

    def _color_callback(self, msg: Image):
        self._frame_count += 1
        if self._frame_count % self.DETECT_EVERY_N_FRAMES != 0:
            return
        if self._is_processing:
            return
        try:
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            if raw.size != msg.height * msg.width * 3:
                return
            img = raw.reshape((msg.height, msg.width, 3))
            if msg.encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (1280, 720))
            self._run_person_detection(img)
        except Exception as e:
            if 'Assertion failed' not in str(e):
                self.get_logger().error(f'color callback error: {e}')

    # ------------------------------------------------------------------
    # Person detection trigger
    # ------------------------------------------------------------------

    def _run_person_detection(self, img: np.ndarray):
        depth = self._latest_depth
        closest = 999.0
        results = self._yolo(img, device=self._device, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != 0:  # 0 = person
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                dist = 0.0
                if depth is not None:
                    try:
                        h, w = depth.shape
                        ry = int(cy * h / 720)
                        rx = int(cx * w / 1280)
                        if 0 <= ry < h and 0 <= rx < w:
                            dist = depth[ry, rx] / 1000.0
                    except Exception:
                        pass
                if 0 < dist < closest:
                    closest = dist

        now = time.time()
        if (
            0 < closest < 0.1
            and not self._is_processing
            and (now - self._last_visual_trigger > self.VISUAL_COOLDOWN)
        ):  
            self._is_processing = True
            self._last_visual_trigger = now
            threading.Thread(
                target=self._gemini_and_speak,
                args=('偵測到有人靠近。請主動溫馨地打招呼，詢問是否需要咖啡、水或茶。嚴禁提到餐點相關。', True),
                daemon=True,
            ).start()

    # ------------------------------------------------------------------
    # Wake-word loop
    # ------------------------------------------------------------------

    def _wakeword_loop(self):
        oww = openwakeword.Model(
            wakeword_models=[self._wakeword_model_path], inference_framework='onnx'
        )
        pa = pyaudio.PyAudio()
        mic = None
        while rclpy.ok():
            try:
                if self._is_processing:
                    if mic:
                        mic.stop_stream()
                        mic.close()
                        mic = None
                    time.sleep(0.5)
                    continue

                if mic is None:
                    mic = pa.open(
                        format=pyaudio.paInt16, channels=1, rate=16000,
                        input=True, frames_per_buffer=1280
                    )
                    self.get_logger().info('listening for wake word...')

                data = mic.read(1280, exception_on_overflow=False)
                audio_float = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                audio_float *= 2.0
                audio_frame = np.clip(audio_float, -32768, 32767).astype(np.int16)
                oww.predict(audio_frame)

                for _mdl, scores in oww.prediction_buffer.items():
                    if scores[-1] > 0.2:
                        self._is_processing = True
                        if mic:
                            mic.stop_stream()
                            mic.close()
                            mic = None
                        self._tts_and_play('在的，請說。')
                        self._handle_voice_interaction()
                        oww.reset()
                        break
            except Exception as e:
                self.get_logger().error(f'wake-word error: {e}')
                time.sleep(2)
        pa.terminate()

    # ------------------------------------------------------------------
    # Voice interaction
    # ------------------------------------------------------------------

    def _handle_voice_interaction(self):
        threading.Thread(target=self._voice_task, daemon=True).start()

    def _voice_task(self):
        try:
            should_continue = True
            while should_continue:
                
                # 1. 串流 STT，邊錄音邊回傳結果
                text = self._streaming_stt() 

                if text and text.strip():                    
                    # 2. 進入 Gemini 與 TTS 邏輯，並獲取是否繼續的指令
                    should_continue = self._gemini_and_speak(text)
                else:
                    should_continue = False
                
        except Exception as e:
            print(f"語音邏輯出錯: {e}")
        finally:
            self._is_processing = False

    def _publish_command(self, text: str):
        msg = String()
        msg.data = text
        self._cmd_pub.publish(msg)

    # ------------------------------------------------------------------
    # Gemini brain
    # ------------------------------------------------------------------

    def _gemini_and_speak(self, prompt: str, auto_listen: bool = True):
        now =time.time()
        response = self._gemini_brain(prompt)
        self.get_logger().info(f"Gemini 回應: {response}") # Debug 用
        self.get_logger().info(f"處理 Gemini 邏輯耗時: {time.time() - now:.2f} 秒") # Debug 用

        if not response or not response.strip():
            response = '我聽不太清楚，可以再說一次嗎？'

        self._tts_and_play(response)

        if auto_listen:
            self._voice_task()
        else:
            self._is_processing = False

    def _gemini_brain(self, user_input):

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
            # 1. 第一次請求：詢問意圖
            response = self.gemini_client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=[user_input],
                config={"tools": self.tools, "system_instruction": instruction}
            )

            # 2. 檢查是否有 function_call
            part = response.candidates[0].content.parts[0]
            if part.function_call:
                fn_name = part.function_call.name
                fn_args = part.function_call.args
                self.get_logger().info(f"執行函式：{fn_name}, 參數：{fn_args}")

                # 執行本地 Python 函式
                if fn_name in self.available_functions:
                    result = self.available_functions[fn_name](**fn_args)
                    
                    # 3. 第二次請求：把結果餵回給 Gemini 讓它組織溫暖的語言
                    response = self.gemini_client.models.generate_content(
                        model="gemini-3.1-flash-lite-preview",
                        contents=[
                            {"role": "user", "parts": [{"text": user_input}]},
                            {"role": "model", "parts": [part]}, # 剛才的 call
                            {
                                "role": "tool", # 我們的執行結果
                                "parts": [{"function_response": {"name": fn_name, "response": {"result": result}}}]
                            }
                        ],
                        config={"system_instruction": instruction}
                    )
                else:
                    return "抱歉，我暫時還不會這項技能。"

            # 4. 回傳最終的文字回應
            if response.text:
                return response.text
            return "抱歉，小美現在腦袋有點轉不過來，請再試一次。"

        except Exception as e:
            self.get_logger().error(f"Gemini API Error: {e}")
            return "抱歉，我的大腦連線發生了一點問題。"

    # ------------------------------------------------------------------
    # Gemini tools
    # ------------------------------------------------------------------

    def get_weather_internal(self, location: str):
        """
        獲取指定城市的即時天氣資訊。
        Args:
            location: 城市名稱，例如 'Tainan'
        """
        
        url = f"http://api.weatherapi.com/v1/current.json?key=040e24d691134027aa8115215262102&q={location}&aqi=no"
        try:
            res = requests.get(url).json()
            self.get_logger().info(f"* 天氣資訊：{res}，不須再次詢問是否需要飲料")
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
        self.get_logger().info(f"* 選擇飲料：{drink}")

        msg = String()
        msg.data = drink
        self._drink_cmd_pub.publish(msg)

        if drink in valid: return f"好的，將為您準備{valid[drink]}。告知使用者準備時間約1分鐘，詢問是否需要查天氣或是詢問要不要聽笑話嗎？嚴禁提到「已經為您準備」等確認完成的詞。"
        return f"抱歉，目前只有咖啡、水和茶，且僅告知中文選項。"


    # ------------------------------------------------------------------
    # Audio utilities
    # ------------------------------------------------------------------

    def _streaming_stt(self) -> str | None:
        """Streaming STT with 4-second window."""
        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code='zh-TW',
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config, interim_results=True
        )
        CHUNK = 1024
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16, channels=1, rate=16000,
            input=True, frames_per_buffer=CHUNK
        )
        self.get_logger().info("開始語音辨識，請說話...（超過3秒未說話將自動結束）")

        def request_generator():
            last_audio_time = time.time()
            
            for _ in range(0, int(16000 / CHUNK * 4)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                if time.time() - last_audio_time > 3.0:
                    self.get_logger().info("超過3.0秒未收到有效操作，自動結束語音辨識...")
                    break 
                    
                yield speech.StreamingRecognizeRequest(audio_content=data)

        responses = client.streaming_recognize(config=streaming_config, requests=request_generator())
        final_transcript = ''
        for response in responses:
            for result in response.results:
                if result.is_final:
                    final_transcript = result.alternatives[0].transcript
                    self.get_logger().info(f"辨識結果: {final_transcript}")

        stream.stop_stream()
        stream.close()
        p.terminate()
        return final_transcript if final_transcript else None

    def _tts_and_play(self, text: str):
        """TTS via Wavenet voice, played through VTSAdapter with mouth sync."""
        now = time.time()
        client = texttospeech.TextToSpeechClient()
        voice = texttospeech.VoiceSelectionParams(
            language_code='cmn-TW', name='cmn-TW-Wavenet-A'
        )
        audio_cfg = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            speaking_rate=1.2,
            pitch=5.0,
        )
        res = client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text), voice=voice, audio_config=audio_cfg
        )
        audio_segment = AudioSegment.from_wav(io.BytesIO(res.audio_content))
        audio_duration = audio_segment.duration_seconds
        wav_path = 'temp.wav'
        with open(wav_path, 'wb') as f:
            f.write(res.audio_content)

        self.get_logger().info(f"TTS 生成總耗時: {time.time() - now:.2f} 秒")
        now = time.time()
        try:
            target_ws = 'ws://100.79.190.70:8001' 
            with VTSAdapter(DBAnalyser(temperature=10),ws_uri=target_ws) as a:
                a.action(audio=wav_path, samplerate=44100, output_device=4)
                time.sleep(audio_duration)
        except Exception as e:
            self.get_logger().error(f'playback error: {e}')
        if os.path.exists(wav_path):
            os.remove(wav_path)
        self.get_logger().info(f"播放總耗時: {time.time() - now:.2f} 秒")

def main(args=None):
    rclpy.init(args=args)
    node = VoiceInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
