import torch

# Monkey patch for systems without xpu support to avoid AttributeError
if not hasattr(torch, "xpu"):
    class DummyXPU:
        def is_available(self):
            return False
        def empty_cache(self):
            pass
        def device_count(self):
            return 0
        def current_device(self):
            return "cpu"
        def synchronize(self):
            pass
        def __getattr__(self, name):
            # Fallback for any other attribute to prevent crash
            def method(*args, **kwargs):
                return None
            return method
    torch.xpu = DummyXPU()

import torch.distributed
if not hasattr(torch.distributed, "device_mesh"):
    class DummyDeviceMeshModule:
        class DeviceMesh:
            def __init__(self, *args, **kwargs):
                pass
    torch.distributed.device_mesh = DummyDeviceMeshModule()

import os

import scipy.io.wavfile as wavfile
import numpy as np
from datetime import datetime

# Stable Audio Open 1.0 は diffusers を使用します
# Note: このモデルはGatedなため、配布時はユーザーにHFトークンを入力してもらう必要があります。

def generate_bgm(vibe="穏やか", duration_seconds=30, output_dir="bgm", token=None):
    """
    Stable Audio Open 1.0 を使用してBGMを生成し、WAVとして保存する。
    """
    from diffusers import StableAudioPipeline
    
    # Vibeを英語プロンプトにマッピング
    vibe_prompts = {
        "穏やか": "beautifully melodic acoustic guitar and piano, lush ambient textures, warm and peaceful, emotionally evolving melody, rich harmonies, high quality, studio recording",
        "エネルギッシュ": "uplifting and catchy energetic pop music, bright synths and driving drums, melodic guitar hooks, high-energy arrangement, professional production, vibrant and happy",
        "感動的": "cinematic orchestral masterpiece, soaring expressive violin melody, rich emotional piano, dramatic shifts, powerful arrangement, evocative and soaring, high quality",
        "かわいい": "playful and whimsical upbeat melody, plucky strings and mallets, bright and cheerful, varied instrumentation, catchy hooks, bouncy and lighthearted, high quality"
    }

    # Vibeに応じたファイル名マッピング
    vibe_to_filename = {
        "穏やか": "calm.wav",
        "エネルギッシュ": "energetic.wav",
        "感動的": "emotional.wav",
        "かわいい": "cute.wav"
    }

    prompt = vibe_prompts.get(vibe, vibe_prompts["穏やか"])
    
    # 出力ファイル名の決定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = vibe_to_filename.get(vibe, "bgm.wav").replace(".wav", "")
    filename = f"{base_name}_{timestamp}.wav"
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    print(f"\n>>> AI BGM生成を開始します...")
    print(f"  雰囲気: {vibe}")
    print(f"  プロンプト: {prompt}")
    print(f"  長さ: {duration_seconds}秒 (Stable Audio制限: 最大47秒)")
    print(f"  出力先: {output_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Mac M1/M2/M3 の場合は mps を優先
    if torch.backends.mps.is_available():
        device = "mps"
    
    # mps は float16 または float32
    dtype = torch.float16 if device != "cpu" else torch.float32
    
    try:
        print(f"  モデルをロード中... ({device}, {dtype})")
        # モデルのロード
        # token が指定されている場合はそれを使用
        pipe = StableAudioPipeline.from_pretrained(
            "stabilityai/stable-audio-open-1.0", 
            torch_dtype=dtype,
            token=token
        )
        pipe = pipe.to(device)
        
        # 生成
        # Stable Audio Open 1.0 は最大47秒まで対応
        audio_duration = min(duration_seconds, 47.0)
        
        print(f"  音楽を生成中...")
        # 生成実行
        output = pipe(
            prompt, 
            num_inference_steps=50, 
            audio_end_in_s=audio_duration
        ).audios
        
        # output[0] は第一生成サンプル (channels, samples)
        # scipy.io.wavfile.write のために NumPy 配列に変換
        audio_data = output[0]
        if hasattr(audio_data, "cpu"):
            audio_data = audio_data.cpu().numpy()
        elif hasattr(audio_data, "numpy"):
            audio_data = audio_data.numpy()
        
        # 44.1kHz で保存
        # scipy.io.wavfile.write は float16 をサポートしていないため float32 に変換
        # また、(samples, channels) を期待するため転置
        audio_data_t = audio_data.T.astype(np.float32)
        
        wavfile.write(output_path, 44100, audio_data_t)
        
        print(f">>> BGM生成完了: {output_path}")
        return True, output_path
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during BGM generation: {e}")
        if "gated" in str(e).lower() or "not found" in str(e).lower() or "unauthorized" in str(e).lower():
            print("ヒント: このモデルには Hugging Face トークンが必要です。設定を確認してください。")
        return False, None

if __name__ == "__main__":
    import sys
    test_token = os.getenv("HF_TOKEN")
    success, _ = generate_bgm(vibe="穏やか", duration_seconds=10, token=test_token)
    sys.exit(0 if success else 1)
