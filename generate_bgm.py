import os
import random
import glob

def generate_bgm(vibe="穏やか", duration_seconds=30, output_dir="bgm", token=None, prompt_index=None, num_inference_steps=50, negative_prompt=None):
    """
    bgm/ ディレクトリからBGMファイルを選択し、そのパスを返す。
    AI生成機能は削除されました。
    """
    
    # プロジェクトルートの bgm/ ディレクトリを探す
    # このファイルの親ディレクトリにあると仮定
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bgm_source_dir = os.path.join(base_dir, "bgm")
    
    if not os.path.exists(bgm_source_dir):
        print(f"Warning: 'bgm' directory not found at {bgm_source_dir}")
        return False, None

    # 対応する拡張子
    extensions = ["*.wav", "*.mp3", "*.m4a"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(bgm_source_dir, ext)))
        
    if not files:
        print(f"Warning: No audio files found in {bgm_source_dir}")
        return False, None
        
    # Vibe名がファイル名に含まれているものを優先的に探す
    # (例: "穏やか.wav", "gentle_piano.wav" など)
    
    # 簡易的なマッピング (必要に応じて拡張)
    keywords = [vibe]
    if vibe == "穏やか": keywords.extend(["gentle", "calm", "relax"])
    elif vibe == "エネルギッシュ": keywords.extend(["energetic", "upbeat", "pop"])
    elif vibe == "感動的": keywords.extend(["emotional", "moving", "cinematic"])
    elif vibe == "かわいい": keywords.extend(["cute", "kawaii", "kids"])
    
    candidates = []
    for f in files:
        fname = os.path.basename(f).lower()
        for k in keywords:
            if k in fname:
                candidates.append(f)
                break
    
    selected_file = None
    if candidates:
        selected_file = random.choice(candidates)
        print(f"BGM selected based on vibe '{vibe}': {os.path.basename(selected_file)}")
    else:
        # マッチしない場合はランダム
        selected_file = random.choice(files)
        print(f"BGM selected randomly (no Vibe match for '{vibe}'): {os.path.basename(selected_file)}")
        
    return True, selected_file

if __name__ == "__main__":
    success, path = generate_bgm(vibe="穏やか")
    print(f"Result: {success}, Path: {path}")
