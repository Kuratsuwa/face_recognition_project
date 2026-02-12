import json
import os
import sys
from moviepy.editor import VideoFileClip, concatenate_videoclips, ColorClip, CompositeVideoClip, AudioFileClip, CompositeAudioClip, ImageClip
import cv2
import numpy as np
import pickle
import gc
import face_recognition
from datetime import datetime
import random
from utils import resource_path

def load_config(config_path='config.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def apply_blur(frame, target_encodings, blur_enabled):
    if not blur_enabled:
        return frame
    
    # 処理速度向上のため、検出用の画像をさらに縮小 (1/4サイズ)
    # 5s/it という極端な遅延を解消するため、精度より速度を優先
    scale = 0.25 
    inv_scale = 1.0 / scale
    
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    
    # モデルを明示的に 'hog' に指定 (cnnより圧倒的に速い)
    # Mac/CPU環境ではこれがボトルネックの大部分
    face_locations = face_recognition.face_locations(small_frame, model="hog")
    
    if not face_locations:
        return frame

    # 変更が必要な場合のみコピーを作成 (メモリ節約)
    processed_frame = frame.copy()
    
    # エンコーディングも縮小画像で行う
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)
    known_encodings = list(target_encodings.values())
    
    for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_enc, tolerance=0.5)
        if not any(matches):
            # 座標を元のサイズに復元
            t = int(top * inv_scale)
            r = int(right * inv_scale)
            b = int(bottom * inv_scale)
            l = int(left * inv_scale)
            
            face_region = processed_frame[t:b, l:r]
            if face_region.size == 0: continue
            
            # ぼかし処理 (カーネルサイズを簡略化)
            kw = max(1, (r - l) // 5 * 2 + 1)
            kh = max(1, (b - t) // 5 * 2 + 1)
            processed_frame[t:b, l:r] = cv2.GaussianBlur(face_region, (kw, kh), 20)
            
    return processed_frame

def apply_color_filter(frame, filter_type):
    if filter_type == "None":
        return frame
    
    img = frame.astype(np.float32)
    if filter_type == "Film":
        img = img * 1.1 - 10
        img[:,:,2] *= 1.1
    elif filter_type == "Sunset":
        img[:,:,0] *= 1.2
        img[:,:,1] *= 1.1
        img[:,:,2] *= 0.8
    elif filter_type == "Cinema":
        # 感動的: 高コントラスト、彩度抑えめ、シネマ階調
        img = img * 1.25 - 20
        avg = np.mean(img, axis=2, keepdims=True)
        img = img * 0.7 + avg * 0.3
    elif filter_type == "Nostalgic":
        # 穏やか: セピア/暖色寄り、低コントラスト、柔らかい
        img[:,:,0] *= 1.15
        img[:,:,1] *= 1.05
        img[:,:,2] *= 0.85
        img = img * 0.9 + 15
    elif filter_type == "Vivid":
        # 元気: 彩度アップ、くっきり
        img = (img - 128) * 1.3 + 128
        img *= 1.1
    elif filter_type == "Pastel":
        # かわいい: 明るい、ピンク/マゼンタ寄り、ソフト
        img = img * 0.6 + 90
        img[:,:,0] *= 1.15 # Red
        img[:,:,2] *= 1.10 # Blue
    
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def add_title_overlay(frame, title_text):
    if not title_text:
        return frame
        
    # RGB -> BGR for CV2
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    # 中央に少し大きめのテキスト
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.2
    thickness = 2
    
    # テキストサイズ取得して中央配置
    (tw, th), baseline = cv2.getTextSize(title_text, font, scale, thickness)
    text_x = (w - tw) // 2
    text_y = (h + th) // 2
    
    # 影 / 背景ボックス（半透明）
    overlay = img.copy()
    cv2.rectangle(overlay, (text_x-20, text_y-th-20), (text_x+tw+20, text_y+20), (0,0,0), -1)
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    
    # 白文字
    cv2.putText(img, title_text, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def add_date_overlay(frame, date_str):
    from PIL import Image, ImageDraw, ImageFont
    
    # Numpy -> PIL
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    
    # フォントロード (Mac標準のAvenir利用、なければデフォルト)
    try:
        # さりげなく: Avenir Next Regular, 40px
        font = ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", 40)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Avenir Next Condensed.ttc", 40)
        except:
            font = ImageFont.load_default()
    
    # 左下 (位置も少し調整)
    pos = (50, 630)
    
    # 影（薄く、細く）
    shadow_color = (0, 0, 0, 100) # RGBA
    
    # 影描画
    offset = 2
    draw.text((pos[0]+offset, pos[1]+offset), date_str, font=font, fill=(0,0,0))
    
    # 本体（白）
    draw.text(pos, date_str, font=font, fill=(255,255,255))
    
    # PIL -> Numpy
    return np.array(img_pil)

def create_title_card(title_text, subtitle_text="", duration=3.0, font_size=80):
    from PIL import Image, ImageDraw, ImageFont
    
    width, height = 1280, 720
    # 黒背景
    img_pil = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img_pil)
    
    # フォントロード (オープンソースのNoto Sans JPを使用)
    try:
        font_path = resource_path("assets/fonts/NotoSansJP-Bold.ttf")
        font_title = ImageFont.truetype(font_path, font_size)
        font_sub = ImageFont.truetype(font_path, 40)
    except:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
    
    # 中央揃え計算 (getbbox or textbbox handles new Pillow versions)
    def draw_centered(text, font, y_offset=0):
        if not text: return
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (width - text_w) // 2
        y = (height - text_h) // 2 + y_offset
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
        return y + text_h

    # タイトル描画
    draw_centered(title_text, font_title, y_offset=-20 if subtitle_text else 0)
    
    # サブタイトル描画
    if subtitle_text:
        draw_centered(subtitle_text, font_sub, y_offset=60)
    
    # ImageClip化
    return ImageClip(np.array(img_pil)).set_duration(duration).set_fps(24)

def render_documentary(playlist_path='story_playlist.json', config_path='config.json', output_dir='output', blur_enabled=None, filter_type=None, bgm_enabled=None):
    if not os.path.exists(playlist_path):
        print(f"Error: Playlist not found: {playlist_path}")
        return

    with open(playlist_path, 'r', encoding='utf-8') as f:
        playlist_data = json.load(f)
    
    # 新しい形式（dict）と古い形式（list）の両方に対応
    if isinstance(playlist_data, dict):
        playlist = playlist_data.get("clips", [])
        dominant_vibe = playlist_data.get("dominant_vibe", "穏やか")
    else:
        playlist = playlist_data
        dominant_vibe = "穏やか"

    config = load_config(config_path)
    # 引数、環境変数、Configの順で優先
    if blur_enabled is None:
        blur_enabled = str(os.environ.get("RENDER_BLUR", config.get("blur_enabled", False))).lower() in ("1", "true", "yes")
        
    if filter_type is None:
        filter_type = os.environ.get("RENDER_FILTER", config.get("color_filter", "None"))
    
    # BGMのVibeに合わせた自動フィルター設定
    if filter_type == "None" or filter_type is None:
        vibe_to_filter = {
            "感動的": "Cinema",
            "穏やか": "Nostalgic",
            "エネルギッシュ": "Vivid",
            "かわいい": "Pastel"
        }
        auto_filter = vibe_to_filter.get(dominant_vibe)
        if auto_filter:
            print(f"  Vibe ({dominant_vibe}) に基づきフィルター '{auto_filter}' を自動適用します")
            filter_type = auto_filter
    if bgm_enabled is None:
        bgm_enabled = str(os.environ.get("RENDER_BGM", "0")).lower() in ("1", "true", "yes")
    target_pkl = resource_path('target_faces.pkl')
    target_encodings = {}
    if blur_enabled:
        if os.path.exists(target_pkl):
            with open(target_pkl, 'rb') as f:
                target_encodings = pickle.load(f)
        else:
            if os.path.exists('target_faces.pkl'):
                with open('target_faces.pkl', 'rb') as f:
                    target_encodings = pickle.load(f)

    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f"documentary_{timestamp_str}.mp4")

    final_clips = []
    print(f"\n>>>> ドキュメンタリーをレンダリング中 (20 clips, 60s) <<<<")

    for i, item in enumerate(playlist):
        video_path = item["video_path"]
        best_t = item["t"]
        
        if not os.path.exists(video_path):
            print(f"  Warning: Video missing: {video_path}")
            continue

        try:
            video = VideoFileClip(video_path)
            duration = video.duration
            start = max(0, best_t - 1.5)
            end = min(duration, best_t + 1.5)
            
            print(f"  [{i+1}/20] Processing: {os.path.basename(video_path)} @ {best_t}s")
            clip = video.subclip(start, end)
            
            # --- Robust Normalization (1280x720 Fixed Canvas) ---
            # 1. 最初に回転を修正 (これが完了した時点で w と h が正しい向きに入れ替わる)
            if hasattr(clip, 'rotation') and clip.rotation != 0:
                clip = clip.rotate(clip.rotation)

            orig_w, orig_h = clip.size
            target_w, target_h = 1280, 720
            print(f"    [DEBUG] After Rotate: {orig_w}x{orig_h}, Ratio: {orig_w/orig_h:.3f}")

            # 2. 正しいアスペクト比を維持したまま 1280x720 に収める
            if (orig_w / orig_h) > (target_w / target_h):
                clip = clip.resize(width=target_w)
            else:
                clip = clip.resize(height=target_h)

            new_w, new_h = clip.size
            print(f"    [DEBUG] Resized: {new_w}x{new_h}, Final Ratio: {new_w/new_h:.3f}")

            # 3. 1280x720の黒背景の中央に配置 (レターボックス/ピラーボックス)
            clip = clip.on_color(size=(target_w, target_h), color=(0,0,0), pos="center")
            
            # 3. Synchronize FPS
            clip = clip.set_fps(24)
            
            # 4. Absolute Sync: Force audio sampling rate and match duration
            if clip.audio is not None:
                clip.audio = clip.audio.set_fps(44100)
            clip = clip.set_duration(3.0) # Explicitly set to intended duration
            
            # Apply filters
            if filter_type != "None":
                clip = clip.fl_image(lambda f: apply_color_filter(f, filter_type))
            
            # Apply blur
            if blur_enabled:
                clip = clip.fl_image(lambda f: apply_blur(f, target_encodings, blur_enabled))

            # Apply Date Overlay (timestamp)
            timestamp = item.get("timestamp", "")
            if timestamp:
                date_str = timestamp.split(" ")[0].replace("-", "/")
                clip = clip.fl_image(lambda f, ds=date_str: add_date_overlay(f, ds))

            # MoviePy 内部の参照を切るために明示的にコピーして追加（簡易的なヒント）
            final_clips.append(clip)
                
        except Exception as e:
            print(f"  Error processing {video_path}: {e}")
        
        # 定期的にGCを走らせてメモリ解放
        if i % 5 == 0:
            gc.collect()

    if not final_clips:
        print("Error: No clips to concatenate.")
        return

    # --- Add Opening and Ending ---
    # 人物名と期間を取得（プレイリストに含まれている場合）
    # 現状はプレイリストファイル自体にはメタデータが少ないため、簡易的に実装
    # 実際には create_story.py で metadata を保存するように改修するか、
    # クリップ情報から推測する。ここではシンプルに "Memory Documentary" とするか、
    # クリップがあればその期間を表示。
    
    period_str = ""
    if playlist:
        try:
            dates = [c.get("timestamp", "").split(" ")[0] for c in playlist if c.get("timestamp")]
            dates.sort()
            if dates:
                start_year = dates[0][:4]
                end_year = dates[-1][:4]
                if start_year == end_year:
                    period_str = start_year
                else:
                    period_str = f"{start_year} - {end_year}"
        except:
            pass
            
    # OP: Title + Period
    person_name = ""
    if isinstance(playlist_data, dict):
        person_name = playlist_data.get("person_name", "")
        
    if person_name:
        op_title = f"The Story of {person_name}"
    else:
        op_title = "Memory Documentary"
        
    op_clip = create_title_card(op_title, period_str, duration=3.0).fadein(1.0)
    
    # ED: To Be Continued...
    # ED: Randomized Text
    ed_texts = [
        "The Best is Yet to Come",
        "Life is a Journey",
        "Moments to Treasure",
        "Timeless Memories",
        "To Be Continued..."
    ]
    ed_text = random.choice(ed_texts)
    ed_clip = create_title_card(ed_text, "", duration=4.0, font_size=50).fadein(1.0).fadeout(1.0)
    
    # 結合: OP + Main + ED
    final_clips = [op_clip] + final_clips + [ed_clip]

    print(f"\nConcatenating {len(final_clips)} clips...")
    try:
        final_video = concatenate_videoclips(final_clips, method="compose")
        
        # BGMミキシング
        if bgm_enabled:
            # vibeに応じたプレフィックス (日本語と英語の両方をチェック)
            vibe_prefix_en = {
                "穏やか": "calm",
                "エネルギッシュ": "energetic",
                "感動的": "emotional",
                "かわいい": "cute"
            }
            
            target_en = vibe_prefix_en.get(dominant_vibe, "calm")
            target_jp = dominant_vibe
            
            # 候補ファイルを検索 (output/bgm を優先し、なければルートの bgm を見る)
            candidates = []
            bgm_search_dirs = [os.path.join(output_dir, "bgm"), "bgm"]
            
            # Unicode NFD/NFC問題への対策 (Mac)
            import unicodedata
            def normalize_path_str(s):
                return unicodedata.normalize('NFC', s)

            target_jp_norm = normalize_path_str(target_jp)
            target_en_norm = normalize_path_str(target_en)

            for d in bgm_search_dirs:
                if os.path.exists(d):
                    for f in os.listdir(d):
                        f_norm = normalize_path_str(f)
                        if (f_norm.startswith(target_jp_norm) or f_norm.startswith(target_en_norm)) and f_norm.endswith(".wav"):
                            candidates.append(os.path.join(d, f))
                if candidates: break # 優先ディレクトリで見つかれば終了
            
            if candidates:
                bgm_file = random.choice(candidates)
                print(f"\n>>> BGMをミックス中: {bgm_file} (vibe: {dominant_vibe})")
                try:
                    bgm_audio = AudioFileClip(bgm_file)
                    
                    # BGMを動画の長さに合わせる
                    video_duration = final_video.duration
                    is_special_vibe = dominant_vibe in ["感動的"]
                    
                    if is_special_vibe:
                        # 感動的: ループさせず、20秒以降かつ動画の終わりに合わせて配置
                        # 47秒のBGMが動画の最後に終わるように開始時間を計算
                        # ただし、開始は最低でも20秒後とする
                        bgm_start = max(20.0, video_duration - bgm_audio.duration)
                        bgm_audio = bgm_audio.set_start(bgm_start)
                        # 動画の長さを超える部分はカット
                        if bgm_start + bgm_audio.duration > video_duration:
                            bgm_audio = bgm_audio.subclip(0, video_duration - bgm_start)
                        print(f"  特殊配置適用 (穏やか/感動): 開始={bgm_start:.1f}s")
                    else:
                        # かわいい・元気: 必要に応じてループ
                        if bgm_audio.duration < video_duration:
                            # 既存のループ処理をベースにするが、47秒の素材を想定
                            crossfade_dur = min(3.0, bgm_audio.duration / 3) 
                            loop_audio = bgm_audio.audio_fadeout(crossfade_dur)
                            current_len = bgm_audio.duration
                            
                            while current_len < video_duration + crossfade_dur:
                                print(f"  BGMループを追加中... (現在: {current_len:.1f}s)")
                                next_segment = bgm_audio.audio_fadein(crossfade_dur).audio_fadeout(crossfade_dur)
                                start_time = current_len - crossfade_dur
                                loop_audio = CompositeAudioClip([loop_audio, next_segment.set_start(start_time)])
                                current_len = loop_audio.duration
                            
                            bgm_audio = loop_audio.subclip(0, video_duration)
                        else:
                            bgm_audio = bgm_audio.subclip(0, video_duration)
                    
                    # フェードアウト（最後の2秒）
                    bgm_audio = bgm_audio.audio_fadeout(2.0)
                    
                    # 元の音声とBGMをミックス（BGMは小さめに）
                    if final_video.audio is not None:
                        # 元の音声を保持しつつBGMを追加（BGMは30%の音量）
                        # volumexは副作用がないので新しいクリップを返す
                        mixed_audio = CompositeAudioClip([
                            final_video.audio.volumex(1.0), 
                            bgm_audio.volumex(0.3)
                        ])
                    else:
                        # 元の音声がない場合はBGMのみ
                        mixed_audio = bgm_audio.volumex(0.3)
                    
                    final_video = final_video.set_audio(mixed_audio)
                    print(f"  BGMミキシング完了")
                except Exception as e:
                    print(f"  BGMミキシングエラー: {e}")
                    print(f"  BGMなしで続行します...")
        
        # --- Absolute Stability: pix_fmt yuv420p, audio_fps, threads ---
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac', 
                                    fps=24, audio_fps=44100, threads=4,
                                    preset='ultrafast',
                                    ffmpeg_params=["-pix_fmt", "yuv420p"])
        print(f"\n>>> DOCUMENTARY GENERATED: {output_path}")
    except Exception as e:
        print(f"Error during concatenation: {e}")
    finally:
        print("\nCleaning up resources...")
        if 'final_video' in locals(): 
            try: final_video.close()
            except: pass
        
        # すべてのサブクリップを明示的に閉じる
        if 'final_clips' in locals():
            for c in final_clips:
                try: c.close()
                except: pass
        
        # 最後にメモリを強制解放
        gc.collect()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--blur", action="store_true")
    parser.add_argument("--no-blur", action="store_false", dest="blur")
    parser.add_argument("--filter", default="None")
    parser.add_argument("--bgm", action="store_true")
    parser.add_argument("--no-bgm", action="store_false", dest="bgm")
    args = parser.parse_args()

    # 環境変数にセットして render_documentary 内で参照
    os.environ["RENDER_BLUR"] = "1" if args.blur else "0"
    os.environ["RENDER_FILTER"] = args.filter
    os.environ["RENDER_BGM"] = "1" if args.bgm else "0"

    render_documentary()
