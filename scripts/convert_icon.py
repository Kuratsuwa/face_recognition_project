from PIL import Image
import os
import sys

def convert_to_ico(input_path, output_path):
    print(f"Converting {input_path} to {output_path}...")
    img = Image.open(input_path)
    # Recommended sizes for Windows icons
    icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(output_path, format="ICO", sizes=icon_sizes)

def convert_to_icns(input_path, output_path):
    print(f"Converting {input_path} to {output_path}...")
    img = Image.open(input_path)
    # macOS icns supports these sizes
    # Pillow doesn't always handle icns well without helper tools, but we'll try basic save.
    # If this fails, we might need iconutil on mac.
    try:
        img.save(output_path, format="ICNS")
    except Exception as e:
        print(f"Pillow ICNS save failed: {e}. Attempting manual fallback if on Mac.")
        if sys.platform == 'darwin':
            # Create iconset directory
            iconset_dir = "assets/icon.iconset"
            os.makedirs(iconset_dir, exist_ok=True)
            # Create standard icon sizes
            for size in [16, 32, 64, 128, 256, 512, 1024]:
                s_img = img.resize((size, size), Image.Resampling.LANCZOS)
                s_img.save(f"{iconset_dir}/icon_{size}x{size}.png")
                # Also 2x retina versions
                if size <= 512:
                    s_img_retina = img.resize((size*2, size*2), Image.Resampling.LANCZOS)
                    s_img_retina.save(f"{iconset_dir}/icon_{size}x{size}@2x.png")
            
            # Use iconutil (standard macOS tool)
            os.system(f"iconutil -c icns {iconset_dir}")
            # Clean up
            import shutil
            if os.path.exists("assets/icon.icns"):
                shutil.move("assets/icon.icns", output_path)
            shutil.rmtree(iconset_dir)
            print("Successfully converted using iconutil.")

if __name__ == "__main__":
    assets_dir = "assets"
    input_png = os.path.join(assets_dir, "icon.png")
    
    if os.path.exists(input_png):
        convert_to_ico(input_png, os.path.join(assets_dir, "icon.ico"))
        convert_to_icns(input_png, os.path.join(assets_dir, "icon.icns"))
    else:
        print(f"Input file not found: {input_png}")
