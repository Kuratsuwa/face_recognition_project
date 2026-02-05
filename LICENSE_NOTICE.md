# License Notice / クレジット表記

このアプリケーション（Omokage）は、以下のオープンソースライブラリおよびアセットを使用して構築されています。

## 1. ソフトウェアのライセンス
Omokage 本体は MIT ライセンスの下で提供されます。

### 主要なライブラリ
- **face_recognition / dlib**: MIT / Boost Software License
- **MoviePy / OpenCV**: MIT / Apache 2.0 License
- **CustomTkinter**: MIT License
- **PyTorch / Transformers**: BSD / Apache 2.0 License

## 2. フォントのライセンス
- **Noto Sans JP**: [SIL Open Font License 1.1](https://scripts.sil.org/OFL)
  - Noto Sans JP は Google が提供するオープンソースフォントであり、再配布および商用利用が許可されています。

## 3. AI BGM生成のライセンス
- **Stable Audio Open 1.0**: [Stability AI Community License](https://stability.ai/license)
  - このアプリはBGM生成に Stable Audio Open を使用しています。
  - **商用利用について**: 年間の総収益が100万ドル（約1.5億円）未満の個人または団体であれば、本アプリを通じて生成されたBGMを伴う動画を広告収益のあるサイト等で利用することが許可されています。

### Hugging Face Tokenの入手方法
Stable Audio Open 1.0 は「Gated Model」であり、利用規約への同意と認証トークンが必要です。
1. [Hugging Face](https://huggingface.co/) でアカウントを作成（無料）。
2. [Hugging Face公式サイトのモデルページ](https://huggingface.co/stabilityai/stable-audio-open-1.0)にアクセスし、規約を確認して「Agree and access repository」をクリックしてアクセス許可を得ます。
3. [Settings > Access Tokens](https://huggingface.co/settings/tokens) から、新しいトークン（`Type: Read` で十分です）を作成し、コピーして本アプリの入力欄に貼り付けてください。

---
© 2024 Omokage Team / AI Memory Lane Project
