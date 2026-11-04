# 集中度推定AIの作成

本プロジェクトは、PCのWebカメラ映像から顔ランドマーク運動および非接触脈波（rPPG）をリアルタイム計測・解析し、作業者・学習者の集中度を推定するAI及びそのシステムを作成するためのものです。

---

## 主要な技術スタック

### フロントエンド(`data_collect`, `tetris`)

![Next.js](https://img.shields.io/badge/Next.js-000000?style=flat-square&logo=next.js&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)
![Unity](https://img.shields.io/badge/Unity-100000?style=flat-square&logo=unity&logoColor=white)

### バックエンド・生体信号解析パイプライン (`preprocess`, `validation`)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-DE5FE9?style=flat-square&logo=astral&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-007FFF?style=flat-square&logo=google&logoColor=white)
![NeuroKit2](https://img.shields.io/badge/NeuroKit2-8A2BE2?style=flat-square)

### 機械学習・深層学習モデル(`learning`)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat-square)

### クラウド・インフラ・環境

![Vercel](https://img.shields.io/badge/Vercel-000000?style=flat-square&logo=vercel&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Google Cloud Run](https://img.shields.io/badge/Google_Cloud_Run-4285F4?style=flat-square&logo=googlecloud&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-DD2C00?style=flat-square&logo=firebase&logoColor=white)

### AI

![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75C2?style=flat-square&logo=googlegemini&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-191919?style=flat-square&logo=anthropic&logoColor=white)
![GitHub Copilot](https://img.shields.io/badge/GitHub%20Copilot-000000?style=flat-square&logo=githubcopilot&logoColor=white)

---

## 全体アーキテクチャ

システム全体の処理フローは以下の構成になっています。

```text
mermaidで記述
```

---

## 主要モジュール・ディレクトリ構成

### data_collect（データ収集フロントエンド）

実験被験者のWebカメラ映像からリアルタイムに MediaPipe FaceLandmarker（478点）および顔頬部 ROI の平均 RGB 輝度値をミリ秒精度で取得し、gzip 圧縮して Firebase Storage へ自動保存する Next.js アプリケーションです。

- **詳細:** [`data_collect/README.md`](./data_collect/README.md) 参照

### preprocess（生体信号前処理・特徴量抽出）

Firebase から収集データを自動取得・解凍し、ランドマーク 11 工程および RGB 7 工程の前処理を実行します。POS アルゴリズムによる脈波抽出、瞬時 BPM 平滑化（5秒移動平均）、放物線補間・Kubios 外れ値補正を経た HRV 4 大指標（`mean_pp`, `rmssd`, `sdnn`, `sd2_sd1`）の算出を行い、機械学習用特徴量テーブルおよび時系列スライディングウィンドウ（5s / 10s / 15s）を生成します。

- **詳細:** [`preprocess/README.md`](./preprocess/README.md) 参照

### learning（機械学習・深層学習モデル）

`preprocess` で生成された多変量時系列特徴量を入力とし、TS2Vec（自己教師あり表現学習）、被験者別 Z スコア正規化、Fold 内 GMM クラスタリング、84 次元の特徴量融合（Feature Fusion: 84次元）、および 6 種の分類・回帰モデル（Ridge, ElasticNet, Random Forest, LightGBM, XGBoost, CatBoost）による **LOSO-CV（Leave-One-Subject-Out Cross-Validation）** を一気通貫で実行します。

- **詳細:** [`learning/README.md`](./learning/README.md) 参照

### validation（精度評価ベンチマーク）

公開データセット（UBFC-rPPG 等）の顔動画に対して本プロジェクトの rPPG アルゴリズムを適用し、接触型生体センサの Ground Truth に対する心拍数（HR）推定精度（MAE, RMSE, $r$）および HRV 各種指標の誤差・追従性を客観的・定量的に評価・検証します。

- **詳細:** [`validation/README.md`](./validation/README.md) 参照

### eda（探索的データ解析）

前処理前後の生体信号・特徴量の特性を可視化・検証します。`eda/raw/` では欠損値分布・サンプリングレート・生波形ノイズ・個人差ベースライン・クラスバランスを分析し、`eda/processed/` では前処理 Before/After 比較・特徴量と目的変数（ラベル）の相関分析・多重共線性（相関ヒートマップ・VIF）を出力します。

---

## 環境構築 (uv による依存関係一元管理)

本プロジェクトでは、高速かつ再現性の高い Python パッケージマネージャー **`uv`** を採用し、ルート直下の `.venv` および [`pyproject.toml`](./pyproject.toml) で一元管理しています。

### 1. 前提要件

- Python 3.12 または 3.13
- [uv](https://docs.astral.sh/uv/)（未インストールの場合は下記コマンドで導入）

```bash
# uv のインストール (Linux / WSL / macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. リポジトリのクローン & 環境同期

```bash
git clone <repository_url> make_ai
cd make_ai

# 全依存関係の同期（本番 + dev依存: validation, eda を含む）
uv sync

# 本番実行用のみ同期する場合（--no-dev で preprocess & learning のみ）
uv sync --no-dev
```

> [!NOTE]
> `pyproject.toml` では、本番運用に必要なコアライブラリ（`preprocess`, `learning`）と、開発・検証用ライブラリ（`validation`, `eda`）が `--group dev`（`[dependency-groups] dev`）として明確に分離管理されています。

---

## 実行方法 (uv run)

`uv run` を使用することで、仮想環境のアクティベート（`source .venv/bin/activate` 等）を意識することなく、プロジェクトルートから各コマンドを直接実行できます。

### 1. 前処理パイプライン (`preprocess`)

```bash
# 全パイプライン（Firebaseダウンロード〜ウィンドウ生成）の一括実行
uv run python preprocess/main.py --stage all

# 中間ファイルを削除せず保持して実行
uv run python preprocess/main.py --stage all --keep-intermediates

# 特定ステージのみ実行 (例: ランドマーク工程のみ)
uv run python preprocess/main.py --stage lm
```

### 2. 学習・評価パイプライン (`learning`)

```bash
# TS2Vec表現学習〜GMMクラスタリング〜特徴量融合〜LOSO-CVの一括実行
uv run python learning/main.py --stage all

# 下流タスク (LOSO-CV 回帰評価) のみ実行
uv run python learning/main.py --stage downstream_task

# 全ウィンドウサイズ (5s, 10s, 15s) および全モデルの一括評価
uv run python learning/downstream_task/main.py -w all --models ridge elastic_net random_forest lightgbm xgboost catboost
```

### 3. ベンチマーク検証 (`validation`)

```bash
# 顔ランドマーク前処理検証 (サニティチェック & 分布確認)
uv run python validation/lm/run_lm_validation.py

# rPPG / 心拍数 (HR) 精度評価
uv run python validation/rppg/hr/evaluate.py --config validation/rppg/hr/config_ver2.yaml

# 心拍変動 (HRV) GT真値比較・散布図プロット
uv run python validation/rppg/hrv/compare_hrv_with_gt.py --config validation/rppg/hrv/config.yaml
```

### 4. 探索的データ解析 (`eda`)

```bash
# 加工前 (Raw Data) EDA (欠損値・サンプリングレート・生波形・個人差・クラスバランス)
uv run python eda/raw/run_eda_raw.py

# 加工後 (Processed Data) EDA (Before/After比較・相関分析・多重共線性)
uv run python eda/processed/run_eda_processed.py
```

---

## Google Cloud Run デプロイ環境

将来の Google Cloud Run デプロイに向けて最適化された [`Dockerfile`](./Dockerfile) および [`.dockerignore`](./.dockerignore) が用意されています。

- **マルチステージビルド**: ビルド環境で `uv sync --frozen --no-dev` を実行し、本番依存関係のみを最小イメージへ転送
- **Cloud Run 準拠**: `$PORT`（デフォルト 8080）環境変数リスニング、非root実行（`appuser`）設定済み

```bash
# Docker イメージのビルド
docker build -t make-ai-service .

# ローカルでの動作確認テスト
docker run -p 8080:8080 -e PORT=8080 make-ai-service
```

---

## 参考文献

1. **POS Algorithm (Remote-PPG 脈波抽出)**

   ```text
   Wenjin Wang, Albertus C. den Brinker, Sander Stuijk, and Gerard de Haan,
   "Algorithmic Principles of Remote-PPG,"
   IEEE Transactions on Biomedical Engineering, Vol. 64, No. 7, pp. 1479-1491, 2017.
   DOI: https://doi.org/10.1109/TBME.2016.2609282
   ```

2. **個人最適化 Eye Aspect Ratio 及び瞬目検出**

   ```text
   斉藤功樹,
   "3D landmarkに基づく個人最適化されたEye Aspect Ratioによる瞬目検出手法,"
   人工知能学会全国大会論文集, 第38回, 2024.
   DOI: https://doi.org/10.11517/pjsai.JSAI2024.0_4I3GS702
   ```

3. **頭部回転角・眼球回転角**

   ```text
   竹内大樹, ほか,
   "VDT作業者の状態評価のための非接触センシング（第2報）,"
   人間工学, Vol. 61, No. 3, pp. 190-198, 2025.
   DOI: https://doi.org/10.5100/jje.61.190
   ```

4. **NeuroKit2**

   ```text
   Dominique Makowski, Tam Pham, Zen J. Lau, Jan C. Brammer, François Lespinasse,
   Hung Pham, Christopher Schölzel, and S. H. Annabel Chen,
   "NeuroKit2: A Python toolbodata_cox for neurophysiological signal processing,"
   Behavior Research Methods, Vol. 53, No. 4, pp. 1689-1696, 2021.
   DOI: https://doi.org/10.3758/s13428-020-01516-y
   ```

5. **UBFC-rPPG 公開ベンチマークデータセット**

   ```text
   S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, and J. Dubois,
   "Unsupervised skin tissue segmentation for remote photoplethysmography,"
   Pattern Recognition Letters, Vol. 124, pp. 82-90, 2019.
   DOI: https://doi.org/10.1016/j.patrec.2017.10.017
   ```

6. **Kubios HRV 外れ値・異所性拍補正手法**

   ```text
   J. P. Lipponen and M. P. Tarvainen,
   "A robust algorithm for heart rate variability time series artefact correction using novel beat classification,"
   Journal of Medical Engineering & Technology, Vol. 43, No. 3, pp. 173-181, 2019.
   DOI: https://doi.org/10.1080/03091902.2019.1640306
   ```
