# 探索的データ解析 (EDA: Exploratory Data Analysis)

本ディレクトリは、生体シグナル（顔ランドマーク・非接触脈波 rPPG）から集中度を推定する機械学習モデル構築における、**加工前データ（Raw Data）**および**加工完了後データ（Processed Data）**の探索的データ解析を行うためのモジュール群です。

すべて **Jupyter Notebook (`.ipynb`)** として作成されており、VS Code や Jupyter Lab 上で対話的に実行・観察できるほか、CLI から `jupyter nbconvert` を用いて一発でヘッドレス実行・レポート出力が可能です。

---

## 主な分析観点と仕様

### eda/raw（加工前データ: `raw_data_eda.ipynb`）

| 分析観点 | 可視化・出力内容 | 前処理・モデル設計への示唆 |
| :--- | :--- | :--- |
| **欠損値とサンプリングレート** | 被験者ごとの実効FPS（中央値・分散）、タイムスタンプ差分 $\Delta t$ 分布、フレームドロップ（>100ms）頻度、顔未検出NaN率 | ブラウザ・カメラ負荷による不等間隔ジッターが確認されるため、**50Hz PCHIP補間等間隔化**が不可欠。 |
| **生波形の可視化とノイズの特定** | 顔頬部生RGB輝度波形（Greenチャネル）、簡易EAR瞬目ディップ、頭部姿勢揺らぎ、急峻なスパイクノイズ | 照明変動や体動による低周波ワンダリングと高周波センサノイズを除去するため、**POS色空間投影**および**バンドパス/SavGolフィルタ**が必須。 |
| **個人差（ベースラインズレ）** | 被験者間の生RGB輝度レベル分布、生EAR値分布の箱ひげ図比較 | 肌色や骨格の違いにより初期水準が大きく異なるため、**初期注視キャリブレーション（比率/差分）**および**個人内Zスコア正規化**により被験者ID依存性を排除。 |
| **クラスバランスの確認** | 被験者別・全体での集中（Label 1）vs 非集中（Label 0）の比率・フレーム数 | 全体比率は約79%（集中）: 21%（非集中）の不均衡。LOSO-CV評価およびモデル重み付け（class weights等）での考慮が必要。 |

### eda/processed（加工後データ: `processed_data_eda.ipynb`）

| 分析観点 | 可視化・出力内容 | 機械学習・下流モデルへの示唆 |
| :--- | :--- | :--- |
| **前処理効果検証 (Before/After)** | 1. 生RGB vs POS脈波抽出（`bvp_selected`）＆平滑化心拍数（`bpm`）<br>2. 生EAR vs キャリブレーション比率 vs 平滑化Zスコア<br>3. 被験者間ベースラインズレ解消（全被験者が平均0・分散1に整列） | 前処理工程によってノイズや個人差が綺麗に除去され、機械学習に適した高品質なゼロ平均化時系列が生成されていることを実証。 |
| **特徴量とラベルの相関分析** | 目的変数との相関係数（Pearson / Spearman）、効果量（Cohen's d）、正負相関ランキングTop20、上位特徴量のラベル別確率密度分布（KDE） | 頭部姿勢の傾き差分（`head_roll_diff_smooth_z`, `head_pitch_diff_smooth_z`）や開眼比率（`ear_ratio_smooth_z`）が集中度と有意に相関（$r \approx 0.10 \sim 0.24$, $d > 0.5$）。 |
| **多重共線性のチェック** | 主要特徴量の相関行列ヒートマップ（Seaborn）、高相関ペア（$|r| \ge 0.85$）リストの自動検出 | 平滑化前後や各ROI間、速度・加速度の派生特徴量間で強い共線性が存在。OLS等では不安定になるため、**L2正則化（Ridge）、ElasticNet**、または共線性に頑健な**ツリー系モデル（LightGBM, CatBoost）**の採用が最適。 |

## ディレクトリ構成

```text
eda/
├── raw/                                  # 加工前データ (Raw Data) EDA
│   ├── raw_data_eda.ipynb                # 【分析Notebook】欠損値・FPS・生波形・個人差・クラスバランス
│   └── reports/                          # 出力アーティファクト
│       ├── raw_eda_summary.csv           # 被験者別統計サマリーテーブル
│       └── figures/                      # 高解像度可視化プロット (PNG)
│           ├── fig1_sampling_rate_distribution.png
│           ├── fig2_subject_fps_and_drops.png
│           ├── fig3_missing_values_summary.png
│           ├── fig4_raw_rppg_signals_noise.png
│           ├── fig5_raw_landmarks_ear_pose.png
│           ├── fig6_individual_baseline_differences.png
│           ├── fig7_class_balance_by_subject.png
│           └── fig8_class_balance_overall.png
│
├── processed/                            # 加工完了後データ (Processed Data) EDA
│   ├── processed_data_eda.ipynb          # 【分析Notebook】Before/After比較・相関分析・多重共線性
│   └── reports/                          # 出力アーティファクト
│       ├── feature_correlations.csv      # 全特徴量と目的変数(ラベル)の相関係数・効果量一覧
│       ├── multicollinearity_pairs.csv   # 高相関ペア (|r| >= 0.85) リスト
│       └── figures/                      # 高解像度可視化プロット (PNG)
│           ├── fig1_rppg_before_after.png
│           ├── fig2_landmarks_before_after.png
│           ├── fig3_baseline_shift_removal.png
│           ├── fig4_feature_label_correlation_ranking.png
│           ├── fig5_top_features_kde_by_label.png
│           └── fig6_multicollinearity_heatmap.png
│
└── README.md                             # 本ドキュメント
```
