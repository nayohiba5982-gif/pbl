import sys
from pathlib import Path

_LEARNING_DIR = Path(__file__).resolve().parent
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.mixture import GaussianMixture
from config_loader import load_config, log
from downstream_task.data_loader import (
    load_embeddings_and_meta,
    load_explicit_features,
    build_modality_feature_names,
    split_modality_feature_indices,
    standardize_subject_wise,
)
from downstream_task.pls import DualStreamPLS

cfg = load_config('config.yaml')
learning_root = Path('learning').resolve()
data_dir = learning_root / 'data' / '10s'
input_data_dir = Path('preprocess/data/windowed/10s').resolve()

# 1. データ読み込み
emb_vis, emb_bio, meta_df = load_embeddings_and_meta(
    data_dir=data_dir,
    input_data_dir=input_data_dir,
    learning_root=learning_root,
    window_size='10s',
    log=log
)
explicit_feats, explicit_names = load_explicit_features(
    meta_df=meta_df,
    input_data_dir=input_data_dir,
    data_dir=data_dir,
    window_size='10s',
    log=log
)

n_explicit = len(explicit_names) if explicit_names else 0
all_names, vis_names, vital_names = build_modality_feature_names(
    n_vis_pls=3, n_bio_pls=3, explicit_names=explicit_names, n_explicit=n_explicit
)

print('=' * 70)
print('【調査1: 特徴量スライシング・分離ロジック】')
print(f'総特徴量数: {len(all_names)}')
print(f'Visual 特徴量 ({len(vis_names)} 次元):')
for i, name in enumerate(vis_names, 1):
    print(f'  {i:2d}. {name}')
print(f'Vital 特徴量 ({len(vital_names)} 次元):')
for i, name in enumerate(vital_names, 1):
    print(f'  {i:2d}. {name}')
print('=' * 70)

# 2. 全16 Fold の GMM Alignment および y_soft と task labels の相関調査
label_col = 'label' if 'label' in meta_df.columns else ('label_mean' if 'label_mean' in meta_df.columns else None)
labels = meta_df[label_col].values.astype(float) if label_col else np.zeros(len(meta_df))
subjects = meta_df['username'].values if 'username' in meta_df.columns else np.arange(len(meta_df))

logo = LeaveOneGroupOut()
fold_alignments = []
oof_y_soft = np.full(len(meta_df), np.nan)

weights = [1.0, 0.0]  # focus=1, relax=0
imbalance_penalty = 1.0
actual_vis_comps = 3
actual_bio_comps = 3
cov_type = 'full'
seed = 42

for fold, (train_idx, test_idx) in enumerate(logo.split(emb_vis, labels, groups=subjects), 1):
    test_user = str(subjects[test_idx[0]])
    train_sub_users = subjects[train_idx]
    test_sub_users = subjects[test_idx]

    # PLS
    pls = DualStreamPLS(n_vis=actual_vis_comps, n_bio=actual_bio_comps)
    pls.fit(emb_vis[train_idx], emb_bio[train_idx], labels[train_idx])
    z_train = pls.transform(emb_vis[train_idx], emb_bio[train_idx])
    z_test = pls.transform(emb_vis[test_idx], emb_bio[test_idx])

    # 標準化
    z_train_norm = standardize_subject_wise(z_train, train_sub_users)
    z_test_norm = standardize_subject_wise(z_test, test_sub_users)

    # GMM
    gmm = GaussianMixture(n_components=2, covariance_type=cov_type, random_state=seed)
    gmm.fit(z_train_norm)
    gmm_probs_train = gmm.predict_proba(z_train_norm)
    gmm_probs_test = gmm.predict_proba(z_test_norm)

    # 相関とアライメント
    r0 = np.corrcoef(gmm_probs_train[:, 0], labels[train_idx])[0, 1]
    r1 = np.corrcoef(gmm_probs_train[:, 1], labels[train_idx])[0, 1]
    r0 = 0.0 if np.isnan(r0) else r0
    r1 = 0.0 if np.isnan(r1) else r1

    focus_c = 1 if r1 > r0 else 0
    relax_c = 1 - focus_c

    y_soft_train = (gmm_probs_train[:, focus_c] * weights[0] + gmm_probs_train[:, relax_c] * weights[1]).astype(np.float32)
    y_soft_test = (gmm_probs_test[:, focus_c] * weights[0] + gmm_probs_test[:, relax_c] * weights[1]).astype(np.float32)
    oof_y_soft[test_idx] = y_soft_test

    r_train_soft = np.corrcoef(y_soft_train, labels[train_idx])[0, 1]
    r_test_soft = np.corrcoef(y_soft_test, labels[test_idx])[0, 1]

    # クラスタごとの正解ラベル平均 (c0_ratio, c1_ratio)
    clusters_train = np.argmax(gmm_probs_train, axis=1)
    c0_ratio = float(np.mean(labels[train_idx][clusters_train == 0])) if np.any(clusters_train == 0) else 0.0
    c1_ratio = float(np.mean(labels[train_idx][clusters_train == 1])) if np.any(clusters_train == 1) else 0.0
    inverted = bool(focus_c == 0)

    fold_alignments.append({
        'fold': fold,
        'user': test_user,
        'r0': r0,
        'r1': r1,
        'focus_c': focus_c,
        'c0_ratio': c0_ratio,
        'c1_ratio': c1_ratio,
        'inverted': inverted,
        'r_train_soft': r_train_soft,
        'r_test_soft': r_test_soft,
    })

print('【調査2: GMM クラスタ自動アライメントの全16 Fold ログ】')
df_align = pd.DataFrame(fold_alignments)
print(df_align.to_string(index=False))
print('=' * 70)

# 3. GMM ソフトラベル全体の品質
print('【調査3: GMM ソフトラベル (y_soft) の品質確認】')
overall_r = np.corrcoef(oof_y_soft, labels)[0, 1]
print(f'OOF 全体での y_soft と 正解ラベル y_task の相関係数 r: {overall_r:.4f}')
train_r_mean = np.mean([f['r_train_soft'] for f in fold_alignments])
print(f'Train 内での y_soft と 正解ラベル y_task の相関係数平均 r: {train_r_mean:.4f}')

# PreviousTask / EDA で作成された y_soft (learning/data/10s/y_soft_train.npy 等) の確認
p_soft_tr = data_dir / 'y_soft_train.npy'
p_lbl_tr = data_dir / 'meta_train.csv'
if p_soft_tr.exists() and p_lbl_tr.exists():
    y_eda_soft = np.load(p_soft_tr)
    df_eda = pd.read_csv(p_lbl_tr)
    l_col = 'label' if 'label' in df_eda.columns else 'label_mean'
    if l_col in df_eda.columns and len(df_eda) == len(y_eda_soft):
        r_eda = np.corrcoef(y_eda_soft, df_eda[l_col].values)[0, 1]
        print(f'PreviousTask/EDA 単体 (10s Train k=8): y_soft と label の相関 r: {r_eda:.4f}')
