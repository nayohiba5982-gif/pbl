import numpy as np
import pandas as pd

def clean_rppg_outliers(
    df_rppg: pd.DataFrame, 
    df_lm: pd.DataFrame = None, 
    fps: int = 25, 
    min_val: float = 5.0, 
    max_val: float = 250.0, 
    diff_threshold: float = 15.0, 
    sigma_threshold: float = 3.0, 
    interp_limit: int = 5
) -> pd.DataFrame:
    """
    rPPG (輝度信号) データに対する外れ値・欠損値除去を行います。
    
    Parameters:
        df_rppg (pd.DataFrame): 入力rPPGデータ
        df_lm (pd.DataFrame, optional): 対応するLandmarkデータ（顔検出失敗の同期用）
        fps (int): フレームレート (デフォルト 25)
        min_val (float): 黒つぶれ・遮蔽の最小閾値
        max_val (float): 白飛びの最大閾値
        diff_threshold (float): 1次差分の固定閾値
        sigma_threshold (float): 移動標準偏差のσ倍率
        interp_limit (int): 連続NaNを線形補間する上限フレーム数
    """
    df_clean = df_rppg.copy()
    rppg_cols = [c for c in df_clean.columns if c.startswith('rppg_')]

    # ─── 1. 物理的上限・下限チェック（輝度飽和・遮蔽） ───
    # 5未満（黒つぶれ/手で覆う）または 250以上（白飛び）を検知
    mask_physical = (df_clean[rppg_cols] <= min_val) | (df_clean[rppg_cols] >= max_val)

    # ─── 2. 急激な輝度変化（差分スパイク）の除去 ───
    # 1秒間（25フレーム）の移動窓で移動平均と移動標準偏差を計算
    window_size = max(3, int(fps * 1.0))
    mask_spike = pd.DataFrame(False, index=df_clean.index, columns=rppg_cols)

    for col in rppg_cols:
        diff = df_clean[col].diff().abs()
        roll_mean = diff.rolling(window=window_size, min_periods=1, center=True).mean()
        roll_std = diff.rolling(window=window_size, min_periods=1, center=True).std().fillna(0)

        # 「3σを超える急変」かつ「固定閾値(15以上)を超える変動」をスパイクと判定
        is_spike = (diff > (roll_mean + sigma_threshold * roll_std)) & (diff > diff_threshold)
        mask_spike[col] = is_spike

    # ─── 3. バウンディングボックス外（顔見切れ/未検出）時の欠損処理 ───
    mask_lm_nan = pd.Series(False, index=df_clean.index)
    if df_lm is not None:
        lm_cols = [c for c in df_lm.columns if c.startswith('lm_')]
        # Landmark側で全ランドマーク座標がNaNになっている（＝顔未検出）行を取得
        lm_nan_rows = df_lm[lm_cols].isna().all(axis=1)

        if len(df_lm) == len(df_clean) and 'time' in df_lm.columns and 'time' in df_clean.columns:
            if np.allclose(df_lm['time'], df_clean['time'], atol=1e-3):
                mask_lm_nan = lm_nan_rows
        else:
            # タイムスタンプが完全に一致しない場合は最近傍で同期
            merged = pd.merge_asof(
                df_clean[['time']].reset_index(),
                df_lm[['time']].assign(is_nan=lm_nan_rows),
                on='time',
                direction='nearest'
            )
            mask_lm_nan = merged['is_nan'].fillna(False).values

    # ─── 4. マスクの適用（NaN化） ───
    for col in rppg_cols:
        df_clean.loc[mask_physical[col], col] = np.nan
        df_clean.loc[mask_spike[col], col] = np.nan

    # 顔が検出されなかったフレームは全rppg列をNaN化
    df_clean.loc[mask_lm_nan, rppg_cols] = np.nan

    # ─── 5. 短時間の欠損のみ線形補完 ───
    # スプラインリサンプリング前の準備として、数フレーム(例:5フレーム=0.2秒)程度の瞬時ノイズのみ補完
    df_clean[rppg_cols] = df_clean[rppg_cols].interpolate(method='linear', limit=interp_limit)

    # print(f"  [rPPGクレンジング完了]")
    # print(f"    ├─ 白飛び/黒つぶれ除外: {mask_physical.sum().sum()} セル")
    # print(f"    ├─ 差分スパイク除外: {mask_spike.sum().sum()} セル")
    # print(f"    └─ 顔未検出(LM同期)除外: {mask_lm_nan.sum()} 行")

    return df_clean