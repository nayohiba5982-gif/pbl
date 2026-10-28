import numpy as np
import pandas as pd
from scipy.signal import welch

def calculate_snr(bvp_signal: np.ndarray, fs: float = 50.0) -> float:
    """
    Welch法によるパワースペクトル密度(PSD)から BVP 信号の SNR [dB] を算出
    """
    s_series = pd.Series(bvp_signal).interpolate(method='linear').bfill().ffill()
    valid_data = s_series.values
    
    if len(valid_data) < int(fs * 2):
        return -99.0

    freqs, psd = welch(valid_data, fs=fs, nperseg=min(len(valid_data), int(fs * 10)))
    
    pulse_band_mask = (freqs >= 0.75) & (freqs <= 2.5)
    if not np.any(pulse_band_mask) or np.sum(psd[pulse_band_mask]) == 0:
        return -99.0
        
    freqs_band = freqs[pulse_band_mask]
    psd_band = psd[pulse_band_mask]
    
    # ピーク周波数 (推定心拍周波数)
    f_hr = freqs_band[np.argmax(psd_band)]
    
    # 信号帯域 (f_hr および 2*f_hr の前後 ±0.15Hz)
    half_bw = 0.15
    signal_mask = ((freqs >= f_hr - half_bw) & (freqs <= f_hr + half_bw)) | \
                  ((freqs >= 2 * f_hr - half_bw) & (freqs <= 2 * f_hr + half_bw))
    
    noise_mask = pulse_band_mask & (~signal_mask)
    
    p_signal = np.sum(psd[signal_mask])
    p_noise = np.sum(psd[noise_mask])
    
    if p_noise <= 1e-8:
        return 20.0
        
    return float(10 * np.log10(p_signal / p_noise))