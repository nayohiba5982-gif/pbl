"""
スライディングウィンドウ作成時のラベル別トリミング（除外区間）設定モジュール
"""

# デフォルト除外秒数設定:
# - label 1: 講義動画（開始時の慣れ15秒、終了前の動作4秒を除外）
# - label 0: リラックス動画（開始時の切り替え慣れ15秒を除外）
DEFAULT_TRIM_MARGINS: dict[int, dict[str, float]] = {
    0: {"start_sec": 15.0, "end_sec": 0.0},
    1: {"start_sec": 15.0, "end_sec": 4.0},
}


def normalize_trim_margins(trim_margins: dict | None) -> dict[int, dict[str, float]]:
    """
    トリミング設定辞書を {int_label: {"start_sec": float, "end_sec": float}} の形式に正規化します。
    指定がないラベルやキーには DEFAULT_TRIM_MARGINS の値がフォールバックとして使用されます。
    """
    normalized = {
        0: {"start_sec": 15.0, "end_sec": 0.0},
        1: {"start_sec": 15.0, "end_sec": 4.0},
    }
    if not trim_margins:
        return normalized

    for k, v in trim_margins.items():
        if not isinstance(v, dict):
            continue
        k_str = str(k).lower().replace("label_", "").replace("label", "").strip()
        try:
            l_int = int(k_str)
        except ValueError:
            continue

        start_sec = float(v.get("start_sec", normalized.get(l_int, {}).get("start_sec", 0.0)))
        end_sec = float(v.get("end_sec", normalized.get(l_int, {}).get("end_sec", 0.0)))
        normalized[l_int] = {"start_sec": max(0.0, start_sec), "end_sec": max(0.0, end_sec)}

    return normalized
