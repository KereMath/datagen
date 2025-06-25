# data-gen/generator/wrapper.py
import os, random, yaml, pandas as pd, numpy as np
from pathlib import Path
from tsgen import TimeSeriesGenerator
from datetime import datetime, timedelta

# ── matplotlib başsız -----------------------------------------------------
import matplotlib
matplotlib.use("Agg")          # GUI yok; figürler dosyaya yazılır
import matplotlib.pyplot as plt

ROOT_OUT    = Path("../output")
START_DAY   = "2020-01-01"
N_PER_CASE  = 20
COL_ORDER   = Path("columns.txt").read_text().strip().splitlines()

# ── yardımcı --------------------------------------------------------------
def add_dates(df, start_date=START_DAY):
    """DataFrame'e tarih sütunu ekler"""
    df["date"] = pd.date_range(start_date, periods=len(df), freq="D")
    return df

def empty_frame(length):
    """Boş DataFrame oluşturur"""
    return pd.DataFrame({c: [0]*length for c in COL_ORDER})

def mark_cols(df, flags):
    """Belirtilen bayrakları DataFrame'de işaretler"""
    for f in flags:
        if f in df.columns:
            df[f] = 1
    return df

def save_csv(df, path):
    """DataFrame'i CSV olarak kaydeder"""
    df.to_csv(path, index=False, columns=[c for c in COL_ORDER if c in df.columns])

def save_plot(df, brk_idx, an_idx, png_path, event_info=None, add_text=True):
    """Zaman serisini grafikleştirir ve kırılma/anomali noktalarını işaretler"""
    plt.figure(figsize=(12, 6))
    plt.plot(df["data"], lw=0.8, color='blue')
    
    # Kırılma noktalarını işaretle
    if len(brk_idx) > 0:
        plt.scatter(brk_idx, df.loc[brk_idx, "data"], marker="x", s=80, color='green', label='Breakpoints')
        
        # Kırılma noktalarını metinlerle etiketle
        if add_text and event_info:
            for i, idx in enumerate(brk_idx):
                info_text = f"Break #{i+1}"
                if 'type' in event_info:
                    info_text += f"\n{event_info['type']}"
                plt.annotate(info_text, (idx, df.loc[idx, "data"]), 
                            xytext=(10, 10), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.7))
    
    # Anomali noktalarını işaretle
    if len(an_idx) > 0:
        plt.scatter(an_idx, df.loc[an_idx, "data"], facecolors="none", 
                   edgecolors="red", s=80, linewidth=2, label='Anomalies')
        
        # Anomali noktalarını metinlerle etiketle
        if add_text and event_info:
            for i, idx in enumerate(an_idx):
                info_text = f"Anomaly #{i+1}"
                if 'type' in event_info:
                    info_text += f"\n{event_info['type']}"
                plt.annotate(info_text, (idx, df.loc[idx, "data"]), 
                            xytext=(10, -20), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", alpha=0.7))
    
    # Graf başlığında açıklayıcı bilgiler
    if event_info:
        title = event_info.get('title', 'Time Series')
        if 'location' in event_info and event_info['location']:
            title += f" - {event_info['location'].capitalize()} location"
        if 'count' in event_info:
            title += f" - {event_info['count']} events"
        if 'type' in event_info:
            title += f" - {event_info['type']}"
        plt.title(title)
    
    plt.tight_layout()
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(png_path, dpi=120)
    plt.close()

def mark_event_characteristics(df, event_info, brk_idx, an_idx):
    """DataFrame'e olay özelliklerini bayrak olarak işaretler (0/1 formatında)"""
    family = event_info.get('family', '')
    location = event_info.get('location', None)
    change_type = event_info.get('type', None)
    is_multi = event_info.get('is_multi', False)
    event_count = len(brk_idx) + len(an_idx)
    event_details = event_info.get('details', '')
    
    # Tüm satır için olay türünü işaretle
    df[family] = 1
    
    # Multi olup olmadığını işaretle
    df["is_multi"] = int(is_multi)
    
    # Olay sayısını ekle
    df["event_count"] = event_count
    
    # Olay açıklamasını ekle
    df["event_details"] = event_details
    
    # Lokasyon işaretleme
    if location == "beginning":
        df["location_beginning"] = 1
    elif location == "middle":
        df["location_middle"] = 1
    elif location == "end":
        df["location_end"] = 1
    
    # Değişim tipi işaretleme
    if change_type == "direction_change":
        df["change_type_direction"] = 1
    elif change_type == "magnitude_change":
        df["change_type_magnitude"] = 1
    elif change_type == "direction_and_magnitude_change":
        df["change_type_direction_magnitude"] = 1
    
    # Kırılma noktalarını işaretle
    for idx in brk_idx:
        df.loc[idx, "is_break"] = 1
    
    # Anomali noktalarını işaretle
    for idx in an_idx:
        df.loc[idx, "is_anomaly"] = 1
    
    return df

# ── ana akış --------------------------------------------------------------
def main():
    cases = yaml.safe_load(Path("cases.yaml").read_text())
    ROOT_OUT.mkdir(exist_ok=True)

    for family, groups in cases.items():
        for mode, cfgs in groups.items():
            for cfg_idx, spec in enumerate(cfgs):
                # Parametre bilgilerini ayıkla
                location = spec.get("location", None)
                num_breaks = spec.get("num_breaks", 1)
                num_anomalies = spec.get("num_anomalies", 1)
                change_type = spec.get("change_type", None)
                target_anomalies = spec.get("target_anomalies", None)
                is_multi = (num_breaks > 1 or num_anomalies > 1 or (family == "point_anomaly" and mode == "multi"))
                
                # Alt klasör yapısını oluştur
                if location:
                    subdir = ROOT_OUT / family / mode / location
                else:
                    subdir = ROOT_OUT / family / mode
                
                if change_type:
                    subdir = subdir / change_type
                
                subdir.mkdir(parents=True, exist_ok=True)
                
                # Her parametre setinden n adet örnek oluştur
                successful_cases = 0
                attempts = 0
                
                while successful_cases < N_PER_CASE and attempts < N_PER_CASE * 3:
                    attempts += 1
                    
                    # Rastgele uzunlukta zaman serisi oluştur
                    length = random.randint(120, 500) if is_multi else random.randint(60, 500)
                    ts = TimeSeriesGenerator(length)
                    base = ts.generate_stationary_base_series('ar')
                    
                    # Olay detayları için bilgileri sakla
                    event_info = {
                        'family': family,
                        'mode': mode,
                        'count': num_breaks if family in ["mean_shift", "variance_shift", "trend_shift"] else (target_anomalies if family == "point_anomaly" else num_anomalies),
                        'location': location,
                        'type': change_type,
                        'is_multi': is_multi,
                    }
                    
                    # --- olay üretimi (TSGen parametreleri direkt kullan) -------
                    if family == "mean_shift":
                        df_raw, shifts_info = ts.generate_mean_shift(base, **spec)
                        brk_idx = shifts_info[0][1] if shifts_info else []
                        an_idx = []
                        flags = ["mean_shift"]
                        event_info['details'] = f"Mean level shift - {location or 'multiple'} location"
                        
                    elif family == "variance_shift":
                        df_raw, shifts_info = ts.generate_variance_shift(base, **spec)
                        brk_idx = shifts_info[0][1] if shifts_info else []
                        an_idx = []
                        flags = ["var_shift"]
                        event_info['details'] = f"Variance change - {location or 'multiple'} location"
                        
                    elif family == "trend_shift":
                        df_raw, shifts_info = ts.generate_trend_shift(base, signs=[1], **spec)
                        brk_idx = shifts_info[0][1] if shifts_info else []
                        an_idx = []
                        flags = ["trend_shift"]
                        event_info['details'] = f"Trend shift ({change_type or 'unknown'}) - {location or 'multiple'} location"
                        
                    elif family == "point_anomaly":
                        if mode == "single":
                            df_raw, info = ts.generate_point_anomaly(base, location=location)
                            an_idx = list(info[0][1]) if info else []
                            flags = ["point_anom_single"]
                        else:
                            # Multi: TSGen num_anomalies desteklemiyor, wrapper kontrolü gerekli
                            df_raw, info = ts.generate_point_anomalies(base)
                            actual_anomalies = info[0][0] if info else 0
                            an_idx = list(info[0][1][0]) if info and len(info[0][1]) > 0 else []
                            
                            # Target anomali sayısı kontrolü
                            if target_anomalies:
                                if actual_anomalies != target_anomalies:
                                    continue
                            else:
                                # Target belirtilmemişse 2-4 arası kabul et
                                if actual_anomalies < 2 or actual_anomalies > 4:
                                    continue
                            
                            flags = ["point_anom_multi"]
                        
                        brk_idx = []
                        event_info['details'] = f"Point anomaly - {location or 'random'} location"
                        
                    elif family == "collective_anomaly":
                        df_raw, info = ts.generate_collective_anomalies(base, **spec)
                        an_idx = [start for start, _ in info] if info else []
                        brk_idx = []
                        flags = ["collect_anom"]
                        event_info['details'] = f"Collective anomaly - {location or 'random'} location"
                        
                    elif family == "contextual_anomaly":
                        # Sadece single mode - TSGen multi desteklemiyor
                        df_raw, info = ts.generate_contextual_anomalies(base)
                        
                        if info and len(info) > 0:
                            start, end = info[0]
                            an_idx = [start]
                        else:
                            continue
                        
                        brk_idx = []
                        flags = ["context_anom"]
                        event_info['details'] = f"Contextual anomaly"
                        
                    else:
                        continue
                    
                    # Veri çerçevesi oluşturma
                    df = empty_frame(length)
                    df["data"] = df_raw["data"].values
                    df["stationary"] = 0
                    
                    # Tarih sütunu ekleme
                    df = add_dates(df)
                    
                    # Olay özelliklerini işaretle
                    df = mark_event_characteristics(df, event_info, brk_idx, an_idx)
                    
                    # Bayrak sütunlarını işaretle
                    df = mark_cols(df, flags)
                    
                    # Grafik başlığı için bilgi
                    event_info['title'] = f"{family.replace('_', ' ').title()} ({mode})"
                    
                    # Kaydetme işlemleri
                    base_fname = f"{family[:2]}_{mode[:2]}_{cfg_idx:02d}_{successful_cases:02d}"
                    csv_path = subdir / f"{base_fname}.csv"
                    png_path = subdir / f"{base_fname}.png"

                    save_csv(df, csv_path)
                    save_plot(df, brk_idx, an_idx, png_path, event_info)
                    
                    successful_cases += 1
                    print(f"✓ {family}/{mode}/{location or ''} - {successful_cases}/{N_PER_CASE}")
                
                if successful_cases < N_PER_CASE:
                    print(f"⚠️  {family}/{mode} için yalnızca {successful_cases}/{N_PER_CASE} örnek üretilebildi!")

    print(f"✅ Üretim tamamlandı. Çıktı klasörü: {ROOT_OUT}")

if __name__ == "__main__":
    main()