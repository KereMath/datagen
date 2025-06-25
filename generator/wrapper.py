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

# İzin verilen anomali/kırılma sayıları
MIN_EVENT_COUNT = 1
MAX_EVENT_COUNT = 10  # Maksimum kabul edilebilir anomali/kırılma sayısı

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
    # Sütun sırasını koruyan şekilde kaydet
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

# Listeyi düzleştiren yardımcı fonksiyon
def flatten_list(nested_list):
    """Herhangi bir iç içe listeyi düz tek seviyeli liste haline getir"""
    if isinstance(nested_list, (list, tuple, np.ndarray)):
        result = []
        for item in nested_list:
            if isinstance(item, (list, tuple, np.ndarray)):
                result.extend(flatten_list(item))
            else:
                result.append(item)
        return result
    else:
        return [nested_list]  # Tek bir eleman

def check_event_count(idx_list, expected_count, allowed_range=None):
    """İndeks listesinin uzunluğunu beklenen sayıyla karşılaştırır"""
    if allowed_range is None:
        allowed_range = (expected_count - 1, expected_count + 1)  # Varsayılan tolerans
    
    if not idx_list:
        return False
    
    actual_count = len(idx_list)
    min_count, max_count = allowed_range
    
    # Özel durum: point_anomalies ve contextual_anomalies için özel kontrol
    if expected_count > 1 and actual_count < min_count:
        return False
    
    return min_count <= actual_count <= max_count

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
            # Contextual anomaly multi case'lerini atla
            if family == "contextual_anomaly" and mode == "multi":
                print(f"NOT: contextual_anomaly multi case'leri desteklenmiyor. Atlanıyor...")
                continue
                
            for cfg_idx, spec in enumerate(cfgs):
                # Parametre bilgilerini ayıkla
                location = spec.get("location", None)
                num_breaks = spec.get("num_breaks", 1)
                num_anomalies = spec.get("num_anomalies", 1)
                change_type = spec.get("change_type", None)
                is_multi = (num_breaks > 1 or num_anomalies > 1)
                
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
                
                while successful_cases < N_PER_CASE and attempts < N_PER_CASE * 5:
                    attempts += 1
                    
                    # Rastgele uzunlukta zaman serisi oluştur
                    length = random.randint(120, 500) if is_multi else random.randint(60, 500)
                    ts = TimeSeriesGenerator(length)
                    base = ts.generate_stationary_base_series('ar')
                    
                    # Olay detayları için bilgileri sakla
                    event_info = {
                        'family': family,
                        'mode': mode,
                        'count': num_breaks if family in ["mean_shift", "variance_shift", "trend_shift"] else num_anomalies,
                        'location': location,
                        'type': change_type,
                        'is_multi': is_multi,
                    }
                    
                    # --- olay üretimi --------------------------------------
                    try:
                        if family == "mean_shift":
                            # num_breaks parametresi fonksiyonda var
                            df_raw, shifts_info = ts.generate_mean_shift(base, **spec)
                            try:
                                idx = flatten_list(df_raw.at[0, "mean_shift_point"])
                            except:
                                idx = []
                                for info in shifts_info:
                                    if isinstance(info, tuple) and len(info) > 1:
                                        idx.extend(flatten_list(info[1]))
                            
                            flags, brk_idx, an_idx = ["mean_shift"], idx, []
                            event_info['details'] = f"Mean level shift - {event_info['location'] or 'multiple'} location"
                            
                            # Kırılma noktası sayısını kontrol et
                            if not check_event_count(brk_idx, num_breaks, (num_breaks, num_breaks)):
                                print(f"Beklenen kırılma sayısı: {num_breaks}, bulunan: {len(brk_idx)}. Yeniden üretiliyor...")
                                continue
                            
                        elif family == "variance_shift":
                            df_raw, shifts_info = ts.generate_variance_shift(base, **spec)
                            try:
                                idx = flatten_list(df_raw.at[0, "variance_shift_points"])
                            except:
                                idx = []
                                for info in shifts_info:
                                    if isinstance(info, tuple) and len(info) > 1:
                                        idx.extend(flatten_list(info[1]))
                            
                            flags, brk_idx, an_idx = ["var_shift"], idx, []
                            event_info['details'] = f"Variance change - {event_info['location'] or 'multiple'} location"
                            
                            # Kırılma noktası sayısını kontrol et
                            if not check_event_count(brk_idx, num_breaks, (num_breaks, num_breaks)):
                                print(f"Beklenen kırılma sayısı: {num_breaks}, bulunan: {len(brk_idx)}. Yeniden üretiliyor...")
                                continue
                            
                        elif family == "trend_shift":
                            df_raw, shifts_info = ts.generate_trend_shift(base, signs=[1], **spec)
                            try:
                                idx = flatten_list(df_raw.at[0, "trend_shift_points"])
                            except:
                                idx = []
                                for info in shifts_info:
                                    if isinstance(info, tuple) and len(info) > 1:
                                        idx.extend(flatten_list(info[1]))
                            
                            flags, brk_idx, an_idx = ["trend_shift"], idx, []
                            event_info['details'] = f"Trend shift ({change_type or 'unknown'}) - {event_info['location'] or 'multiple'} location"
                            
                            # Kırılma noktası sayısını kontrol et
                            if not check_event_count(brk_idx, num_breaks, (num_breaks, num_breaks)):
                                print(f"Beklenen kırılma sayısı: {num_breaks}, bulunan: {len(brk_idx)}. Yeniden üretiliyor...")
                                continue
                            
                        elif family == "point_anomaly":
                            if mode == "single":
                                # Single point anomaly
                                df_raw, info = ts.generate_point_anomaly(base, location=spec.get('location'))
                                flags = ["point_anom_single"]
                                idx = flatten_list(info[0][1])
                                expected_count = 1
                            else:
                                # Multi point anomaly - num_anomalies parametresini KABUL ETMİYOR
                                # Random üretildiği için istenen sayıda anomali üretene kadar dene
                                df_raw, info = ts.generate_point_anomalies(base, scale_factor=spec.get('scale_factor', 1))
                                flags = ["point_anom_multi"]
                                idx = flatten_list(info[0][1])
                                expected_count = num_anomalies
                            
                            flags, brk_idx, an_idx = flags, [], idx
                            event_info['details'] = f"Point anomaly - {event_info['location'] or 'random'} location"
                            
                            # Multi mode ise ve beklenen sayıda anomali yoksa yeniden üret
                            if mode == "multi" and not check_event_count(an_idx, expected_count):
                                print(f"Beklenen anomali sayısı: {expected_count}, bulunan: {len(an_idx)}. Yeniden üretiliyor...")
                                continue
                            
                        elif family == "collective_anomaly":
                            # num_anomalies parametresi fonksiyonda var
                            clean_spec = {k: v for k, v in spec.items() 
                                        if k in ['num_anomalies', 'location', 'scale_factor', 
                                                'min_distance', 'return_debug']}
                            df_raw, info = ts.generate_collective_anomalies(base, **clean_spec)
                            flags = ["collect_anom"]
                            # info içinden başlangıç indeksleri alınır
                            an_idx = [start for start, _ in info]
                            brk_idx = []
                            event_info['details'] = f"Collective anomaly - {event_info['location'] or 'random'} location"
                            
                            # Anomali sayısını kontrol et
                            if mode == "multi" and not check_event_count(an_idx, num_anomalies):
                                print(f"Beklenen anomali sayısı: {num_anomalies}, bulunan: {len(an_idx)}. Yeniden üretiliyor...")
                                continue
                            
                        elif family == "contextual_anomaly":
                            # Sadece 'single' mod desteklenir
                            flags = ["context_anom"]
                            
                            # Contextual anomaly üret
                            df_raw, info = ts.generate_contextual_anomalies(base, scale_factor=spec.get('scale_factor', 1))
                            
                            # info içinde (start, end) tuple olarak gelir
                            if info and len(info) > 0:
                                start, end = info[0]
                                
                                # Lokasyon kontrolü
                                if location:
                                    n = len(base['data'])
                                    # Anomali konumunu kontrol et
                                    if location == "beginning" and start > n * 0.3:
                                        print(f"Anomali başlangıç bölgesi uygun değil: {start} > {n * 0.3}. Yeniden üretiliyor...")
                                        continue
                                    elif location == "middle" and (start < n * 0.4 or start > n * 0.6):
                                        print(f"Anomali başlangıç bölgesi uygun değil: {start} < {n * 0.4} veya > {n * 0.6}. Yeniden üretiliyor...")
                                        continue
                                    elif location == "end" and start < n * 0.7:
                                        print(f"Anomali başlangıç bölgesi uygun değil: {start} < {n * 0.7}. Yeniden üretiliyor...")
                                        continue
                                
                                # Sadece başlangıç noktasını işaretleriz
                                an_idx = [start]
                            else:
                                print("Contextual anomali üretilemedi. Yeniden deneniyor...")
                                continue
                            
                            brk_idx = []
                            event_info['details'] = f"Contextual anomaly - {event_info['location'] or 'random'} location"
                            
                        else:
                            continue
                    except Exception as e:
                        print(f"Hata oluştu: {e}. Yeniden üretiliyor...")
                        continue
                    # -------------------------------------------------------

                    # Veri çerçevesi oluşturma (tüm sütunlar 0 ile başlar)
                    df = empty_frame(length)
                    df["data"] = df_raw["data"].values
                    df["stationary"] = 0
                    
                    # Kırılma/anomali noktalarını temizleme ve hazırlama
                    if brk_idx is None:
                        brk_idx = []
                    if an_idx is None:
                        an_idx = []
                    
                    # Her eleman int olmalı
                    brk_idx = [int(idx) for idx in brk_idx if isinstance(idx, (int, np.integer))]
                    an_idx = [int(idx) for idx in an_idx if isinstance(idx, (int, np.integer))]
                    
                    # DataFrame sınırları içinde olmalı
                    brk_idx = [idx for idx in brk_idx if 0 <= idx < length]
                    an_idx = [idx for idx in an_idx if 0 <= idx < length]
                    
                    # Son kontrol: İstenen sayıda kırılma/anomali var mı?
                    expected_count = num_breaks if family in ["mean_shift", "variance_shift", "trend_shift"] else num_anomalies
                    if is_multi and len(brk_idx) + len(an_idx) < expected_count:
                        print(f"İstenen {expected_count} olaydan daha az ({len(brk_idx) + len(an_idx)}) var. Yeniden üretiliyor...")
                        continue
                    
                    # Tarih sütunu ekleme
                    df = add_dates(df)
                    
                    # Olay özelliklerini işaretle (0/1 formatında)
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
                    print(f"Başarılı: {family}/{mode}/{location or ''} - {successful_cases}/{N_PER_CASE}")
                
                if successful_cases < N_PER_CASE:
                    print(f"UYARI: {family}/{mode}/{location or ''} için yalnızca {successful_cases}/{N_PER_CASE} örnek üretilebildi!")

    print(f"✓ Üretim tamamlandı. Çıktı klasörü: {ROOT_OUT}")

if __name__ == "__main__":
    main()