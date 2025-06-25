import os
import random
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")  # No GUI; figures are written to file
import matplotlib.pyplot as plt

# Import the original TimeSeriesGenerator from the same directory
from tsgen import TimeSeriesGenerator

# --- Sabitler ve Ayarlar -----------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent # multi-cases dizini
GENERATOR_DIR = ROOT_DIR / "generator"
OUTPUT_DIR = ROOT_DIR / "multi-output"
CASES_YAML = GENERATOR_DIR / "multi_cases.yaml"
COL_ORDER_FILE = GENERATOR_DIR / "columns.txt"

START_DAY = "2020-01-01"
N_PER_CASE = 1 # Her varyasyon için sadece 1 seri üretiyoruz, yaml'da zaten çok kombinasyon var

# Sütun sırasını oku
COL_ORDER = COL_ORDER_FILE.read_text().strip().splitlines()

# --- Yardımcı Fonksiyonlar ---------------------------------------------------
def add_dates(df, start_date=START_DAY):
    """DataFrame'e tarih sütunu ekler."""
    df["date"] = pd.date_range(start_date, periods=len(df), freq="D")
    return df

def empty_frame(length):
    """Boş DataFrame oluşturur ve tüm olası sütunları 0 ile başlatır."""
    df = pd.DataFrame({'data': np.zeros(length)}) # Start with 'data' column
    for col in COL_ORDER:
        if col not in df.columns:
            if col in ['point_anom_point', 'collect_anom_start', 'collect_anom_end',
                        'mean_shift_point', 'variance_shift_points', 'trend_shift_points']:
                df[col] = "" # Listeleri string olarak saklamak için
            elif col in ['point_anom_location', 'collect_anom_location', 'mean_shift_location',
                         'variance_shift_location', 'trend_shift_location', 'trend_shift_type',
                         'characteristic_1_type', 'characteristic_2_type']:
                 df[col] = "none" # Default string for location/type
            else:
                df[col] = 0 # Numeric flags
    return df

def save_csv(df, path):
    """DataFrame'i CSV olarak kaydeder, yalnızca mevcut sütunları kullanır."""
    # Sütunları COL_ORDER'a göre filtrele ve sırala
    cols_to_save = [c for c in COL_ORDER if c in df.columns]
    df.to_csv(path, index=False, columns=cols_to_save)

def save_plot(df, event_indices_for_plot, png_path, title, add_text=True):
    """Zaman serisini grafikleştirir ve kırılma/anomali noktalarını işaretler."""
    plt.figure(figsize=(12, 6))
    plt.plot(df["data"], lw=0.8, color='blue', label='Time Series Data')
    
    # Olay noktalarını işaretle
    for event_type, indices in event_indices_for_plot.items():
        if event_type == 'anomaly':
            color = 'red'
            marker = 'o'
            label_prefix = 'Anomaly'
            # Anomaly için boş daire (sadece kenar)
            plt.scatter(indices, df.loc[indices, "data"], 
                        marker=marker, s=80, color=color, 
                        facecolors="none", edgecolors=color,
                        linewidth=2, label=f'{label_prefix} Points')
        elif event_type == 'break':
            color = 'green'
            marker = 'x'
            label_prefix = 'Break'
            # Break için sadece color (x marker unfilled olduğu için)
            plt.scatter(indices, df.loc[indices, "data"], 
                        marker=marker, s=80, color=color, 
                        linewidth=2, label=f'{label_prefix} Points')
        else:
            continue

        if indices and add_text and len(indices) < 5: 
            for i, idx in enumerate(indices):
                plt.annotate(f"{label_prefix} {i+1}", (idx, df.loc[idx, "data"]), 
                             xytext=(10, 10 if event_type == 'break' else -20), 
                             textcoords="offset points",
                             bbox=dict(boxstyle="round,pad=0.3", 
                                       fc="lightgreen" if event_type == 'break' else "lightcoral", alpha=0.7))
    
    plt.title(title)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(png_path, dpi=120)
    plt.close() # *** ÖNEMLİ: Figure'ı kapat - memory leak'i önlemek için ***

def extract_char_name_from_function(function_name):
    """Fonksiyon adından karakteristik tipini çıkarır."""
    if "point_anomaly" in function_name:
        return "Point Anomaly"
    elif "collective_anomalies" in function_name:
        return "Collective Anomaly"
    elif "contextual_anomalies" in function_name:
        return "Contextual Anomaly"
    elif "mean_shift" in function_name:
        return "Mean Shift"
    elif "variance_shift" in function_name:
        return "Variance Shift"
    elif "trend_shift" in function_name:
        return "Trend Shift"
    elif "deterministic_trend_linear" in function_name:
        return "Linear Trend"
    elif "deterministic_trend_quadratic" in function_name:
        return "Quadratic Trend"
    elif "deterministic_trend_cubic" in function_name:
        return "Cubic Trend"
    elif "deterministic_trend_exponential" in function_name:
        return "Exponential Trend"
    elif "deterministic_trend_damped" in function_name:
        return "Damped Trend"
    elif "single_seasonality" in function_name:
        return "Single Seasonality"
    elif "multiple_seasonality" in function_name:
        return "Multiple Seasonality"
    else:
        return function_name.replace("generate_", "").replace("_", " ").title()

# --- Ana Akış ---------------------------------------------------------------
def main():
    if not CASES_YAML.exists():
        print(f"Hata: '{CASES_YAML}' bulunamadi. Lutfen once 'generate_multi_cases_yaml.py' calistirin.")
        return

    with open(CASES_YAML, 'r') as f:
        cases_config = yaml.safe_load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_generated = 0

    for case_spec in cases_config:
        case_name = case_spec["name"]
        base_model_info = case_spec["base_model_info"]
        characteristics_to_apply = case_spec["characteristics"]
        output_folder_name = case_spec["output_folder"]
        num_series_per_case = case_spec.get("num_series_per_case", N_PER_CASE)
        base_plot_title = case_spec.get("base_plot_title", case_name)

        case_output_dir = OUTPUT_DIR / output_folder_name
        case_output_dir.mkdir(parents=True, exist_ok=True)

        successful_generations = 0
        attempts_for_case = 0

        print(f"\n--- Uretiliyor: {case_name} ---")

        while successful_generations < num_series_per_case and attempts_for_case < num_series_per_case * 5:
            attempts_for_case += 1
            length = random.randint(200, 800)

            ts_gen = TimeSeriesGenerator(length=length)
            
            # 1. Temel seri üretimi (ORİJİNAL tsgen.py'ye göre uyarlanmıştır)
            series_data = None
            if base_model_info["type"] == "Stationary":
                base_df = ts_gen.generate_stationary_base_series(distribution=base_model_info["distribution"])
                series_data = base_df['data']
            elif base_model_info["type"] == "Stochastic":
                base_df = ts_gen.generate_stochastic_trend(kind=base_model_info["distribution"])
                series_data = base_df['data']
            elif base_model_info["type"] == "Seasonal":
                if base_model_info["distribution"] == "sarma":
                    base_df = ts_gen.generate_seasonality_from_base_series(kind="sarma")
                    series_data = base_df['data']
                elif base_model_info["distribution"] == "sarima":
                    base_df = ts_gen.generate_seasonality_from_base_series(kind="sarima")
                    series_data = base_df['data']
            elif base_model_info["type"] == "Volatility":
                if base_model_info["distribution"] == "arch":
                    base_df = ts_gen.generate_volatility(kind="arch")
                    series_data = base_df['data']
                elif base_model_info["distribution"] in ["garch", "egarch", "aparch"]:
                    base_df = ts_gen.generate_volatility(kind="garch")
                    series_data = base_df['data']
                
            if series_data is None or (isinstance(series_data, np.ndarray) and not np.any(np.isfinite(series_data))):
                print(f"Uyari: Temel seri uretimi basarisiz oldu ({base_model_info['type']}-{base_model_info['distribution']}). White noise ile devam ediliyor.")
                series_data = ts_gen.generate_white_noise(length=length)

            df = empty_frame(length)
            df['data'] = series_data

            if base_model_info["type"] == "Stationary": 
                df['stationary'] = 1
            elif base_model_info["type"] == "Stochastic": 
                df['stoc_trend'] = 1
                df['is_stochastic_base_active'] = 1
            elif base_model_info["type"] == "Seasonal": 
                df['single_seas'] = 1
                df['is_seasonal_base_active'] = 1
            elif base_model_info["type"] == "Volatility": 
                df['volatility'] = 1
                df['is_volatility_base_active'] = 1

            current_event_indices = {'anomaly': [], 'break': []}
            all_successful_characteristics = True
            applied_char_names = []

            # 2. Karakteristikleri sırasıyla uygula
            for i, char_item in enumerate(characteristics_to_apply):
                char_function_name = char_item["function"]
                char_params = char_item["params"].copy()
                output_flag = char_item["output_flag"]
                plot_type = char_item.get("plot_type")

                char_function = getattr(ts_gen, char_function_name)
                
                try:
                    result = None
                    info_returned = None

                    if char_function_name in ["generate_single_seasonality", "generate_multiple_seasonality"]:
                        # Original tsgen.py returns (period, df) or (periods, df)
                        _, result_df = char_function(df, **char_params)
                        result = result_df
                    elif char_function_name in ["generate_point_anomaly", "generate_point_anomalies",
                                                "generate_collective_anomalies", "generate_contextual_anomalies",
                                                "generate_mean_shift", "generate_variance_shift", "generate_trend_shift"]:
                        # Bu fonksiyonlar (df, info) döndürür
                        result, info_returned = char_function(df, **char_params)
                    else:
                        # Diğer tüm fonksiyonlar (trendler) sadece df döndürür
                        result = char_function(df, **char_params)

                    df_updated = None
                    if isinstance(result, pd.DataFrame):
                        df_updated = result
                    elif isinstance(result, np.ndarray):
                        # numpy array döndürdüğünde, df'in diğer kolonlarını koru
                        temp_df = pd.DataFrame({'data': result})
                        for col in df.columns:
                            if col != 'data':
                                temp_df[col] = df[col]
                        df_updated = temp_df
                    else:
                        raise ValueError(f"Beklenmeyen fonksiyon donus tipi: {type(result)}")

                    if df_updated is None:
                        raise ValueError("Fonksiyon DF dondurmedi veya None dondurdu.")
                    
                    # Ana DataFrame'i güncelle
                    df = df_updated 

                    # Karakteristik adını kaydet
                    applied_char_names.append(extract_char_name_from_function(char_function_name))

                    # Event indices toplama (sadece anomali/kırılma fonksiyonları için)
                    if info_returned:
                        if char_function_name in ["generate_point_anomaly", "generate_point_anomalies"]:
                            for item in info_returned:
                                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], np.ndarray):
                                    current_event_indices['anomaly'].extend(item[1].tolist())
                            # point_anom_point ve point_anom_location güncellemeleri
                            if 'point_anom_point' in df.columns and hasattr(df_updated, 'point_anom_point') and isinstance(df_updated['point_anom_point'].iloc[0], list):
                                df['point_anom_point'] = df_updated['point_anom_point'].apply(lambda x: str(x)).iloc[0]
                            if 'point_anom_location' in df.columns and hasattr(df_updated, 'point_anom_location'):
                                if isinstance(df_updated['point_anom_location'].iloc[0], str):
                                    df['point_anom_location'] = df_updated['point_anom_location'].iloc[0]
                                else:
                                    df['point_anom_location'] = str(df_updated['point_anom_location'].iloc[0])

                        elif char_function_name == "generate_collective_anomalies":
                             current_event_indices['anomaly'].extend([item[0] for item in info_returned if isinstance(item, tuple) and len(item) == 2])
                             if 'collect_anom_start' in df.columns and hasattr(df_updated, 'collect_anom_start') and isinstance(df_updated['collect_anom_start'].iloc[0], list):
                                 df['collect_anom_start'] = df_updated['collect_anom_start'].apply(lambda x: str(x)).iloc[0]
                                 df['collect_anom_end'] = df_updated['collect_anom_end'].apply(lambda x: str(x)).iloc[0]

                        elif char_function_name == "generate_contextual_anomalies":
                             current_event_indices['anomaly'].extend([item[0] for item in info_returned if isinstance(item, tuple) and len(item) == 2])

                        elif char_function_name in ["generate_mean_shift", "generate_variance_shift", "generate_trend_shift"]:
                            for item in info_returned:
                                if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], list):
                                    current_event_indices['break'].extend(item[1])
                            # Shift bilgileri için
                            if 'mean_shift_point' in df.columns and hasattr(df_updated, 'mean_shift_point') and isinstance(df_updated['mean_shift_point'].iloc[0], list):
                                df['mean_shift_point'] = df_updated['mean_shift_point'].apply(lambda x: str(x)).iloc[0]
                            if 'variance_shift_points' in df.columns and hasattr(df_updated, 'variance_shift_points') and isinstance(df_updated['variance_shift_points'].iloc[0], list):
                                df['variance_shift_points'] = df_updated['variance_shift_points'].apply(lambda x: str(x)).iloc[0]
                            if 'trend_shift_points' in df.columns and hasattr(df_updated, 'trend_shift_points') and isinstance(df_updated['trend_shift_points'].iloc[0], list):
                                df['trend_shift_points'] = df_updated['trend_shift_points'].apply(lambda x: str(x)).iloc[0]
                    
                    # İlgili karakteristik bayrağını işaretle
                    if output_flag in df.columns:
                        df[output_flag] = 1
                    
                    # Detaylı karakteristik bayraklarını işaretle (orijinal tsgen.py'deki isimlendirmeye göre)
                    if char_function_name == "generate_deterministic_trend_linear": 
                        df['det_lin_up'] = 1 if char_params.get('sign') == 1 else 0
                        df['det_lin_down'] = 1 if char_params.get('sign') == -1 else 0
                    elif char_function_name == "generate_deterministic_trend_quadratic": df['det_quad'] = 1
                    elif char_function_name == "generate_deterministic_trend_cubic": df['det_cubic'] = 1
                    elif char_function_name == "generate_deterministic_trend_exponential": df['det_exp'] = 1
                    elif char_function_name == "generate_deterministic_trend_damped": df['det_damped'] = 1
                    
                    elif char_function_name == "generate_point_anomaly": df['point_anom_single'] = 1
                    elif char_function_name == "generate_point_anomalies": df['point_anom_multi'] = 1
                    elif char_function_name == "generate_collective_anomalies": df['collect_anom'] = 1
                    elif char_function_name == "generate_contextual_anomalies": df['context_anom'] = 1
                    elif char_function_name == "generate_mean_shift": df['mean_shift'] = 1
                    elif char_function_name == "generate_variance_shift": df['var_shift'] = 1
                    elif char_function_name == "generate_trend_shift": 
                        df['trend_shift'] = 1
                        df['trend_shift_type'] = char_params.get('change_type', 'unknown')
                    elif char_function_name == "generate_single_seasonality": df['single_seas'] = 1
                    elif char_function_name == "generate_multiple_seasonality": df['multiple_seas'] = 1

                except Exception as e:
                    print(f"Hata: '{char_function_name}' uygulanirken hata olustu: {e}")
                    all_successful_characteristics = False
                    break

            if not all_successful_characteristics:
                continue

            # Karakteristik tiplerini ayarla
            if applied_char_names:
                df['characteristic_1_type'] = applied_char_names[0]
                if len(applied_char_names) > 1:
                    df['characteristic_2_type'] = applied_char_names[1]
                    df['is_multi_characteristic'] = 1
                else:
                    df['is_multi_characteristic'] = 0

            df = add_dates(df)

            file_prefix = f"{successful_generations:02d}"
            csv_filename = case_output_dir / f"{output_folder_name}_{file_prefix}.csv"
            png_filename = case_output_dir / f"{output_folder_name}_{file_prefix}.png"

            save_csv(df, csv_filename)

            current_event_indices['anomaly'] = sorted(list(set(current_event_indices['anomaly'])))
            current_event_indices['break'] = sorted(list(set(current_event_indices['break'])))
            save_plot(df, current_event_indices, png_filename, base_plot_title)

            successful_generations += 1
            total_generated += 1
            print(f"  [OK] {csv_filename.name} basariyla uretildi. ({successful_generations}/{num_series_per_case})")

        if successful_generations < num_series_per_case:
            print(f"Uyari: '{case_name}' icin istenen {num_series_per_case} seriden yalnizca {successful_generations} adet uretilebildi.")

    print(f"\n--- Tum uretim tamamlandi. Toplam {total_generated} seri uretildi. ---")
    print(f"Ciktilar: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()