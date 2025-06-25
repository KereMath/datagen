import pandas as pd
import os

def filter_time_series_combinations():
    """
    time_series_valid_combinations_final.csv dosyasını filtreler
    SADECE Characteristic 1 sütununda belirtilen anomali/break türlerini içeren satırları bırakır
    """
    
    # Dosya yolları
    input_file = r"C:\Users\user\Desktop\data-gen\time_series_valid_combinations_final.csv"
    output_file = r"C:\Users\user\Desktop\data-gen\output_combinations.csv"
    
    # Filtrelenecek terimler
    keep_terms = [
        "Break: Variance Shift",
        "Break: Trend Shift", 
        "Anomaly: Point",
        "Break: Mean Shift",
        "Anomaly: Collective",
        "Anomaly: Contextual"
    ]
    
    print("📖 CSV dosyası okunuyor...")
    
    try:
        # CSV'yi oku
        df = pd.read_csv(input_file, sep=';')
        print(f"✅ {len(df)} satır okundu")
        
        # Sütun isimlerini kontrol et
        print(f"📋 Sütunlar: {list(df.columns)}")
        
        # Filtreleme: SADECE Characteristic 1'de istenen terimler var mı?
        mask = pd.Series([False] * len(df))
        
        for term in keep_terms:
            # Sadece Characteristic 1 sütununda ara
            mask_char1 = df['Characteristic 1'].str.contains(term, na=False)
            
            # Term geçen satırları işaretle
            mask = mask | mask_char1
            
            print(f"🔍 '{term}' Characteristic 1'de bulunan satırlar: {mask_char1.sum()}")
        
        # Filtrelenmiş dataframe
        filtered_df = df[mask].copy()
        
        print(f"\n📊 Filtreleme sonucu:")
        print(f"   • Orijinal satır sayısı: {len(df)}")
        print(f"   • Filtrelenmiş satır sayısı: {len(filtered_df)}")
        print(f"   • Çıkarılan satır sayısı: {len(df) - len(filtered_df)}")
        
        # Filtrelenmiş veriyi kaydet
        filtered_df.to_csv(output_file, index=False, sep=';')
        print(f"\n✅ Filtrelenmiş veri kaydedildi: {output_file}")
        
        # Özet istatistikler
        print(f"\n📈 Kalan tür dağılımı (Characteristic 1):")
        for term in keep_terms:
            char1_count = filtered_df['Characteristic 1'].str.contains(term, na=False).sum()
            print(f"   • {term}: {char1_count} satır")
            
    except FileNotFoundError:
        print(f"❌ Hata: Dosya bulunamadı - {input_file}")
        print("📂 Dosya yolunu kontrol edin!")
        
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {e}")

if __name__ == "__main__":
    print("🔧 Time Series Combinations Filtreleme Başlıyor...\n")
    filter_time_series_combinations()
    print("\n🎯 İşlem tamamlandı!")