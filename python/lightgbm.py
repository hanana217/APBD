# simple_export.py
import pandas as pd
import mysql.connector

DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3308,
    'user': 'apbd_user',
    'password': 'apbd_pass',
    'database': 'pos'
}

def simple_export():
    """
    Export simple de la table dataset vers CSV
    """
    print("📁 Chargement depuis la table 'dataset'...")
    
    conn = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql("SELECT * FROM dataset", conn)
    conn.close()
    
    print(f"✅ Données chargées: {df.shape}")
    
    # Nettoyage basique
    df = df.dropna(subset=['is_slow'])
    
    # Remplacer les NaN
    df = df.fillna(0)
    
    # Sauvegarde
    output_path = "../data/exports/dataset_lightgbm.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n💾 Dataset sauvegardé: {output_path}")
    print(f"📊 Dimensions: {df.shape}")
    print(f"🎯 Colonnes: {len(df.columns)}")
    
    # Distribution de la target
    slow_count = df['is_slow'].sum()
    fast_count = len(df) - slow_count
    slow_percentage = (slow_count / len(df)) * 100
    
    print(f"\n📈 Distribution de is_slow:")
    print(f"   FAST (0): {fast_count} requêtes ({100-slow_percentage:.1f}%)")
    print(f"   SLOW (1): {slow_count} requêtes ({slow_percentage:.1f}%)")
    
    # Liste des colonnes
    print(f"\n📋 Colonnes exportées:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2}. {col}")
    
    return df

if __name__ == "__main__":
    import os
    os.makedirs("../data/exports", exist_ok=True)
    
    print("="*50)
    print("🚀 EXPORT SIMPLE POUR LIGHTGBM/XGBOOST")
    print("="*50)
    
    df = simple_export()
    
    print("\n" + "="*50)
    print("✅ EXPORT TERMINÉ!")
    print("="*50)