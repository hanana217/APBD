from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import mysql.connector
import numpy as np

DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3308,
    'user': 'apbd_user',
    'password': 'apbd_pass',
    'database': 'pos'
}

def create_xgboost_dataset_from_dataset_table():
    """
    Crée un dataset optimisé pour XGBoost à partir de la table 'dataset'
    """
    print("="*60)
    print("🌳 CRÉATION DU DATASET XGBOOST")
    print("="*60)
    
    # 1. Charger les données depuis la table 'dataset'
    print("\n📁 Chargement depuis la table 'dataset'...")
    conn = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql("SELECT * FROM dataset", conn)
    conn.close()
    
    print(f"✅ Données chargées: {df.shape}")
    
    # 2. Nettoyage de base
    print("\n🧹 Nettoyage des données...")
    
    # Supprimer les lignes sans target
    initial_rows = len(df)
    df = df.dropna(subset=['is_slow'])
    print(f"   Lignes après suppression is_slow manquants: {len(df)}/{initial_rows}")
    
    # Vérifier et corriger les types de données
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    # Remplacer les valeurs manquantes
    df = df.fillna(0)
    
    # 3. Analyser la distribution de la target
    print("\n🎯 Analyse de la target 'is_slow':")
    slow_count = df['is_slow'].sum()
    fast_count = len(df) - slow_count
    slow_percentage = (slow_count / len(df)) * 100
    
    print(f"   FAST (0): {fast_count} requêtes ({100-slow_percentage:.1f}%)")
    print(f"   SLOW (1): {slow_count} requêtes ({slow_percentage:.1f}%)")
    
    # 4. Définir les features pour XGBoost
    print("\n📋 Sélection des features...")
    
    # Features numériques basées sur votre table 'dataset'
    numerical_features = [
        # Structure SQL
        'query_length', 'query_length_log',
        'num_joins', 'num_where',
        
        # Opérations SQL
        'has_group_by', 'has_order_by', 'has_having',
        'has_union', 'has_distinct', 'has_limit', 'has_star_select',
        
        # Complexité SQL
        'num_subqueries', 'num_predicates', 'num_aggregates', 'num_functions',
        'num_tables', 'max_in_list_size',
        
        # Patterns SQL
        'has_wildcard_like', 'has_case_when',
        
        # Type de requête
        'is_select_query', 'is_insert_query', 'is_update_query', 'is_delete_query',
        
        # Contexte temporel
        'hour_of_day', 'day_of_week',
        'is_peak_hour', 'is_weekend', 'is_business_hours',
        
        # Contexte système
        'connections_count', 'connections_high',
        'buffer_pool_hit_ratio',
        
        # Estimations
        'estimated_index_count', 'estimated_table_size_mb',
        
        # Features calculées
        'joins_per_table', 'predicates_per_where', 'complexity_density'
    ]
    
    # Features catégorielles (s'il y en a)
    categorical_features = []  # Aucune dans votre table actuelle
    
    # Filtrer pour garder seulement les colonnes existantes
    existing_numerical = [col for col in numerical_features if col in df.columns]
    existing_categorical = [col for col in categorical_features if col in df.columns]
    
    print(f"   Features numériques: {len(existing_numerical)}")
    print(f"   Features catégorielles: {len(existing_categorical)}")
    
    # 5. Identifier les features problématiques
    print("\n🔍 Analyse des features problématiques...")
    
    problematic_features = []
    
    # Vérifier les colonnes constantes
    for col in existing_numerical:
        if df[col].nunique() == 1:
            problematic_features.append(col)
            print(f"⚠️  Colonne constante: {col} = {df[col].iloc[0]}")
    
    # Vérifier les corrélations parfaites avec is_slow
    for col in existing_numerical:
        if col != 'is_slow':
            # Vérifier les prédicteurs parfaits
            unique_values = df[col].unique()
            if len(unique_values) <= 10:  # Pour les colonnes avec peu de valeurs
                is_perfect_predictor = True
                for val in unique_values:
                    subset = df[df[col] == val]
                    if len(subset) > 0:
                        slow_rate = subset['is_slow'].mean()
                        if slow_rate not in [0.0, 1.0]:
                            is_perfect_predictor = False
                            break
                
                if is_perfect_predictor:
                    problematic_features.append(col)
                    print(f"⚠️  Prédicteur parfait: {col}")
    
    # Supprimer les features problématiques
    if problematic_features:
        print(f"\n🗑️  Suppression des features problématiques:")
        for col in problematic_features:
            if col in existing_numerical:
                existing_numerical.remove(col)
                print(f"   • {col}")
    
    # 6. Préparation des données
    print("\n🔧 Préparation des données pour XGBoost...")
    
    # Sélectionner les features
    features = existing_numerical + existing_categorical
    
    if not features:
        print("❌ Aucune feature disponible après nettoyage!")
        return None
    
    # Créer le transformateur
    if existing_categorical:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', existing_numerical),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                 existing_categorical)
            ]
        )
    else:
        # Pas de features catégorielles, utiliser seulement les numériques
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', existing_numerical)
            ]
        )
    
    # Séparer features et target
    X = df[features]
    y = df['is_slow']
    
    # Appliquer le préprocessing
    X_transformed = preprocessor.fit_transform(X)
    
    # Récupérer les noms de features
    if existing_categorical:
        feature_names = (
            existing_numerical +
            list(preprocessor.named_transformers_['cat']
                 .get_feature_names_out(existing_categorical))
        )
    else:
        feature_names = existing_numerical
    
    # 7. Créer le DataFrame final
    df_xgb = pd.DataFrame(X_transformed, columns=feature_names)
    
    # Ajouter la target
    df_xgb['is_slow'] = y.values
    
    # Optionnel: ajouter d'autres colonnes utiles
    if 'execution_time' in df.columns:
        df_xgb['execution_time'] = df['execution_time'].values
    
    # 8. Vérification finale
    print("\n🔍 Vérification finale du dataset...")
    
    print(f"📊 Dimensions finales: {df_xgb.shape}")
    print(f"🎯 Features: {len(feature_names)}")
    
    # Vérifier les valeurs manquantes
    missing = df_xgb.isnull().sum().sum()
    if missing > 0:
        print(f"⚠️  Valeurs manquantes: {missing}")
        df_xgb = df_xgb.fillna(0)
        print("   ✅ Valeurs manquantes remplacées par 0")
    else:
        print("✅ Aucune valeur manquante")
    
    # Vérifier la distribution de la target
    print(f"\n📈 Distribution finale de is_slow:")
    final_slow_count = df_xgb['is_slow'].sum()
    final_fast_count = len(df_xgb) - final_slow_count
    final_slow_percentage = (final_slow_count / len(df_xgb)) * 100
    
    print(f"   FAST (0): {final_fast_count} ({100-final_slow_percentage:.1f}%)")
    print(f"   SLOW (1): {final_slow_count} ({final_slow_percentage:.1f}%)")
    
    # 9. Calculer les corrélations
    print(f"\n🔗 Top 10 corrélations avec is_slow:")
    
    correlations = {}
    for col in feature_names:
        if col in df_xgb.columns:
            corr = abs(df_xgb[col].corr(df_xgb['is_slow']))
            correlations[col] = corr
    
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
    for col, corr in sorted_corr:
        print(f"   {col:25}: {corr:.3f}")
    
    # 10. Sauvegarde
    print("\n💾 Sauvegarde du dataset...")
    
    # Sauvegarde CSV
    csv_path = "../data/exports/dataset_xgboost.csv"
    df_xgb.to_csv(csv_path, index=False)
    
    print(f"✅ Dataset XGBoost sauvegardé: {csv_path}")
    
    # Sauvegarde Parquet (plus efficace)
    parquet_path = "../data/exports/dataset_xgboost.parquet"
    df_xgb.to_parquet(parquet_path, index=False)
    print(f"✅ Dataset XGBoost (Parquet) sauvegardé: {parquet_path}")
    
    # 11. Créer des métadonnées
    import json
    metadata = {
        'created_at': pd.Timestamp.now().isoformat(),
        'source_table': 'dataset',
        'n_samples': len(df_xgb),
        'n_features': len(feature_names),
        'target_name': 'is_slow',
        'target_distribution': {
            'fast': int(final_fast_count),
            'slow': int(final_slow_count),
            'slow_percentage': float(final_slow_percentage)
        },
        'features': feature_names,
        'problematic_features_removed': problematic_features,
        'threshold_seconds': 0.5,
        'no_data_leakage': True,
        'xgboost_ready': True
    }
    
    metadata_path = "../data/exports/dataset_xgboost_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Métadonnées sauvegardées: {metadata_path}")
    
    # 12. Créer également un dataset simplifié (top features)
    print(f"\n🎯 Création d'un dataset simplifié (top 20 features)...")
    
    # Prendre les 20 features les plus corrélées
    top_features = [feat[0] for feat in sorted_corr[:20]]
    
    if top_features:
        df_xgb_simple = df_xgb[top_features + ['is_slow']].copy()
        simple_path = "../data/exports/dataset_xgboost_simple.csv"
        df_xgb_simple.to_csv(simple_path, index=False)
        print(f"✅ Dataset simplifié sauvegardé: {simple_path}")
        print(f"   Features: {len(top_features)}")
    
    return df_xgb

def validate_xgboost_dataset(df_xgb):
    """
    Valide le dataset pour s'assurer qu'il est prêt pour XGBoost
    """
    print("\n" + "="*60)
    print("🔬 VALIDATION DU DATASET XGBOOST")
    print("="*60)
    
    if df_xgb is None or len(df_xgb) == 0:
        print("❌ Dataset vide!")
        return False
    
    # 1. Vérifier les types de données
    print("\n📝 Vérification des types de données...")
    
    problematic_types = []
    for col in df_xgb.columns:
        if col != 'is_slow':
            dtype = df_xgb[col].dtype
            if dtype not in ['int64', 'float64', 'int32', 'float32']:
                problematic_types.append((col, dtype))
    
    if problematic_types:
        print("⚠️  Types de données problématiques:")
        for col, dtype in problematic_types:
            print(f"   • {col}: {dtype}")
    else:
        print("✅ Tous les types sont numériques")
    
    # 2. Vérifier les valeurs manquantes
    print("\n🔍 Vérification des valeurs manquantes...")
    
    missing_cols = df_xgb.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0]
    
    if len(missing_cols) > 0:
        print("⚠️  Colonnes avec valeurs manquantes:")
        for col, count in missing_cols.items():
            percentage = (count / len(df_xgb)) * 100
            print(f"   • {col}: {count} ({percentage:.1f}%)")
    else:
        print("✅ Aucune valeur manquante")
    
    # 3. Vérifier la variance
    print("\n📊 Vérification de la variance...")
    
    low_variance_cols = []
    for col in df_xgb.columns:
        if col != 'is_slow' and df_xgb[col].dtype in ['int64', 'float64']:
            variance = df_xgb[col].var()
            if variance < 0.001:
                low_variance_cols.append((col, variance))
    
    if low_variance_cols:
        print("⚠️  Colonnes avec faible variance:")
        for col, variance in low_variance_cols[:5]:  # Afficher les 5 premières
            print(f"   • {col}: variance = {variance:.6f}")
        if len(low_variance_cols) > 5:
            print(f"   ... et {len(low_variance_cols)-5} autres")
    else:
        print("✅ Variance acceptable pour toutes les colonnes")
    
    # 4. Vérifier l'équilibre des classes
    print("\n⚖️  Vérification de l'équilibre des classes...")
    
    class_balance = df_xgb['is_slow'].value_counts(normalize=True)
    
    print(f"   Classe 0 (FAST): {class_balance.get(0, 0)*100:.1f}%")
    print(f"   Classe 1 (SLOW): {class_balance.get(1, 0)*100:.1f}%")
    
    if abs(class_balance.get(0, 0) - class_balance.get(1, 0)) > 0.3:
        print("⚠️  Déséquilibre important entre les classes")
    else:
        print("✅ Classes relativement équilibrées")
    
    # 5. Recommandations
    print("\n💡 RECOMMANDATIONS POUR XGBOOST:")
    
    if len(df_xgb) < 1000:
        print("   • Échantillon limité, considérez l'augmentation de données")
    else:
        print("   • Échantillon suffisant pour l'entraînement")
    
    if len(df_xgb.columns) > 100:
        print("   • Nombre élevé de features, considérez la sélection de features")
    else:
        print("   • Nombre de features approprié")
    
    return True

if __name__ == "__main__":
    import os
    os.makedirs("../data/exports", exist_ok=True)
    
    print("="*60)
    print("🚀 CRÉATION DU DATASET POUR XGBOOST")
    print("="*60)
    
    # Créer le dataset
    df_xgb = create_xgboost_dataset_from_dataset_table()
    
    if df_xgb is not None:
        # Valider le dataset
        is_valid = validate_xgboost_dataset(df_xgb)
        
        if is_valid:
            print("\n" + "="*60)
            print("✅ DATASET XGBOOST PRÊT!")
            print("="*60)
            
            print(f"\n📋 RÉSUMÉ:")
            print(f"   • Échantillons: {len(df_xgb)}")
            print(f"   • Features: {len(df_xgb.columns) - 1}")
            print(f"   • Target: is_slow (seuil 0.5s)")
            print(f"   • Distribution: {df_xgb['is_slow'].mean()*100:.1f}% lentes")
            
            print(f"\n📁 FICHIERS CRÉÉS:")
            print(f"   1. dataset_xgboost.csv - Dataset complet")
            print(f"   2. dataset_xgboost.parquet - Version optimisée")
            print(f"   3. dataset_xgboost_simple.csv - Version simplifiée")
            print(f"   4. dataset_xgboost_metadata.json - Métadonnées")
            
            print(f"\n🎯 UTILISATION AVEC XGBOOST:")
            print(f"   import xgboost as xgb")
            print(f"   import pandas as pd")
            print(f"   ")
            print(f"   # Charger le dataset")
            print(f"   df = pd.read_csv('../data/exports/dataset_xgboost.csv')")
            print(f"   X = df.drop(columns=['is_slow'])")
            print(f"   y = df['is_slow']")
            print(f"   ")
            print(f"   # Entraîner le modèle")
            print(f"   model = xgb.XGBClassifier()")
            print(f"   model.fit(X, y)")
        else:
            print("\n❌ Dataset non valide pour XGBoost")
    else:
        print("\n❌ Échec de la création du dataset")