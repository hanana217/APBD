"""
Générateur de Dataset  - Prédiction Requêtes SQL Lentes
===========================================================
Dataset réaliste pour entraîner un modèle de classification.

IMPORTANT: La lenteur NE dépend PAS de la longueur de la requête.
Elle dépend de facteurs DB réalistes (index, rows, joins, etc.)
"""

import numpy as np
import pandas as pd
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Seed pour reproductibilité
np.random.seed(42)


class SQLQueryDatasetGenerator:
    """Génère un dataset réaliste de requêtes SQL avec label slow/fast."""

    def __init__(self, n_samples: int = 3000, slow_ratio: float = 0.45):
        """
        Args:
            n_samples: Nombre de lignes à générer
            slow_ratio: Proportion de requêtes lentes (0.4-0.5 recommandé)
        """
        self.n_samples = n_samples
        self.slow_ratio = slow_ratio
        self.noise_flip_rate = 0.05  # 5% de bruit réaliste

    def _generate_query_complexity_features(self) -> dict:
        """Génère les features de complexité de requête."""
        n = self.n_samples

        # Longueur requête (log) - DÉCORRÉLÉE de la lenteur
        # On génère indépendamment pour casser la corrélation
        query_length_log = np.random.uniform(3.5, 8.5, n)  # log(30) à log(5000)

        # Nombre de JOINs (0-6)
        num_joins = np.random.choice([0, 1, 2, 3, 4, 5, 6], n,
                                      p=[0.25, 0.30, 0.20, 0.12, 0.08, 0.03, 0.02])

        # Nombre de tables (1-8)
        num_tables = np.clip(num_joins + np.random.randint(1, 3, n), 1, 8)

        # Nombre de conditions WHERE (0-10)
        num_where = np.random.poisson(3, n)
        num_where = np.clip(num_where, 0, 10)

        # Clauses spéciales
        has_group_by = np.random.binomial(1, 0.30, n)
        has_order_by = np.random.binomial(1, 0.45, n)
        has_limit = np.random.binomial(1, 0.40, n)

        # Agrégats (0-5)
        num_aggregates = np.where(has_group_by == 1,
                                   np.random.poisson(2, n),
                                   np.random.binomial(1, 0.2, n))
        num_aggregates = np.clip(num_aggregates, 0, 5)

        # Sous-requêtes (0-3)
        num_subqueries = np.random.choice([0, 1, 2, 3], n,
                                           p=[0.65, 0.25, 0.08, 0.02])

        return {
            'query_length_log': query_length_log,
            'num_joins': num_joins,
            'num_tables': num_tables,
            'num_where': num_where,
            'has_group_by': has_group_by,
            'has_order_by': has_order_by,
            'has_limit': has_limit,
            'num_aggregates': num_aggregates,
            'num_subqueries': num_subqueries
        }

    def _generate_db_simulation_features(self) -> dict:
        """Génère les features simulant l'état de la DB."""
        n = self.n_samples

        # Taille estimée des tables (MB) - distribution log-normale
        # Petites tables (1-10MB) fréquentes, grandes tables (1000MB+) rares
        estimated_table_size_mb = np.exp(np.random.normal(4, 2, n))
        estimated_table_size_mb = np.clip(estimated_table_size_mb, 0.1, 10000)

        # Rows examined - corrélé à la taille table + absence index
        base_rows = estimated_table_size_mb * np.random.uniform(50, 200, n)
        rows_examined = np.clip(base_rows, 10, 50_000_000).astype(int)

        # Index utilisé (0/1) - crucial pour la performance
        # Plus la table est grande, plus important d'avoir un index
        index_probability = np.where(estimated_table_size_mb > 100, 0.7, 0.5)
        has_index_used = np.random.binomial(1, index_probability)

        # Nombre d'index sur la table (0-8)
        index_count = np.where(has_index_used == 1,
                                np.random.poisson(3, n),
                                np.random.poisson(1, n))
        index_count = np.clip(index_count, 0, 8)

        return {
            'estimated_table_size_mb': np.round(estimated_table_size_mb, 2),
            'rows_examined': rows_examined,
            'has_index_used': has_index_used,
            'index_count': index_count
        }

    def _generate_server_context_features(self) -> dict:
        """Génère les features de contexte serveur."""
        n = self.n_samples

        # Buffer pool hit ratio (0.3 à 0.99)
        # La plupart des DB bien configurées ont un ratio élevé
        buffer_pool_hit_ratio = np.random.beta(8, 2, n) * 0.69 + 0.30
        buffer_pool_hit_ratio = np.clip(buffer_pool_hit_ratio, 0.30, 0.99)

        # Nombre de connexions actives (1-50)
        connections_count = np.random.poisson(15, n)
        connections_count = np.clip(connections_count, 1, 50)

        # Heure de pointe (0/1)
        is_peak_hour = np.random.binomial(1, 0.35, n)

        return {
            'buffer_pool_hit_ratio': np.round(buffer_pool_hit_ratio, 3),
            'connections_count': connections_count,
            'is_peak_hour': is_peak_hour
        }

    def _compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features dérivées."""

        # Joins par table
        df['joins_per_table'] = np.where(
            df['num_tables'] > 0,
            df['num_joins'] / df['num_tables'],
            0
        )
        df['joins_per_table'] = np.round(df['joins_per_table'], 3)

        # Score de complexité composite
        df['complexity_score'] = (
            df['num_joins'] * 2 +
            df['num_subqueries'] * 3 +
            df['num_aggregates'] * 1.5 +
            df['has_group_by'] * 2 +
            df['has_order_by'] * 1
        )
        df['complexity_score'] = np.round(df['complexity_score'], 2)

        return df

    def _compute_slow_label(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calcule le label is_slow base sur des regles DB realistes.

        VERSION EQUILIBREE: L'index est important mais pas dominant.
        Plusieurs features contribuent de maniere significative.
        """
        n = len(df)
        slow_score = np.zeros(n)

        # Ajouter du bruit aleatoire pour rendre le probleme moins trivial
        noise = np.random.normal(0, 1.5, n)
        slow_score += noise

        # ==========================================
        # FACTEURS QUI RALENTISSENT (poids positifs)
        # ==========================================

        # 1. Rows examined - IMPACT FORT
        rows_log = np.log10(df['rows_examined'] + 1)
        slow_score += rows_log * 1.2  # Augmente

        # 2. Absence d'index - Impact MODERE (reduit)
        no_index_penalty = (1 - df['has_index_used']) * 1.5  # Reduit de 3.0 a 1.5
        slow_score += no_index_penalty

        # 3. Taille table - Impact FORT
        table_size_log = np.log10(df['estimated_table_size_mb'] + 1)
        slow_score += table_size_log * 0.8  # Augmente

        # 4. Nombre de JOINs - Impact FORT
        slow_score += df['num_joins'] * 1.0  # Augmente

        # 5. Sous-requetes - Impact FORT
        slow_score += df['num_subqueries'] * 1.2  # Augmente

        # 6. GROUP BY - Impact modere
        slow_score += df['has_group_by'] * 1.5

        # 7. ORDER BY sans LIMIT - Impact fort
        order_no_limit = df['has_order_by'] * (1 - df['has_limit'])
        slow_score += order_no_limit * table_size_log * 0.5

        # 8. Agregats multiples
        slow_score += df['num_aggregates'] * 0.5

        # 9. Peak hour - Impact contextuel
        slow_score += df['is_peak_hour'] * 1.2

        # 10. Connexions elevees - Impact
        connections_factor = (df['connections_count'] - 15) / 35
        slow_score += np.clip(connections_factor, 0, 1) * 1.0

        # 11. Faible buffer hit ratio - Impact
        low_buffer = (1 - df['buffer_pool_hit_ratio']) * 2.0
        slow_score += low_buffer

        # 12. Complexite combinee (joins * tables)
        slow_score += df['joins_per_table'] * 1.5

        # ==========================================
        # FACTEURS QUI ACCELERENT (poids negatifs)
        # ==========================================

        # 1. LIMIT present - Reduit significativement
        slow_score -= df['has_limit'] * 2.5  # Augmente impact

        # 2. Index utilise - Impact MODERE (reduit)
        slow_score -= df['has_index_used'] * 1.2  # Reduit de 2.5 a 1.2

        # 3. Peu de rows avec index
        efficient_query = (df['rows_examined'] < 10000) & (df['has_index_used'] == 1)
        slow_score -= efficient_query.astype(int) * 1.5

        # 4. Requete simple (peu de joins, pas de subquery)
        simple_query = (df['num_joins'] <= 1) & (df['num_subqueries'] == 0)
        slow_score -= simple_query.astype(int) * 1.0

        # ==========================================
        # NORMALISATION ET SEUIL
        # ==========================================

        # Normaliser le score
        slow_score_normalized = (slow_score - slow_score.min()) / (slow_score.max() - slow_score.min())

        # Determiner le seuil pour obtenir le ratio slow souhaite
        threshold = np.percentile(slow_score_normalized, (1 - self.slow_ratio) * 100)

        # Creer le label
        is_slow = (slow_score_normalized > threshold).astype(int)

        # ==========================================
        # AJOUT DE BRUIT REALISTE (augmente)
        # ==========================================

        # Flip aleatoire de 8% des labels pour simuler variabilite reelle
        flip_rate = 0.08  # Augmente de 5% a 8%
        flip_mask = np.random.binomial(1, flip_rate, n).astype(bool)
        is_slow[flip_mask] = 1 - is_slow[flip_mask]

        return is_slow, slow_score_normalized

    def _force_counterexamples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Force des cas contre-intuitifs pour:
        1. Casser la correlation longueur
        2. Casser la dominance de l'index

        4 types de contre-exemples pour equilibrer le dataset.
        """
        n = len(df)
        n_per_type = int(n * 0.08)  # 8% par type = 32% total

        used_indices = set()

        # ===== CAS 1: Courtes mais lentes (pas d'index) =====
        available = list(set(range(n)) - used_indices)
        indices_type1 = np.random.choice(available, n_per_type, replace=False)
        used_indices.update(indices_type1)

        for idx in indices_type1:
            df.loc[idx, 'query_length_log'] = np.random.uniform(3.5, 4.5)
            df.loc[idx, 'estimated_table_size_mb'] = np.random.uniform(1000, 8000)
            df.loc[idx, 'has_index_used'] = 0
            df.loc[idx, 'index_count'] = 0
            df.loc[idx, 'rows_examined'] = np.random.randint(5_000_000, 40_000_000)
            df.loc[idx, 'is_slow'] = 1

        # ===== CAS 2: Longues mais rapides (bon index + limit) =====
        available = list(set(range(n)) - used_indices)
        indices_type2 = np.random.choice(available, n_per_type, replace=False)
        used_indices.update(indices_type2)

        for idx in indices_type2:
            df.loc[idx, 'query_length_log'] = np.random.uniform(7.0, 8.5)
            df.loc[idx, 'num_joins'] = np.random.randint(3, 6)
            df.loc[idx, 'num_subqueries'] = np.random.randint(1, 3)
            df.loc[idx, 'has_index_used'] = 1
            df.loc[idx, 'index_count'] = np.random.randint(4, 8)
            df.loc[idx, 'has_limit'] = 1
            df.loc[idx, 'estimated_table_size_mb'] = np.random.uniform(10, 100)
            df.loc[idx, 'rows_examined'] = np.random.randint(100, 10000)
            df.loc[idx, 'is_slow'] = 0

        # ===== CAS 3: Index present MAIS lent (table enorme, beaucoup de rows) =====
        available = list(set(range(n)) - used_indices)
        indices_type3 = np.random.choice(available, n_per_type, replace=False)
        used_indices.update(indices_type3)

        for idx in indices_type3:
            df.loc[idx, 'has_index_used'] = 1
            df.loc[idx, 'index_count'] = np.random.randint(2, 5)
            # Mais facteurs qui ralentissent malgre l'index
            df.loc[idx, 'estimated_table_size_mb'] = np.random.uniform(2000, 8000)
            df.loc[idx, 'rows_examined'] = np.random.randint(1_000_000, 20_000_000)
            df.loc[idx, 'num_joins'] = np.random.randint(4, 7)
            df.loc[idx, 'has_group_by'] = 1
            df.loc[idx, 'has_order_by'] = 1
            df.loc[idx, 'has_limit'] = 0
            df.loc[idx, 'is_peak_hour'] = 1
            df.loc[idx, 'is_slow'] = 1

        # ===== CAS 4: Pas d'index MAIS rapide (petite table, limit) =====
        available = list(set(range(n)) - used_indices)
        indices_type4 = np.random.choice(available, n_per_type, replace=False)
        used_indices.update(indices_type4)

        for idx in indices_type4:
            df.loc[idx, 'has_index_used'] = 0
            df.loc[idx, 'index_count'] = 0
            # Mais facteurs qui accelerent malgre absence d'index
            df.loc[idx, 'estimated_table_size_mb'] = np.random.uniform(0.1, 5)
            df.loc[idx, 'rows_examined'] = np.random.randint(10, 1000)
            df.loc[idx, 'num_joins'] = 0
            df.loc[idx, 'num_subqueries'] = 0
            df.loc[idx, 'has_limit'] = 1
            df.loc[idx, 'buffer_pool_hit_ratio'] = np.random.uniform(0.92, 0.99)
            df.loc[idx, 'is_slow'] = 0

        return df

    def generate(self) -> pd.DataFrame:
        """Génère le dataset complet."""

        # 1. Générer toutes les features
        complexity = self._generate_query_complexity_features()
        db_sim = self._generate_db_simulation_features()
        server = self._generate_server_context_features()

        # 2. Créer le DataFrame
        df = pd.DataFrame({**complexity, **db_sim, **server})

        # 3. Calculer features dérivées
        df = self._compute_derived_features(df)

        # 4. Calculer le label
        df['is_slow'], df['_slow_score'] = self._compute_slow_label(df)

        # 5. Forcer les contre-exemples
        df = self._force_counterexamples(df)

        # 6. Réordonner les colonnes
        column_order = [
            # Complexité requête
            'query_length_log', 'num_joins', 'num_tables', 'num_where',
            'has_group_by', 'has_order_by', 'has_limit',
            'num_aggregates', 'num_subqueries',
            # DB simulation
            'estimated_table_size_mb', 'rows_examined',
            'has_index_used', 'index_count',
            # Contexte serveur
            'buffer_pool_hit_ratio', 'connections_count', 'is_peak_hour',
            # Features dérivées
            'joins_per_table', 'complexity_score',
            # Target
            'is_slow'
        ]

        df = df[column_order]

        # 7. Mélanger les lignes
        df = df.sample(frac=1).reset_index(drop=True)

        return df

    def validate_dataset(self, df: pd.DataFrame) -> dict:
        """Valide la qualité du dataset généré."""

        stats = {
            'n_samples': len(df),
            'slow_ratio': df['is_slow'].mean(),
            'fast_ratio': 1 - df['is_slow'].mean(),
        }

        # Corrélation longueur vs slow (doit être FAIBLE)
        length_slow_corr = df['query_length_log'].corr(df['is_slow'])
        stats['length_slow_correlation'] = round(length_slow_corr, 3)

        # Corrélation rows_examined vs slow (doit être FORTE)
        rows_slow_corr = df['rows_examined'].corr(df['is_slow'])
        stats['rows_slow_correlation'] = round(rows_slow_corr, 3)

        # Corrélation index vs slow (doit être NÉGATIVE)
        index_slow_corr = df['has_index_used'].corr(df['is_slow'])
        stats['index_slow_correlation'] = round(index_slow_corr, 3)

        # Vérifier les contre-exemples
        short_slow = df[(df['query_length_log'] < 5) & (df['is_slow'] == 1)]
        long_fast = df[(df['query_length_log'] > 7) & (df['is_slow'] == 0)]

        stats['short_slow_count'] = len(short_slow)
        stats['long_fast_count'] = len(long_fast)

        return stats


def main():
    """Genere et sauvegarde le dataset."""

    print("=" * 60)
    print("GENERATEUR DE DATASET - REQUETES SQL")
    print("=" * 60)

    # Configuration
    N_SAMPLES = 20000  # Nombre de requêtes à générer
    SLOW_RATIO = 0.45  # 45% slow, 55% fast

    # Generation
    print(f"\n[*] Generation de {N_SAMPLES} echantillons...")
    generator = SQLQueryDatasetGenerator(n_samples=N_SAMPLES, slow_ratio=SLOW_RATIO)
    df = generator.generate()

    # Validation
    print("\n[OK] Validation du dataset:")
    stats = generator.validate_dataset(df)

    print(f"   - Taille: {stats['n_samples']} lignes")
    print(f"   - Distribution: {stats['slow_ratio']:.1%} slow / {stats['fast_ratio']:.1%} fast")
    print(f"   - Correlation longueur<->slow: {stats['length_slow_correlation']:.3f} (doit etre ~0)")
    print(f"   - Correlation rows<->slow: {stats['rows_slow_correlation']:.3f} (doit etre >0.3)")
    print(f"   - Correlation index<->slow: {stats['index_slow_correlation']:.3f} (doit etre <0)")
    print(f"   - Contre-exemples courts+lents: {stats['short_slow_count']}")
    print(f"   - Contre-exemples longs+rapides: {stats['long_fast_count']}")

    # Sauvegarde
    output_path = "sql_query_performance_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[SAVE] Dataset sauvegarde: {output_path}")

    # Apercu
    print("\n[INFO] Apercu des premieres lignes:")
    print(df.head(10).to_string())

    # Stats descriptives
    print("\n[STATS] Statistiques descriptives:")
    print(df.describe().round(2).to_string())

    print("\n" + "=" * 60)
    print("[OK] GENERATION TERMINEE")
    print("=" * 60)

    return df


if __name__ == "__main__":
    df = main()
