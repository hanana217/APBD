# SQL Query Performance Dataset

Dataset synthetique pour entrainer un modele de classification predisant si une requete SQL sera **lente** ou **rapide**.

## Objectif

Predire `is_slow` (0 = rapide, 1 = lent) a partir de caracteristiques realistes de requetes SQL.

**Important**: Ce dataset est concu pour eviter la correlation triviale `longueur requete = lent`. La lenteur depend de facteurs DB realistes.

---

## Structure du Dataset

| Fichier | Description |
|---------|-------------|
| `sql_query_performance_dataset.csv` | Dataset genere (20000 lignes) |
| `generate_aml_dataset.py` | Script de generation |
| `validate_dataset.py` | Script de validation ML |

---

## Features

### A. Complexite Requete

| Feature | Type | Description |
|---------|------|-------------|
| `query_length_log` | float | Log de la longueur de la requete (3.5-8.5) |
| `num_joins` | int | Nombre de JOINs (0-6) |
| `num_tables` | int | Nombre de tables impliquees (1-8) |
| `num_where` | int | Nombre de conditions WHERE (0-10) |
| `has_group_by` | 0/1 | Presence de GROUP BY |
| `has_order_by` | 0/1 | Presence de ORDER BY |
| `has_limit` | 0/1 | Presence de LIMIT |
| `num_aggregates` | int | Nombre de fonctions d'agregation (0-5) |
| `num_subqueries` | int | Nombre de sous-requetes (0-3) |

### B. Simulation Base de Donnees

| Feature | Type | Description |
|---------|------|-------------|
| `estimated_table_size_mb` | float | Taille estimee des tables (MB) |
| `rows_examined` | int | Nombre de lignes examinees |
| `has_index_used` | 0/1 | Index utilise pour la requete |
| `index_count` | int | Nombre d'index disponibles (0-8) |

### C. Contexte Serveur

| Feature | Type | Description |
|---------|------|-------------|
| `buffer_pool_hit_ratio` | float | Ratio de cache hit (0.3-0.99) |
| `connections_count` | int | Connexions actives (1-50) |
| `is_peak_hour` | 0/1 | Heure de pointe |

### D. Features Derivees

| Feature | Type | Description |
|---------|------|-------------|
| `joins_per_table` | float | Ratio joins/tables |
| `complexity_score` | float | Score de complexite composite |

### Target

| Feature | Type | Description |
|---------|------|-------------|
| `is_slow` | 0/1 | 0 = rapide, 1 = lent |

---

## Regles de Generation

La lenteur est calculee selon des regles DB realistes:

### Facteurs qui RALENTISSENT
- `rows_examined` eleve (impact fort)
- `estimated_table_size_mb` grande (impact fort)
- `num_joins` multiple (impact fort)
- `num_subqueries` (impact fort)
- `has_group_by` sans index
- `has_order_by` sur grande table
- `is_peak_hour` = 1
- Faible `buffer_pool_hit_ratio`

### Facteurs qui ACCELERENT
- `has_limit` = 1 (impact fort)
- `has_index_used` = 1
- Table petite avec index
- Requete simple (peu de joins)

### Contre-exemples forces (32% du dataset)

Pour eviter un modele trivial:

1. **Courtes mais lentes**: petite query + table enorme + pas d'index
2. **Longues mais rapides**: grande query + bon index + LIMIT
3. **Index present mais lent**: table enorme + beaucoup de joins + peak hour
4. **Pas d'index mais rapide**: petite table + LIMIT + bon cache

---

## Utilisation

### Generer le dataset

```bash
python generate_aml_dataset.py
```

### Valider avec ML

```bash
python validate_dataset.py
```

---

## Metriques Attendues

Un bon modele sur ce dataset devrait obtenir:

| Metrique | Plage Attendue |
|----------|----------------|
| Accuracy | 75% - 90% |
| ROC AUC | 0.80 - 0.95 |
| Top feature importance | < 35% |

### Feature Importance Typique

```
rows_examined             ~22%
estimated_table_size_mb   ~19%
has_limit                 ~10%
complexity_score          ~8%
num_joins                 ~5%
has_index_used            ~5%
```

**Si accuracy > 95%**: dataset trop facile, ajuster les poids.

---

## Distribution

```
Fast (is_slow=0): ~53%
Slow (is_slow=1): ~47%
```

---

## Configuration

Dans `generate_dataset.py`:

```python
N_SAMPLES = 20000      # Nombre de lignes
SLOW_RATIO = 0.45     # Proportion de slow
```


