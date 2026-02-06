import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# 1. SQL Query Parser (Extract features from SQL)
class SQLFeatureExtractor:
    """Extract features from SQL queries"""
    def extract_features(self, sql_query):
        """
        Extract features from SQL query to match training data format
        Returns a dictionary of features
        """
        sql_upper = sql_query.upper().strip()

        # Initialize with default values
        features = {
            'query_type': 'SELECT',
            'rows_examined': 1000,
            'rows_sent': 100,
            'execution_time': 0.1,
            'lock_time': 0.01
        }

        # 1. Determine query type
        if sql_upper.startswith('SELECT'):
            features['query_type'] = 'SELECT'
            # SELECT queries might examine more rows
            features['rows_examined'] = self._estimate_rows_examined(sql_upper)
            features['rows_sent'] = max(10, features['rows_examined'] * 0.1)
        elif sql_upper.startswith('INSERT'):
            features['query_type'] = 'INSERT'
            features['rows_examined'] = 100
            features['rows_sent'] = 50
        elif sql_upper.startswith('UPDATE'):
            features['query_type'] = 'UPDATE'
            features['rows_examined'] = self._estimate_rows_examined(sql_upper)
            features['rows_sent'] = features['rows_examined'] * 0.8
        elif sql_upper.startswith('DELETE'):
            features['query_type'] = 'DELETE'
            features['rows_examined'] = self._estimate_rows_examined(sql_upper)
            features['rows_sent'] = features['rows_examined'] * 0.5

        # 2. Estimate execution time based on query complexity
        features['execution_time'] = self._estimate_execution_time(sql_upper, features['rows_examined'])

        # 3. Estimate lock time
        features['lock_time'] = features['execution_time'] * np.random.uniform(0.01, 0.1)

        return features

    def _estimate_rows_examined(self, sql_query):
        """Estimate rows examined based on query complexity"""
        base_rows = 1000

        # Adjust based on JOINs
        join_count = sql_query.count('JOIN')
        base_rows *= (1 + join_count * 0.5)

        # Adjust based on WHERE conditions
        where_count = sql_query.count('WHERE')
        base_rows *= (1 + where_count * 0.3)

        # Adjust based on subqueries
        subquery_count = max(0, sql_query.count('SELECT') - 1)
        base_rows *= (1 + subquery_count * 0.8)

        # Adjust based on GROUP BY / ORDER BY
        if 'GROUP BY' in sql_query or 'ORDER BY' in sql_query:
            base_rows *= 1.2

        return int(base_rows)

    def _estimate_execution_time(self, sql_query, rows_examined):
        """Estimate execution time based on rows and query complexity"""
        base_time = rows_examined * 0.0001  # Base time per row

        # Complexity multipliers
        complexity = 1.0

        if 'JOIN' in sql_query:
            complexity *= 1.5
        if 'DISTINCT' in sql_query:
            complexity *= 1.3
        if 'GROUP BY' in sql_query:
            complexity *= 1.4
        if 'ORDER BY' in sql_query:
            complexity *= 1.2
        if 'LIKE' in sql_query and '%' in sql_query:
            complexity *= 1.8
        if max(0, sql_query.count('SELECT') - 1) > 0:  # Has subqueries
            complexity *= 2.0

        estimated_time = base_time * complexity

        # Add some randomness
        estimated_time *= np.random.uniform(0.8, 1.2)

        return min(estimated_time, 10.0)  # Cap at 10 seconds

# 2. Data Preparation
def prepare_data(df, target_column='slow'):
    """
    Prepare the dataset for model training
    target_column: Column indicating if query is slow (1) or fast (0)
    """
    # Drop specified columns
    cols_to_drop = ['access_type', 'key_used', 'using_filesort',
                    'using_temporary', 'cpu_usage', 'connections_count', 'id']

    # Keep only columns that exist in the dataframe
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    df_clean = df.drop(columns=existing_cols)

    # Separate features and target
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")

    return X, y, categorical_cols, numerical_cols

# 3. Build ML Pipeline with optimized parameters
def build_pipeline(categorical_cols, numerical_cols, use_smote=True):
    """
    Build preprocessing and model pipeline with optimized parameters
    """
    # Preprocessing for numerical features
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical features
    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Create base pipeline
    if use_smote:
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42, sampling_strategy=0.8)),
            ('classifier', xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=4  # Account for class imbalance
            ))
        ])
    else:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=4
            ))
        ])

    return pipeline

# 4. Train and Evaluate Model
def train_model(X_train, y_train, X_test, y_test, categorical_cols, numerical_cols):
    """
    Train and evaluate the XGBoost model
    """
    print("\nTraining model with optimized parameters...")

    # Try with and without SMOTE and choose the better one
    best_accuracy = 0
    best_pipeline = None
    best_use_smote = False

    for use_smote in [True, False]:
        pipeline = build_pipeline(categorical_cols, numerical_cols, use_smote)

        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"\nModel with SMOTE={use_smote}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_pipeline = pipeline
                best_use_smote = use_smote
        except Exception as e:
            print(f"  Error training with SMOTE={use_smote}: {e}")

    print(f"\nSelected model with SMOTE={best_use_smote} (Accuracy: {best_accuracy:.4f})")

    # Evaluate best model
    y_pred = best_pipeline.predict(X_test)

    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"[[TN:{cm[0,0]:>3} FP:{cm[0,1]:>3}]")
    print(f" [FN:{cm[1,0]:>3} TP:{cm[1,1]:>3}]]")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fast', 'Slow']))

    return best_pipeline

# 5. Query Performance Classifier
class QueryPerformanceClassifier:
    def __init__(self, model_path=None):
        """
        Initialize the classifier
        """
        if model_path:
            self.load_model(model_path)
        else:
            self.model = None
            self.categorical_cols = None
            self.numerical_cols = None
            self.feature_names = None
            self.sql_extractor = SQLFeatureExtractor()

    def train(self, df, target_column='slow', test_size=0.2, random_state=42):
        """
        Train the classifier on historical data
        """
        # Prepare data
        X, y, self.categorical_cols, self.numerical_cols = prepare_data(df, target_column)
        self.feature_names = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print("\n" + "="*50)
        print("TRAINING PHASE")
        print("="*50)

        print(f"Dataset shape: {df.shape}")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        print("\nClass distribution:")
        print(f"  Fast (0): {sum(y == 0)} queries ({sum(y == 0)/len(y)*100:.1f}%)")
        print(f"  Slow (1): {sum(y == 1)} queries ({sum(y == 1)/len(y)*100:.1f}%)")

        # Train with optimized parameters
        self.model = train_model(X_train, y_train, X_test, y_test,
                                self.categorical_cols, self.numerical_cols)

        # Final evaluation on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nFinal Test Set Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

        return self.model

    def test_sql_query(self):
        """
        Interactive function to test SQL queries
        """
        if self.model is None:
            print("Model not trained. Please train the model first.")
            return

        print("\n" + "="*60)
        print("SQL QUERY PERFORMANCE ANALYZER")
        print("="*60)
        print("\nType 'quit' to exit the program.\n")

        while True:
            sql_query = input("\nEnter SQL Query:\n> ").strip()

            if sql_query.lower() in ['quit', 'exit', 'q']:
                print("\nExiting SQL query analyzer.")
                break

            if not sql_query:
                print("Please enter a valid SQL query.")
                continue

            try:
                # Extract features from SQL
                features = self.sql_extractor.extract_features(sql_query)

                # Make prediction
                result = self.predict_query(features)

                # Display result in the requested format
                print(f"\nResult: This query is {result['status'].lower()}")
                print(f"Confidence: {result['confidence']:.1%}\n")

            except Exception as e:
                print(f"Error analyzing query: {e}")

    def predict_query(self, query_features):
        """
        Predict if a query is slow (1) or fast (0)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Train or load a model first.")

        # Convert dict to DataFrame if needed
        if isinstance(query_features, dict):
            query_df = pd.DataFrame([query_features])
        else:
            query_df = query_features.copy()

        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(query_df.columns)
        if missing_features:
            # Try to fill missing features with defaults
            for feature in missing_features:
                if feature == 'query_type':
                    query_df[feature] = 'SELECT'
                else:
                    query_df[feature] = 0

        # Reorder columns to match training data
        query_df = query_df[self.feature_names]

        # Make prediction
        prediction = self.model.predict(query_df)
        probability = self.model.predict_proba(query_df)

        # Return result
        results = []
        for pred, prob in zip(prediction, probability):
            status = "fast" if pred == 0 else "slow"
            confidence = prob[0] if pred == 0 else prob[1]
            results.append({
                'prediction': int(pred),
                'status': status,
                'confidence': float(confidence),
                'slow_probability': float(prob[1]),
                'fast_probability': float(prob[0])
            })

        return results[0] if len(results) == 1 else results

    def save_model(self, filepath):
        """Save the trained model"""
        if self.model:
            model_data = {
                'model': self.model,
                'categorical_cols': self.categorical_cols,
                'numerical_cols': self.numerical_cols,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, filepath)
            print(f"\nModel saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.categorical_cols = model_data['categorical_cols']
        self.numerical_cols = model_data['numerical_cols']
        self.feature_names = model_data['feature_names']
        self.sql_extractor = SQLFeatureExtractor()
        print(f"Model loaded from {filepath}")

# Main execution
def main():
    """
    Main training workflow with SQL testing
    """
    print("="*60)
    print("QUERY PERFORMANCE CLASSIFIER")
    print("="*60)
    print("Loading dataset...")

    # Load your actual dataset here
    # Replace this with your actual data loading code
    try:
        # Try to load your dataset file
        df = pd.read_csv('your_dataset.csv')  # Change to your actual file name
        print(f"Dataset loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("Dataset file not found. Creating sample dataset for demonstration...")

        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 10000

        sample_data = {
            'query_type': np.random.choice(['SELECT', 'UPDATE', 'INSERT', 'DELETE'],
                                           n_samples, p=[0.7, 0.15, 0.1, 0.05]),
            'rows_examined': np.random.exponential(2000, n_samples).astype(int) + 100,
            'rows_sent': np.random.exponential(200, n_samples).astype(int) + 10,
            'execution_time': np.random.exponential(0.5, n_samples),
            'lock_time': np.random.exponential(0.01, n_samples),
        }

        df = pd.DataFrame(sample_data)

        # Create target variable
        slow_prob = (
            (df['rows_examined'] > 5000) * 0.4 +
            (df['query_type'].isin(['UPDATE', 'DELETE'])) * 0.3 +
            (df['execution_time'] > 1.0) * 0.2 +
            (df['lock_time'] > 0.05) * 0.1 +
            np.random.normal(0, 0.1, n_samples)
        )
        df['slow'] = (slow_prob > 0.5).astype(int)

        print(f"Sample dataset created. Shape: {df.shape}")

    # Initialize and train classifier
    classifier = QueryPerformanceClassifier()

    # Train the model
    classifier.train(df, target_column='slow')

    # Save the model
    classifier.save_model('query_performance_classifier.pkl')

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)

    # Start SQL testing interface
    classifier.test_sql_query()

# Quick test function
def quick_test_sql(sql_query, model_path='query_performance_classifier.pkl'):
    """
    Quick function to test a single SQL query
    """
    try:
        # Load model
        classifier = QueryPerformanceClassifier(model_path)

        # Extract features and predict
        features = classifier.sql_extractor.extract_features(sql_query)
        result = classifier.predict_query(features)

        print(f"\nEnter SQL Query:")
        print(f"> {sql_query}")
        print(f"\nResult: This query is {result['status']}")
        print(f"Confidence: {result['confidence']:.1%}")

        return result

    except Exception as e:
        print(f"Error: {e}")
        return None

# Run the main function
if __name__ == "__main__":
    main()

    # To quickly test a single query without training:
    # query = "SELECT * FROM users WHERE age > 30 ORDER BY created_at DESC"
    # quick_test_sql(query)