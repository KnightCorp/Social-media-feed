"""
Feed Recommendation System - Main Pipeline
A deployment-ready pipeline for social media feed ranking and recommendation.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))

import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# --- Custom module imports (adjust to your structure) ---
from src.data_pipeline.data_loader import load_csv_data, show_basic_info
from src.data_pipeline.cleaner import clean_data
from src.data_pipeline.feature_engineer import engineer_features
from src.data_pipeline.eda import run_eda
from src.ranking.algorithm_ranker import Post, User, df_to_posts, run_all_algorithms
from src.modeling.xgboost_ranker import train_xgboost_classifier
from src.modeling.dl_model import train_keras_mlp
from models.model_comparision import compare_multiple_models
from src.ranking.model_integrator import train_and_rank_pipeline, calculate_social_metrics

class FeedRecommendationPipeline:
    """Main pipeline class for feed recommendation system."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        self.data_paths = config.get('data_paths', {})
        self.model_params = config.get('model_params', {})
        self._create_directories()

    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=getattr(logging, self.config.get('log_level', 'INFO')),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.get('log_file', 'pipeline.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger("FeedRecommendationPipeline")

    def _create_directories(self):
        for directory in [
            'data/processed', 'data/eda_outputs', 'models/saved', 'results'
        ]:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def run_data_pipeline(self) -> pd.DataFrame:
        self.logger.info("Starting data pipeline...")
        df = load_csv_data(self.data_paths.get('raw_data', 'data/raw/feed.csv'))
        show_basic_info(df)
        df_clean = clean_data(df)
        cleaned_path = self.data_paths.get('cleaned_data', 'data/processed/feed_cleaned.csv')
        df_clean.to_csv(cleaned_path, index=False)
        self.logger.info(f"Cleaned data saved to {cleaned_path}")
        df_features = engineer_features(df_clean)
        features_path = self.data_paths.get('features_data', 'data/processed/feed_features.csv')
        df_features.to_csv(features_path, index=False)
        self.logger.info(f"Feature-engineered data saved to {features_path}")
        if self.config.get('run_eda', False):
            run_eda(df_features, save_path=self.data_paths.get('eda_outputs', 'data/eda_outputs'))
        return df_features

    def run_algorithm_ranking(self, df: pd.DataFrame):
        self.logger.info("Running algorithmic ranking...")
        posts = df_to_posts(df)
        user = User(user_id='u100', connections=['u101', 'u102'], interaction_history={'u101': 5, 'u102': 2},
                    topic_interests={'music': 0.8, 'art': 0.5}, format_preferences={'text': 0.7, 'video': 0.4},
                    creator_preferences={'u101': 0.9}, location=(37.7749, -122.4194), expertise_levels={'music': 0.6})
        results = run_all_algorithms(posts, user)
        for algo, ranked in results.items():
            top_post_ids = [p.post_id for p in ranked[:10]]
            self.logger.info(f"{algo} Top 10 Posts: {top_post_ids}")
        return results

    def prepare_ml_data(self, df: pd.DataFrame):
        feature_cols = self.config.get('feature_columns', [
            'likes', 'comments', 'shares', 'avg_time_spent', 'content_length',
            'sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_compound',
            'hours_since_post', 'num_topics', 'num_creator_connections'
        ])
        X = df[feature_cols]
        threshold = df['engagement_score'].median() if self.config.get('engagement_threshold', 'median') == 'median' \
            else float(self.config['engagement_threshold'])
        y = (df['engagement_score'] > threshold).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.get('test_size', 0.2), random_state=self.config.get('random_state', 42)
        )
        return X_train, X_test, y_train, y_test, feature_cols

    def train_models(self, X_train, X_test, y_train, y_test, feature_cols):
        self.logger.info("Training machine learning models...")
        models = {}
        if self.config.get('train_xgboost', True):
            models['xgboost'] = train_xgboost_classifier(X_train, y_train, X_test, y_test, feature_names=feature_cols)
        if self.config.get('train_neural_net', True):
            models['neural_net'] = train_keras_mlp(X_train, y_train, X_test, y_test, feature_names=feature_cols)
        # Add other models as needed
        self._save_models(models)
        return models

    def _save_models(self, models: Dict[str, Any]):
        model_save_path = Path('models/saved')
        for model_name, model in models.items():
            if model is not None:
                model_file = model_save_path / f"{model_name}.joblib"
                try:
                    joblib.dump(model, model_file)
                    self.logger.info(f"Saved {model_name} model to {model_file}")
                except Exception as e:
                    self.logger.error(f"Failed to save {model_name} model: {e}")

    def run_model_comparison(self, df: pd.DataFrame):
        self.logger.info("Running model comparison...")
        score_columns = ['rf_score', 'xgb_score', 'mlp_score']
        available_scores = [col for col in score_columns if col in df.columns]
        if len(available_scores) < 2:
            self.logger.warning("Insufficient model scores for comparison")
            return
        model_tops = {}
        for score_col in available_scores:
            model_name = score_col.replace('_score', '').upper()
            top_posts = set(df.sort_values(score_col, ascending=False).head(10)['post_id'])
            model_tops[model_name] = top_posts
        compare_multiple_models(model_tops)

    def run_integrated_ranking(self, df: pd.DataFrame):
        self.logger.info("Running integrated ranking pipeline...")
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_ml_data(df)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        # Build and train a simple neural network
        from tensorflow import keras
        model = keras.Sequential([
            keras.layers.Input(shape=(X_train.shape[1],)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
        ranked_df = train_and_rank_pipeline(df, feature_cols, model, scaler, top_n=self.config.get('top_n', 10), score_col='dl_score')
        calculate_social_metrics(df, top_n=self.config.get('top_n', 10), score_col='dl_score')
        return ranked_df

    def run_full_pipeline(self):
        self.logger.info("Starting full pipeline execution...")
        results = {}
        try:
            if self.config.get('run_data_pipeline', True):
                df_features = self.run_data_pipeline()
                results['processed_data'] = df_features
            else:
                features_path = self.data_paths.get('features_data', 'data/processed/feed_features.csv')
                df_features = pd.read_csv(features_path)
                results['processed_data'] = df_features
            if self.config.get('run_algorithmic_ranking', True):
                algo_results = self.run_algorithm_ranking(df_features)
                results['algorithmic_ranking'] = algo_results
            if self.config.get('run_ml_training', True):
                X_train, X_test, y_train, y_test, feature_cols = self.prepare_ml_data(df_features)
                ml_models = self.train_models(X_train, X_test, y_train, y_test, feature_cols)
                results['ml_models'] = ml_models
            if self.config.get('run_model_comparison', True):
                self.run_model_comparison(df_features)
            if self.config.get('run_integrated_ranking', True):
                ranked_df = self.run_integrated_ranking(df_features)
                results['ranked_data'] = ranked_df
            self.logger.info("Pipeline execution completed successfully!")
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise
        return results

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path and Path(config_path).exists():
        import json
        with open(config_path, 'r') as f:
            return json.load(f)
    # Default configuration
    return {
        'log_level': 'INFO',
        'log_file': 'pipeline.log',
        'data_paths': {
            'raw_data': 'data/raw/feed.csv',
            'cleaned_data': 'data/processed/feed_cleaned.csv',
            'features_data': 'data/processed/feed_features.csv',
            'eda_outputs': 'data/eda_outputs'
        },
        'feature_columns': [
            'likes', 'comments', 'shares', 'avg_time_spent', 'content_length',
            'sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_compound',
            'hours_since_post', 'num_topics', 'num_creator_connections'
        ],
        'engagement_threshold': 'median',
        'test_size': 0.2,
        'random_state': 42,
        'top_n': 10,
        'run_data_pipeline': True,
        'run_algorithmic_ranking': True,
        'run_ml_training': True,
        'run_model_comparison': True,
        'run_integrated_ranking': True,
        'train_xgboost': True,
        'train_neural_net': True
    }

def main():
    parser = argparse.ArgumentParser(description='Feed Recommendation Pipeline')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--mode', type=str, choices=['full', 'data', 'ml', 'ranking'], default='full')
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()
    config = load_config(args.config)
    # Adjust configuration based on mode
    if args.mode == 'data':
        config.update({'run_algorithmic_ranking': False, 'run_ml_training': False, 'run_model_comparison': False, 'run_integrated_ranking': False})
    elif args.mode == 'ml':
        config.update({'run_data_pipeline': False, 'run_algorithmic_ranking': False})
    elif args.mode == 'ranking':
        config.update({'run_data_pipeline': False, 'run_ml_training': False, 'run_model_comparison': False})
    pipeline = FeedRecommendationPipeline(config)
    results = pipeline.run_full_pipeline()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    import json
    with open(output_dir / 'pipeline_results.json', 'w') as f:
        serializable_results = {k: (f"DataFrame with shape {v.shape}" if isinstance(v, pd.DataFrame) else str(type(v)) if hasattr(v, '__dict__') else str(v)) for k, v in results.items()}
        json.dump(serializable_results, f, indent=2)
    print(f"Pipeline completed successfully! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
