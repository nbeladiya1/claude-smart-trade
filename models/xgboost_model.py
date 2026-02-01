"""
XGBoost Model for Trading Signal Prediction
Ported from Smart-treading with enhancements
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import ML libraries
try:
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML libraries not available. XGBoost predictions disabled.")


class XGBoostPredictor:
    """XGBoost classifier for BUY/SELL/HOLD signal generation"""

    FEATURE_COLUMNS = [
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'ema50', 'ema200', 'adx', 'atr',
        'price_to_ema50', 'price_to_ema200', 'ema_crossover',
        'returns', 'log_returns', 'high_low_ratio'
    ]

    SIGNAL_MAP = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    REVERSE_SIGNAL_MAP = {'SELL': 0, 'HOLD': 1, 'BUY': 2}

    def __init__(self, symbol: str, models_dir: str = None):
        self.symbol = symbol.replace('/', '_').replace('-', '_')
        self.model = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.model_version = None
        
        # Models directory
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), 'saved')
        self.models_dir = models_dir
        
        self.model_path = os.path.join(self.models_dir, f'xgboost_{self.symbol}.joblib')
        self.scaler_path = os.path.join(self.models_dir, f'scaler_{self.symbol}.joblib')
        
        # Create models directory if not exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load existing model if available
        self.load_model()

    def load_model(self) -> bool:
        """Load saved model and scaler"""
        if not ML_AVAILABLE:
            return False
            
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Loaded XGBoost model for {self.symbol}")
                return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
        return False

    def save_model(self) -> bool:
        """Save model and scaler"""
        if not ML_AVAILABLE or not self.model:
            return False
            
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"Saved XGBoost model for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def _create_labels(self, df: pd.DataFrame, lookahead: int = 5, threshold: float = 0.001) -> pd.Series:
        """Create target labels based on future price movement"""
        future_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        labels = pd.Series(1, index=df.index)  # Default to HOLD
        labels[future_returns > threshold] = 2   # BUY
        labels[future_returns < -threshold] = 0  # SELL
        
        return labels

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model input"""
        available_features = [col for col in self.FEATURE_COLUMNS if col in df.columns]
        
        if len(available_features) < 5:
            logger.warning(f"Insufficient features: {available_features}")
            return pd.DataFrame()
        
        features = df[available_features].copy()
        features = features.ffill().bfill()
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        return features

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        lookahead: int = 5,
        threshold: float = 0.001
    ) -> Dict:
        """Train the XGBoost model"""
        if not ML_AVAILABLE:
            return {'error': 'ML libraries not available'}
            
        logger.info(f"Training XGBoost model for {self.symbol}...")
        
        # Prepare features and labels
        features = self.prepare_features(df)
        if features.empty:
            return {'error': 'Insufficient features'}
            
        labels = self._create_labels(df, lookahead, threshold)
        
        # Remove invalid rows
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_idx].iloc[:-lookahead]
        labels = labels[valid_idx].iloc[:-lookahead]
        
        if len(features) < 100:
            return {'error': f'Insufficient data: {len(features)} samples'}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softmax',
            num_class=3,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        self.model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = (y_pred == y_test).mean()
        
        # Save model
        self.save_model()
        self.model_version = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        metrics = {
            'accuracy': float(accuracy),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'training_date': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Training complete. Accuracy: {accuracy:.4f}")
        return metrics

    def predict(self, df: pd.DataFrame) -> Dict:
        """Generate trading signal prediction"""
        if not ML_AVAILABLE or not self.model:
            return {
                'signal': 'HOLD',
                'confidence': 50.0,
                'available': False
            }
        
        try:
            features = self.prepare_features(df)
            if features.empty:
                return {'signal': 'HOLD', 'confidence': 50.0, 'error': 'No features'}
            
            # Get latest row
            latest = features.iloc[[-1]]
            latest_scaled = self.scaler.transform(latest)
            
            # Predict
            prediction = self.model.predict(latest_scaled)[0]
            probabilities = self.model.predict_proba(latest_scaled)[0]
            
            signal = self.SIGNAL_MAP.get(prediction, 'HOLD')
            confidence = float(max(probabilities) * 100)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'probabilities': {
                    'SELL': float(probabilities[0]) * 100,
                    'HOLD': float(probabilities[1]) * 100,
                    'BUY': float(probabilities[2]) * 100
                },
                'available': True
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'signal': 'HOLD', 'confidence': 50.0, 'error': str(e)}


# Cache for model instances
_model_cache: Dict[str, XGBoostPredictor] = {}


def get_predictor(symbol: str) -> XGBoostPredictor:
    """Get or create XGBoost predictor for symbol"""
    if symbol not in _model_cache:
        _model_cache[symbol] = XGBoostPredictor(symbol)
    return _model_cache[symbol]
