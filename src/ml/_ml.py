import os
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import joblib
from src.config import ML_PHONE_SEGMENTS, feature_list


class PhonePricePredictor:
    def __init__(self, 
                 random_state: int=42, 
                 model_dir: str="src/ml/models"):
        self.random_state = random_state
        self.models = {}
        self.model_dir = model_dir
        self.is_models_loaded = False
        self.price_segments = ML_PHONE_SEGMENTS
        
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Model directory '{model_dir}' ensured to exist.")


    def load_and_preprocess(self, file_path):
        """Load and preprocess the dataset"""
        logger.info(f"Loading and preprocessing data from: {file_path}")
        df = pd.read_parquet(file_path)
        logger.debug(f"Dataset's columns: {df.columns.to_list()}")
        df.columns = df.columns.str.strip()
        keep_cols = list(set(df.columns.to_list()).intersection(set([i.lower() for i in feature_list])))
        df = df[keep_cols]
        logger.debug(f"After dropping extra columns: {df.columns}")

        if "NFC" in df.columns:
            df["NFC"] = df["NFC"].fillna(-1)

        if "CPU" in df.columns:
            df["CPU_manufacturer"] = df["CPU"].apply(
                lambda x: x.split()[0] if isinstance(x, str) else "Unknown"
            )
            logger.info("Extracted 'CPU_manufacturer' from 'CPU'.")

        # Convert RAM and ROM to GB if in MB
        if "RAM_MB" in df.columns:
            df["RAM_GB"] = df["RAM_MB"] / 1024
            logger.info("Converted RAM_MB to RAM_GB.")
        if "ROM_MB" in df.columns:
            df["ROM_GB"] = df["ROM_MB"] / 1024
            logger.info("Converted ROM_MB to ROM_GB.")


        # a little bit feature engineering
        # Brand tier categorization (if known premium brands exist in data)
        premium_brands = ["Apple", "Samsung", "Google", 'OnePlus']
        mid_tier_brands = ["Xiaomi", 'Huawei', 'Honor', 'Oppo']

        if "brand" in df.columns:
            df["brand_tier"] = df["brand"].apply(
                lambda x: (
                    "premium"
                    if x in premium_brands
                    else ("mid_tier" if x in mid_tier_brands else "budget")
                )
            )
            logger.info("Engineered 'brand_tier' feature.")

        # RAM to ROM ratio (premium phones often have different ratios)
        if "RAM_GB" in df.columns and "ROM_GB" in df.columns:
            # Ensure ROM_GB is not zero to avoid division by zero error
            df["RAM_ROM_ratio"] = df.apply(
                lambda row: row["RAM_GB"] / row["ROM_GB"] if row["ROM_GB"] > 0 else 0, axis=1
            )
            logger.info("Engineered 'RAM_ROM_ratio' feature.")


        # Add price category for segmentation
        df["price_category"] = df["Prices"].apply(self._get_price_category)
        logger.info("Added 'price_category' for segmentation.")

        # Define features - use only columns that exist in the dataset
        self.features = []

        # Core features - check each one exists before adding
        for feature in feature_list:
            if feature in df.columns:
                self.features.append(feature)

        # Engineered features - check each one exists before adding
        for feature in ["brand_tier", "RAM_ROM_ratio"]:
            if feature in df.columns:
                self.features.append(feature)

        # Store the final list of features for prediction
        self.final_features = self.features.copy()

        logger.info(f"Using features: {self.features}")

        self.target = "Prices"

        # Drop rows with missing target
        df = df.dropna(subset=[self.target])
        logger.info(f"Dropped rows with missing target '{self.target}'.")

        # Store default values for each feature for prediction
        self.feature_defaults = {}
        for feature in self.features:
            if feature in df.columns:
                if df[feature].dtype == "object":
                    self.feature_defaults[feature] = "Unknown"
                else:
                    self.feature_defaults[feature] = df[feature].median()
        logger.info(f"Stored feature defaults: {self.feature_defaults}")

        self.data = df
        logger.info("Data loading and preprocessing complete.")
        return df

    def _get_price_category(self, price):
        """Determine the price category for a given price"""
        if price < self.price_segments["budget"][1]:
            return "budget"
        elif price < self.price_segments["mid_range"][1]:
            return "mid_range"
        else:
            return "premium"

    def segment_and_train(self):
        """Segment the data and train models for each segment"""
        logger.info("Starting model training for each price segment.")
        # Segment the data
        for category in self.price_segments.keys():
            segment_df = self.data[self.data["price_category"] == category]

            if len(segment_df) > 10:  # Only train if we have enough data
                X = segment_df[self.features]
                y = segment_df[self.target]

                # For premium segment, consider log-transforming the target
                if category == "premium":
                    log_transform = True
                    if log_transform:
                        y = np.log1p(y)  # log(1+y) to handle potential zeros
                        logger.info(f"Applied log-transform to target for '{category}' segment.")
                else:
                    log_transform = False

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_state
                )

                # preprocessor
                categorical_features = X.select_dtypes(
                    include=["object"]
                ).columns.tolist()
                numerical_features = X.select_dtypes(
                    include=["int64", "float64"]
                ).columns.tolist()

                # scaling for numerical features
                preprocessor = ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="median")),
                                    (
                                        "scaler",
                                        StandardScaler(),
                                    ),
                                ]
                            ),
                            numerical_features,
                        ),
                        (
                            "cat",
                            Pipeline(
                                [
                                    (
                                        "imputer",
                                        SimpleImputer(
                                            strategy="constant", fill_value="Unknown"
                                        ),
                                    ),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            categorical_features,
                        ),
                    ]
                )

                # Create pipeline with XGBoost
                pipeline = Pipeline(
                    [
                        ("preprocessor", preprocessor),
                        ("regressor", xgb.XGBRegressor(random_state=self.random_state)),
                    ]
                )

                # parameter grid based on segment
                if category == "premium":
                    # More extensive tuning for premium segment
                    param_grid = {
                        "regressor__n_estimators": [100, 200, 300],
                        "regressor__max_depth": [4, 6, 8],
                        "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "regressor__min_child_weight": [1, 3, 5],
                        "regressor__gamma": [0, 0.1, 0.2],
                        "regressor__subsample": [0.8, 0.9, 1.0],
                    }
                elif category == "mid_range":
                    # The mid-range model performed worst, so give it more tuning options
                    param_grid = {
                        "regressor__n_estimators": [50, 100, 200],
                        "regressor__max_depth": [3, 6, 9],
                        "regressor__learning_rate": [0.05, 0.1, 0.15],
                        "regressor__min_child_weight": [1, 3],
                        "regressor__subsample": [0.8, 1.0],
                    }
                else:
                    # Budget model already performs well
                    param_grid = {
                        "regressor__n_estimators": [50, 100],
                        "regressor__max_depth": [4, 6],
                        "regressor__learning_rate": [0.05, 0.1],
                    }

                # Use GridSearchCV for hyperparameter tuning
                cv_folds = min(5, len(X_train) // 100 + 2) if len(X_train) >= 100 else min(5, max(2, len(X_train) // 10)) # ensure at least 2 folds for small data
                if len(X_train) < 2 * cv_folds: 
                    cv_folds = max(2, len(X_train) // 2) if len(X_train) >= 4 else None # If very few samples, CV might not be feasible or might be set to None for GridSearchCV to handle

                grid_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=cv_folds, 
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                )

                # Train the model
                logger.info(f"Training {category} model with {len(X_train)} samples (CV folds: {cv_folds})...")
                grid_search.fit(X_train, y_train)

                # Evaluate on test set
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)

                # Inverse transform for premium if log-transformed
                if log_transform:
                    y_test_original = np.expm1(y_test)  # expm1 is inverse of log1p
                    y_pred_original = np.expm1(y_pred)
                    mae = mean_absolute_error(y_test_original, y_pred_original)
                    r2 = r2_score(y_test_original, y_pred_original)
                else:
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                # Calculate and log metrics
                logger.info(f"{category.title()} Model Evaluation:")
                logger.info(f"  - Number of phones in segment: {len(segment_df)}")
                logger.info(f"  - Best parameters: {grid_search.best_params_}")
                logger.info(f"  - Mean Absolute Error: ${mae:.2f}")
                logger.info(f"  - R² Score: {r2:.4f}")

                # Store the model and whether it uses log transform
                self.models[category] = {
                    "model": best_model,
                    "log_transform": log_transform,
                    "features": self.features,  # Store features used for this model
                }
            else:
                logger.warning(
                    f"Not enough data for {category} segment ({len(segment_df)} phones). Skipping model training."
                )

        return self.models

    def save_models(self, prefix="phone_price_model"):
        """Save trained models to disk"""
        logger.info(f"Saving models with prefix '{prefix}' to directory '{self.model_dir}'.")
        model_info = {
            "features": self.features,
            "feature_defaults": self.feature_defaults,
            "price_segments": self.price_segments
        }
        model_info_path = os.path.join(self.model_dir, "model_info.pkl")
        joblib.dump(model_info, model_info_path)
        logger.info(f"Saved model_info.pkl to {model_info_path}")
        
        for segment_name, model_data in self.models.items():
            model_path = os.path.join(self.model_dir, f"{prefix}_{segment_name}.pkl")
            joblib.dump(model_data, model_path)
            logger.info(f"Saved {segment_name} model to {model_path}")

    def load_models(self, prefix="phone_price_model"):
        """Load trained models from disk"""
        logger.info(f"Loading models with prefix '{prefix}' from directory '{self.model_dir}'.")
        try:
            # Load model info
            model_info_path = os.path.join(self.model_dir, "model_info.pkl")
            if not os.path.exists(model_info_path):
                logger.error(f"Model info file not found at {model_info_path}. Cannot load models.")
                return False
                
            model_info_data = joblib.load(model_info_path)
            
            # Set model attributes
            self.features = model_info_data["features"]
            self.final_features = self.features.copy() # Ensure final_features is also set
            self.feature_defaults = model_info_data["feature_defaults"]
            self.price_segments = model_info_data["price_segments"]
            logger.info(f"Loaded model_info.pkl from {model_info_path}.")
            
            # Load models for each segment
            self.models = {}
            for segment_name in self.price_segments.keys():
                model_path = os.path.join(self.model_dir, f"{prefix}_{segment_name}.pkl")
                if os.path.exists(model_path):
                    self.models[segment_name] = joblib.load(model_path)
                    logger.info(f"Loaded {segment_name} model from {model_path}")
                else:
                    logger.warning(f"No model found for {segment_name} segment at {model_path}")
            
            if self.models:
                self.is_models_loaded = True
                logger.info("Models loaded successfully.")
                return True
            else:
                logger.warning("No models were loaded. Train models or check model paths.")
                return False
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
        

    def predict_price(self, phone_specs: dict):
        """Predict the price for a given phone specs"""
        # Check if models are loaded
        if not self.is_models_loaded and not self.models:
            logger.warning("No models loaded. Please load models first with load_models().")
            return None
            
        # Make a copy of the input specs to avoid modifying the original
        specs = phone_specs.copy()
        
        # Check for missing features and add defaults
        for feature in self.final_features: # Use self.final_features which should be set during load/train
            if feature not in specs:
                if feature in self.feature_defaults:
                    # Use stored default value
                    specs[feature] = self.feature_defaults[feature]
                    logger.info(f"Added default value for missing feature '{feature}': {specs[feature]}")
                else:
                    # Add a sensible default if we don't have a stored one
                    if feature == 'OS':
                        specs[feature] = 'Android'  # Most common OS
                    elif feature == 'brand':
                        specs[feature] = 'Unknown'
                    elif feature == 'CPU_manufacturer':
                        specs[feature] = 'Unknown'
                    elif feature == 'brand_tier':
                        specs[feature] = 'mid_tier'  # Default to mid-tier
                    else:
                        # For numerical features, use 0 as default
                        specs[feature] = 0
                    logger.info(f"Added fallback default for missing feature '{feature}': {specs[feature]}")
        
        # Convert specs to DataFrame, ensuring correct feature order
        try:
            specs_df = pd.DataFrame([specs])[self.final_features]
        except KeyError as e:
            logger.error(f"Missing expected feature in input specs for DataFrame creation: {e}. Specs: {specs}, Expected: {self.final_features}")
            return None 

        # Get predictions from all models
        predictions = {}
        for segment_name, model_data in self.models.items():
            model = model_data["model"]
            log_transform = model_data["log_transform"]
            
            # Try to predict
            try:
                pred = model.predict(specs_df)[0]
                
                # Inverse transform if needed
                if log_transform:
                    pred = np.expm1(pred)
                
                predictions[segment_name] = pred
            except Exception as e:
                logger.error(f"Error predicting with {segment_name} model: {e}")
                continue
        
        if not predictions:
            logger.warning("No predictions could be made by any segment model.")
            return None
        
        # Intelligently select the right model based on phone specs
        # Ensure necessary keys exist in specs before accessing them
        brand_spec = specs.get("brand", "Unknown")
        ram_gb_spec = specs.get("RAM_GB", 0)
        rom_gb_spec = specs.get("ROM_GB", 0)

        selected_segment = None
        if "premium" in predictions and brand_spec in ["Apple", "Samsung"] and ram_gb_spec >= 8:
            prediction = predictions["premium"]
            selected_segment = "premium"
        elif "budget" in predictions and ram_gb_spec <= 4 and rom_gb_spec <= 64:
            prediction = predictions["budget"]
            selected_segment = "budget"
        elif "mid_range" in predictions: # Fallback to mid_range if available
            prediction = predictions["mid_range"]
            selected_segment = "mid_range"
        elif predictions: # If no specific segment matches, pick the first available
            selected_segment = list(predictions.keys())[0]
            prediction = predictions[selected_segment]
            logger.info(f"Defaulted to first available segment '{selected_segment}' for prediction.")
        else:
            logger.error("No suitable model segment found for prediction logic and no predictions available.")
            return None

        logger.info(f"Selected {selected_segment} model for prediction. Predicted price: AZN: {prediction:.2f}")
        return prediction

    def train_and_test(self, file_path):
        """Train models and evaluate on test data"""
        logger.info(f"Starting train_and_test process with file: {file_path}")
        # Load and preprocess data
        self.load_and_preprocess(file_path)
        
        # Train models on each segment
        self.segment_and_train()
        
        # Save models
        self.save_models()
        
        # Set models as loaded
        self.is_models_loaded = True
        logger.info("train_and_test process completed. Models are trained, saved, and loaded.")
        
        return self.models


def train_models(data_path: str | Path):
    """ """
    predictor = PhonePricePredictor()
    predictor.train_and_test(data_path)


def predict_price(phone_specs: dict):
    """ """
    predictor = PhonePricePredictor()
    if predictor.load_models():
        return predictor.predict_price(phone_specs)
    else:
        return None
