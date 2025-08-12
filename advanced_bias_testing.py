#!/usr/bin/env python3
"""
Advanced Bias Testing for Fraud Detection Models

This script performs comprehensive bias analysis on fraud detection models,
including basic metrics, advanced fairness metrics, intersectional analysis,
counterfactual fairness testing, and SHAP explainability.
"""

import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Fairness and bias testing
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    selection_rate
)

# SHAP for explainability
import shap

# Statistical analysis
from scipy import stats


class BiasTestingFramework:
    """Framework for comprehensive bias testing of fraud detection models."""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.sensitive_attributes = ['Customer_Type', 'Transaction_Type']
        self.results = {}
        
    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and testing datasets.
        
        Args:
            train_path: Path to training CSV file
            test_path: Path to testing CSV file
            
        Returns:
            Tuple of (train_df, test_df)
        """
        print("Loading datasets...")
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            print(f"Training data shape: {train_df.shape}")
            print(f"Testing data shape: {test_df.shape}")
            print(f"Training columns: {train_df.columns.tolist()}")
            
            # Verify required columns exist
            required_cols = ['Transaction_Date', 'Transaction_Time', 'Customer_ID', 
                           'Customer_Type', 'Transaction_Type', 'Transaction_Amount', 'Is_Fraudulent']
            
            for col in required_cols:
                if col not in train_df.columns:
                    raise ValueError(f"Required column '{col}' not found in training data")
                if col not in test_df.columns:
                    raise ValueError(f"Required column '{col}' not found in testing data")
            
            return train_df, test_df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Preprocess data for model training.
        
        Args:
            train_df: Training dataframe
            test_df: Testing dataframe
            
        Returns:
            X_train, X_test, y_train, y_test, train_processed, test_processed
        """
        print("Preprocessing data...")
        
        # Create copies for processing
        train_processed = train_df.copy()
        test_processed = test_df.copy()
        
        # Convert datetime features
        for df in [train_processed, test_processed]:
            df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
            df['Transaction_Hour'] = pd.to_datetime(df['Transaction_Time']).dt.hour
            df['Transaction_DayOfWeek'] = df['Transaction_Date'].dt.dayofweek
            df['Transaction_Month'] = df['Transaction_Date'].dt.month
        
        # Encode categorical variables
        categorical_cols = ['Customer_Type', 'Transaction_Type']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                # Fit on combined data to ensure consistent encoding
                combined_values = pd.concat([train_processed[col], test_processed[col]]).unique()
                self.label_encoders[col].fit(combined_values)
            
            train_processed[col + '_encoded'] = self.label_encoders[col].transform(train_processed[col])
            test_processed[col + '_encoded'] = self.label_encoders[col].transform(test_processed[col])
        
        # Select features for model training
        self.feature_columns = [
            'Customer_ID', 'Transaction_Amount', 'Transaction_Hour',
            'Transaction_DayOfWeek', 'Transaction_Month',
            'Customer_Type_encoded', 'Transaction_Type_encoded'
        ]
        
        X_train = train_processed[self.feature_columns].values
        X_test = test_processed[self.feature_columns].values
        y_train = train_processed['Is_Fraudulent'].values
        y_test = test_processed['Is_Fraudulent'].values
        
        print(f"Feature columns: {self.feature_columns}")
        print(f"Training features shape: {X_train.shape}")
        print(f"Training labels distribution: {np.bincount(y_train)}")
        
        return X_train, X_test, y_train, y_test, train_processed, test_processed
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, model_type: str = 'randomforest') -> Any:
        """
        Train fraud detection model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: Type of model to train ('randomforest' or 'xgboost')
            
        Returns:
            Trained model
        """
        print(f"Training {model_type} model...")
        
        if model_type.lower() == 'randomforest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type.lower() == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            )
        else:
            raise ValueError("Model type must be 'randomforest' or 'xgboost'")
        
        self.model.fit(X_train, y_train)
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nFeature Importances:")
            print(importance_df)
        
        return self.model
    
    def calculate_basic_bias_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   sensitive_attr: np.ndarray, attr_name: str) -> Dict[str, Any]:
        """
        Calculate basic bias metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attr: Sensitive attribute values
            attr_name: Name of sensitive attribute
            
        Returns:
            Dictionary of bias metrics
        """
        print(f"Calculating basic bias metrics for {attr_name}...")
        
        metrics = {}
        unique_groups = np.unique(sensitive_attr)
        
        # Selection rates per group
        selection_rates = {}
        for group in unique_groups:
            mask = sensitive_attr == group
            if np.sum(mask) > 0:
                selection_rates[f"{attr_name}_{group}"] = selection_rate(y_true[mask], y_pred[mask])
        
        metrics['selection_rates'] = selection_rates
        
        # Demographic parity difference and ratio
        try:
            dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_attr)
            dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_attr)
            
            metrics['demographic_parity_difference'] = dp_diff
            metrics['demographic_parity_ratio'] = dp_ratio
        except Exception as e:
            print(f"Error calculating demographic parity for {attr_name}: {e}")
            metrics['demographic_parity_difference'] = None
            metrics['demographic_parity_ratio'] = None
        
        return metrics
    
    def calculate_advanced_bias_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      sensitive_attr: np.ndarray, attr_name: str) -> Dict[str, Any]:
        """
        Calculate advanced bias metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attr: Sensitive attribute values
            attr_name: Name of sensitive attribute
            
        Returns:
            Dictionary of advanced bias metrics
        """
        print(f"Calculating advanced bias metrics for {attr_name}...")
        
        metrics = {}
        unique_groups = np.unique(sensitive_attr)
        
        # Calculate per-group metrics
        group_metrics = {}
        for group in unique_groups:
            mask = sensitive_attr == group
            if np.sum(mask) > 0:
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]
                
                # Confusion matrix components
                tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1]).ravel()
                
                # Calculate rates
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
                
                group_metrics[group] = {
                    'tpr': tpr,
                    'fpr': fpr,
                    'fnr': fnr,
                    'tp': tp,
                    'fp': fp,
                    'tn': tn,
                    'fn': fn
                }
        
        # Calculate differences between groups
        if len(unique_groups) >= 2:
            groups = list(unique_groups)
            
            # Equal Opportunity Difference (TPR difference)
            tpr_diff = abs(group_metrics[groups[0]]['tpr'] - group_metrics[groups[1]]['tpr'])
            metrics['equal_opportunity_difference'] = tpr_diff
            
            # Predictive Equality (FPR difference)
            fpr_diff = abs(group_metrics[groups[0]]['fpr'] - group_metrics[groups[1]]['fpr'])
            metrics['predictive_equality_difference'] = fpr_diff
            
            # Equalized Odds (max of TPR and FPR differences)
            metrics['equalized_odds_difference'] = max(tpr_diff, fpr_diff)
            
            # False Negative Rate difference
            fnr_diff = abs(group_metrics[groups[0]]['fnr'] - group_metrics[groups[1]]['fnr'])
            metrics['fnr_difference'] = fnr_diff
            
            # Theil Index calculation
            theil_index = self._calculate_theil_index(y_true, y_pred, sensitive_attr)
            metrics['theil_index'] = theil_index
        
        metrics['group_metrics'] = group_metrics
        return metrics
    
    def _calculate_theil_index(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              sensitive_attr: np.ndarray) -> float:
        """Calculate Theil Index for measuring inequality."""
        unique_groups = np.unique(sensitive_attr)
        total_samples = len(y_true)
        
        # Calculate advantage for each group (precision or similar metric)
        group_advantages = []
        group_sizes = []
        
        for group in unique_groups:
            mask = sensitive_attr == group
            if np.sum(mask) > 0:
                group_y_true = y_true[mask]
                group_y_pred = y_pred[mask]
                
                # Calculate advantage as positive prediction rate
                advantage = np.mean(group_y_pred) if len(group_y_pred) > 0 else 0
                group_advantages.append(advantage)
                group_sizes.append(np.sum(mask))
        
        if len(group_advantages) == 0:
            return 0.0
        
        # Calculate overall advantage
        overall_advantage = np.mean(y_pred)
        
        if overall_advantage == 0:
            return 0.0
        
        # Calculate Theil index
        theil = 0.0
        for i, group_adv in enumerate(group_advantages):
            if group_adv > 0:
                weight = group_sizes[i] / total_samples
                theil += weight * (group_adv / overall_advantage) * np.log(group_adv / overall_advantage)
        
        return theil
    
    def analyze_intersectional_bias(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze bias at intersections of sensitive attributes.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            test_df: Test dataframe with sensitive attributes
            
        Returns:
            Dictionary of intersectional bias metrics
        """
        print("Analyzing intersectional bias...")
        
        metrics = {}
        
        # Create intersectional groups
        test_df = test_df.copy()
        test_df['intersectional_group'] = (
            test_df['Customer_Type'].astype(str) + '_' + 
            test_df['Transaction_Type'].astype(str)
        )
        
        intersectional_attr = test_df['intersectional_group'].values
        unique_intersections = np.unique(intersectional_attr)
        
        print(f"Found {len(unique_intersections)} intersectional groups: {unique_intersections}")
        
        # Calculate metrics for each intersection
        intersection_metrics = {}
        for intersection in unique_intersections:
            mask = intersectional_attr == intersection
            if np.sum(mask) > 10:  # Only analyze groups with sufficient samples
                y_true_int = y_true[mask]
                y_pred_int = y_pred[mask]
                
                # Basic metrics
                selection_rate_int = selection_rate(y_true_int, y_pred_int)
                
                # Confusion matrix
                if len(np.unique(y_true_int)) > 1 and len(np.unique(y_pred_int)) > 1:
                    tn, fp, fn, tp = confusion_matrix(y_true_int, y_pred_int, labels=[0, 1]).ravel()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                else:
                    tpr, fpr = 0, 0
                
                intersection_metrics[intersection] = {
                    'sample_size': np.sum(mask),
                    'selection_rate': selection_rate_int,
                    'tpr': tpr,
                    'fpr': fpr
                }
        
        metrics['intersection_metrics'] = intersection_metrics
        
        # Calculate max differences across intersections
        if len(intersection_metrics) > 1:
            selection_rates = [m['selection_rate'] for m in intersection_metrics.values()]
            tprs = [m['tpr'] for m in intersection_metrics.values()]
            fprs = [m['fpr'] for m in intersection_metrics.values()]
            
            metrics['max_selection_rate_diff'] = max(selection_rates) - min(selection_rates)
            metrics['max_tpr_diff'] = max(tprs) - min(tprs)
            metrics['max_fpr_diff'] = max(fprs) - min(fprs)
        
        return metrics
    
    def perform_counterfactual_testing(self, X_test: np.ndarray, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform counterfactual fairness testing.
        
        Args:
            X_test: Test features
            test_df: Test dataframe
            
        Returns:
            Dictionary of counterfactual testing results
        """
        print("Performing counterfactual fairness testing...")
        
        if self.model is None:
            raise ValueError("Model must be trained before counterfactual testing")
        
        results = {}
        
        # Original predictions
        original_predictions = self.model.predict_proba(X_test)[:, 1]
        
        # Test 1: Flip Customer_Type
        test_cf1 = test_df.copy()
        customer_types = test_cf1['Customer_Type'].unique()
        if len(customer_types) >= 2:
            # Create mapping to flip customer types (handle all possible values)
            type_mapping = {}
            customer_types_list = list(customer_types)
            for i, ctype in enumerate(customer_types_list):
                # Map to the next type in a circular fashion
                next_idx = (i + 1) % len(customer_types_list)
                type_mapping[ctype] = customer_types_list[next_idx]
            
            test_cf1['Customer_Type'] = test_cf1['Customer_Type'].map(type_mapping)
            # Fill any NaN values that might occur
            test_cf1['Customer_Type'] = test_cf1['Customer_Type'].fillna(customer_types_list[0])
            test_cf1['Customer_Type_encoded'] = self.label_encoders['Customer_Type'].transform(test_cf1['Customer_Type'])
            
            X_test_cf1 = test_cf1[self.feature_columns].values
            cf1_predictions = self.model.predict_proba(X_test_cf1)[:, 1]
            
            # Calculate prediction changes
            prediction_changes = np.abs(original_predictions - cf1_predictions)
            results['customer_type_flip'] = {
                'mean_prediction_change': np.mean(prediction_changes),
                'max_prediction_change': np.max(prediction_changes),
                'samples_with_change': np.sum(prediction_changes > 0.01),
                'percentage_changed': np.sum(prediction_changes > 0.01) / len(prediction_changes) * 100
            }
        
        # Test 2: Flip Transaction_Type
        test_cf2 = test_df.copy()
        transaction_types = test_cf2['Transaction_Type'].unique()
        if len(transaction_types) >= 2:
            # Create mapping to flip transaction types (handle all possible values)
            type_mapping = {}
            transaction_types_list = list(transaction_types)
            for i, ttype in enumerate(transaction_types_list):
                # Map to the next type in a circular fashion
                next_idx = (i + 1) % len(transaction_types_list)
                type_mapping[ttype] = transaction_types_list[next_idx]
            
            test_cf2['Transaction_Type'] = test_cf2['Transaction_Type'].map(type_mapping)
            # Fill any NaN values that might occur
            test_cf2['Transaction_Type'] = test_cf2['Transaction_Type'].fillna(transaction_types_list[0])
            test_cf2['Transaction_Type_encoded'] = self.label_encoders['Transaction_Type'].transform(test_cf2['Transaction_Type'])
            
            X_test_cf2 = test_cf2[self.feature_columns].values
            cf2_predictions = self.model.predict_proba(X_test_cf2)[:, 1]
            
            prediction_changes = np.abs(original_predictions - cf2_predictions)
            results['transaction_type_flip'] = {
                'mean_prediction_change': np.mean(prediction_changes),
                'max_prediction_change': np.max(prediction_changes),
                'samples_with_change': np.sum(prediction_changes > 0.01),
                'percentage_changed': np.sum(prediction_changes > 0.01) / len(prediction_changes) * 100
            }
        
        # Test 3: Flip both attributes
        if len(customer_types) >= 2 and len(transaction_types) >= 2:
            test_cf3 = test_df.copy()
            # Apply both mappings
            customer_types_list = list(customer_types)
            transaction_types_list = list(transaction_types)
            
            # Create customer mapping
            customer_mapping = {}
            for i, ctype in enumerate(customer_types_list):
                next_idx = (i + 1) % len(customer_types_list)
                customer_mapping[ctype] = customer_types_list[next_idx]
            
            # Create transaction mapping
            transaction_mapping = {}
            for i, ttype in enumerate(transaction_types_list):
                next_idx = (i + 1) % len(transaction_types_list)
                transaction_mapping[ttype] = transaction_types_list[next_idx]
            
            test_cf3['Customer_Type'] = test_cf3['Customer_Type'].map(customer_mapping)
            test_cf3['Transaction_Type'] = test_cf3['Transaction_Type'].map(transaction_mapping)
            # Fill any NaN values that might occur
            test_cf3['Customer_Type'] = test_cf3['Customer_Type'].fillna(customer_types_list[0])
            test_cf3['Transaction_Type'] = test_cf3['Transaction_Type'].fillna(transaction_types_list[0])
            test_cf3['Customer_Type_encoded'] = self.label_encoders['Customer_Type'].transform(test_cf3['Customer_Type'])
            test_cf3['Transaction_Type_encoded'] = self.label_encoders['Transaction_Type'].transform(test_cf3['Transaction_Type'])
            
            X_test_cf3 = test_cf3[self.feature_columns].values
            cf3_predictions = self.model.predict_proba(X_test_cf3)[:, 1]
            
            prediction_changes = np.abs(original_predictions - cf3_predictions)
            results['both_attributes_flip'] = {
                'mean_prediction_change': np.mean(prediction_changes),
                'max_prediction_change': np.max(prediction_changes),
                'samples_with_change': np.sum(prediction_changes > 0.01),
                'percentage_changed': np.sum(prediction_changes > 0.01) / len(prediction_changes) * 100
            }
        
        return results
    
    def perform_shap_analysis(self, X_test: np.ndarray, test_df: pd.DataFrame, 
                            sample_size: int = 1000) -> Dict[str, Any]:
        """
        Perform SHAP explainability analysis.
        
        Args:
            X_test: Test features
            test_df: Test dataframe
            sample_size: Number of samples to analyze (for performance)
            
        Returns:
            Dictionary of SHAP analysis results
        """
        print("Performing SHAP explainability analysis...")
        
        if self.model is None:
            raise ValueError("Model must be trained before SHAP analysis")
        
        # Sample data for SHAP analysis (for performance)
        if len(X_test) > sample_size:
            indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample = X_test[indices]
            test_sample = test_df.iloc[indices].copy()
        else:
            X_sample = X_test
            test_sample = test_df.copy()
        
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class SHAP values
            
            results = {}
            
            # Overall feature importance from SHAP
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'mean_abs_shap': mean_abs_shap
            }).sort_values('mean_abs_shap', ascending=False)
            
            results['feature_importance'] = feature_importance_df.to_dict('records')
            
            # Analyze SHAP values by sensitive attributes
            sensitive_features_idx = []
            for attr in self.sensitive_attributes:
                if f"{attr}_encoded" in self.feature_columns:
                    idx = self.feature_columns.index(f"{attr}_encoded")
                    sensitive_features_idx.append(idx)
            
            if sensitive_features_idx:
                sensitive_shap_importance = np.sum(mean_abs_shap[sensitive_features_idx])
                total_importance = np.sum(mean_abs_shap)
                
                results['sensitive_attribute_importance'] = {
                    'absolute_importance': sensitive_shap_importance,
                    'relative_importance': sensitive_shap_importance / total_importance if total_importance > 0 else 0,
                    'percentage': (sensitive_shap_importance / total_importance * 100) if total_importance > 0 else 0
                }
            
            # Analyze SHAP value distributions by group
            group_analysis = {}
            for attr in self.sensitive_attributes:
                if attr in test_sample.columns:
                    unique_groups = test_sample[attr].unique()
                    if f"{attr}_encoded" in self.feature_columns:
                        attr_idx = self.feature_columns.index(f"{attr}_encoded")
                        
                        group_shap_stats = {}
                        for group in unique_groups:
                            mask = test_sample[attr].values == group
                            if np.sum(mask) > 0:
                                group_shap_values = shap_values[mask, attr_idx]
                                group_shap_stats[str(group)] = {
                                    'mean': np.mean(group_shap_values),
                                    'std': np.std(group_shap_values),
                                    'median': np.median(group_shap_values)
                                }
                        
                        group_analysis[attr] = group_shap_stats
            
            results['group_analysis'] = group_analysis
            
            return results
            
        except Exception as e:
            print(f"Error in SHAP analysis: {e}")
            return {'error': str(e)}
    
    def generate_bias_report(self, all_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate comprehensive bias report.
        
        Args:
            all_results: Dictionary containing all bias testing results
            
        Returns:
            DataFrame containing bias report
        """
        print("Generating bias report...")
        
        report_data = []
        
        # Basic metrics
        for attr in self.sensitive_attributes:
            if attr in all_results.get('basic_metrics', {}):
                basic_metrics = all_results['basic_metrics'][attr]
                
                # Selection rates
                for group, rate in basic_metrics.get('selection_rates', {}).items():
                    report_data.append({
                        'metric_category': 'Basic',
                        'metric_name': 'Selection Rate',
                        'attribute': attr,
                        'group': group.replace(f"{attr}_", ""),
                        'value': rate,
                        'interpretation': 'Rate of positive predictions'
                    })
                
                # Demographic parity
                if basic_metrics.get('demographic_parity_difference') is not None:
                    report_data.append({
                        'metric_category': 'Basic',
                        'metric_name': 'Demographic Parity Difference',
                        'attribute': attr,
                        'group': 'Overall',
                        'value': basic_metrics['demographic_parity_difference'],
                        'interpretation': 'Difference in selection rates (closer to 0 is better)'
                    })
        
        # Advanced metrics
        for attr in self.sensitive_attributes:
            if attr in all_results.get('advanced_metrics', {}):
                advanced_metrics = all_results['advanced_metrics'][attr]
                
                metrics_to_add = [
                    ('Equal Opportunity Difference', 'equal_opportunity_difference', 'TPR difference between groups'),
                    ('Predictive Equality Difference', 'predictive_equality_difference', 'FPR difference between groups'),
                    ('Equalized Odds Difference', 'equalized_odds_difference', 'Max of TPR and FPR differences'),
                    ('FNR Difference', 'fnr_difference', 'False Negative Rate difference'),
                    ('Theil Index', 'theil_index', 'Inequality measure (lower is better)')
                ]
                
                for metric_name, metric_key, interpretation in metrics_to_add:
                    if metric_key in advanced_metrics:
                        report_data.append({
                            'metric_category': 'Advanced',
                            'metric_name': metric_name,
                            'attribute': attr,
                            'group': 'Overall',
                            'value': advanced_metrics[metric_key],
                            'interpretation': interpretation
                        })
        
        # Intersectional metrics
        if 'intersectional_bias' in all_results:
            intersectional = all_results['intersectional_bias']
            
            if 'max_selection_rate_diff' in intersectional:
                report_data.append({
                    'metric_category': 'Intersectional',
                    'metric_name': 'Max Selection Rate Difference',
                    'attribute': 'Customer_Type x Transaction_Type',
                    'group': 'Overall',
                    'value': intersectional['max_selection_rate_diff'],
                    'interpretation': 'Maximum selection rate difference across intersections'
                })
        
        # Counterfactual results
        if 'counterfactual_testing' in all_results:
            cf_results = all_results['counterfactual_testing']
            
            for test_name, test_results in cf_results.items():
                if isinstance(test_results, dict):
                    report_data.append({
                        'metric_category': 'Counterfactual',
                        'metric_name': f'{test_name.replace("_", " ").title()} - Mean Change',
                        'attribute': test_name,
                        'group': 'Overall',
                        'value': test_results.get('mean_prediction_change', 0),
                        'interpretation': 'Average prediction change when flipping attribute'
                    })
                    
                    report_data.append({
                        'metric_category': 'Counterfactual',
                        'metric_name': f'{test_name.replace("_", " ").title()} - % Changed',
                        'attribute': test_name,
                        'group': 'Overall',
                        'value': test_results.get('percentage_changed', 0),
                        'interpretation': 'Percentage of predictions that changed'
                    })
        
        # SHAP results
        if 'shap_analysis' in all_results and 'sensitive_attribute_importance' in all_results['shap_analysis']:
            shap_results = all_results['shap_analysis']['sensitive_attribute_importance']
            
            report_data.append({
                'metric_category': 'Explainability',
                'metric_name': 'Sensitive Attribute Importance',
                'attribute': 'Customer_Type + Transaction_Type',
                'group': 'Overall',
                'value': shap_results.get('percentage', 0),
                'interpretation': 'Percentage of total SHAP importance from sensitive attributes'
            })
        
        # Create DataFrame
        report_df = pd.DataFrame(report_data)
        
        return report_df
    
    def run_complete_analysis(self, train_path: str, test_path: str, model_type: str = 'randomforest') -> Dict[str, Any]:
        """
        Run complete bias analysis pipeline.
        
        Args:
            train_path: Path to training data
            test_path: Path to testing data
            model_type: Type of model to train
            
        Returns:
            Dictionary containing all analysis results
        """
        print("="*80)
        print("ADVANCED BIAS TESTING FOR FRAUD DETECTION")
        print("="*80)
        
        # Load and preprocess data
        train_df, test_df = self.load_data(train_path, test_path)
        X_train, X_test, y_train, y_test, train_processed, test_processed = self.preprocess_data(train_df, test_df)
        
        # Train model
        model = self.train_model(X_train, y_train, model_type)
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Model performance
        print(f"\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
        # Initialize results dictionary
        all_results = {
            'model_performance': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Basic bias metrics
        basic_metrics = {}
        for attr in self.sensitive_attributes:
            if attr in test_processed.columns:
                sensitive_attr = test_processed[attr].values
                basic_metrics[attr] = self.calculate_basic_bias_metrics(y_test, y_pred, sensitive_attr, attr)
        all_results['basic_metrics'] = basic_metrics
        
        # Advanced bias metrics
        advanced_metrics = {}
        for attr in self.sensitive_attributes:
            if attr in test_processed.columns:
                sensitive_attr = test_processed[attr].values
                advanced_metrics[attr] = self.calculate_advanced_bias_metrics(y_test, y_pred, sensitive_attr, attr)
        all_results['advanced_metrics'] = advanced_metrics
        
        # Intersectional bias analysis
        intersectional_bias = self.analyze_intersectional_bias(y_test, y_pred, test_processed)
        all_results['intersectional_bias'] = intersectional_bias
        
        # Counterfactual testing
        counterfactual_results = self.perform_counterfactual_testing(X_test, test_processed)
        all_results['counterfactual_testing'] = counterfactual_results
        
        # SHAP analysis
        shap_results = self.perform_shap_analysis(X_test, test_processed)
        all_results['shap_analysis'] = shap_results
        
        # Generate bias report
        bias_report_df = self.generate_bias_report(all_results)
        all_results['bias_report_df'] = bias_report_df
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print summary of bias analysis results."""
        print("\n" + "="*80)
        print("BIAS ANALYSIS SUMMARY")
        print("="*80)
        
        # Basic metrics summary
        print("\n📊 BASIC BIAS METRICS:")
        for attr in self.sensitive_attributes:
            if attr in results.get('basic_metrics', {}):
                metrics = results['basic_metrics'][attr]
                print(f"\n{attr}:")
                
                # Selection rates
                if 'selection_rates' in metrics:
                    print("  Selection Rates:")
                    for group, rate in metrics['selection_rates'].items():
                        print(f"    {group}: {rate:.3f}")
                
                # Demographic parity
                if metrics.get('demographic_parity_difference') is not None:
                    dp_diff = metrics['demographic_parity_difference']
                    print(f"  Demographic Parity Difference: {dp_diff:.3f}")
                    if abs(dp_diff) < 0.1:
                        print("    ✅ Low bias (< 0.1)")
                    elif abs(dp_diff) < 0.2:
                        print("    ⚠️  Moderate bias (0.1-0.2)")
                    else:
                        print("    🚨 High bias (> 0.2)")
        
        # Advanced metrics summary
        print("\n🔍 ADVANCED BIAS METRICS:")
        for attr in self.sensitive_attributes:
            if attr in results.get('advanced_metrics', {}):
                metrics = results['advanced_metrics'][attr]
                print(f"\n{attr}:")
                
                advanced_metrics_list = [
                    ('Equal Opportunity Difference', 'equal_opportunity_difference'),
                    ('Predictive Equality Difference', 'predictive_equality_difference'),
                    ('Equalized Odds Difference', 'equalized_odds_difference'),
                    ('FNR Difference', 'fnr_difference'),
                    ('Theil Index', 'theil_index')
                ]
                
                for metric_name, metric_key in advanced_metrics_list:
                    if metric_key in metrics:
                        value = metrics[metric_key]
                        print(f"  {metric_name}: {value:.3f}")
        
        # Intersectional bias summary
        if 'intersectional_bias' in results:
            print("\n🔀 INTERSECTIONAL BIAS:")
            intersectional = results['intersectional_bias']
            
            if 'max_selection_rate_diff' in intersectional:
                print(f"  Max Selection Rate Difference: {intersectional['max_selection_rate_diff']:.3f}")
            
            if 'intersection_metrics' in intersectional:
                print(f"  Number of intersectional groups analyzed: {len(intersectional['intersection_metrics'])}")
        
        # Counterfactual testing summary
        if 'counterfactual_testing' in results:
            print("\n🔄 COUNTERFACTUAL FAIRNESS:")
            cf_results = results['counterfactual_testing']
            
            for test_name, test_results in cf_results.items():
                if isinstance(test_results, dict):
                    pct_changed = test_results.get('percentage_changed', 0)
                    mean_change = test_results.get('mean_prediction_change', 0)
                    print(f"  {test_name.replace('_', ' ').title()}:")
                    print(f"    Predictions changed: {pct_changed:.1f}%")
                    print(f"    Mean change: {mean_change:.3f}")
        
        # SHAP analysis summary
        if 'shap_analysis' in results and 'sensitive_attribute_importance' in results['shap_analysis']:
            print("\n🔍 EXPLAINABILITY (SHAP):")
            shap_results = results['shap_analysis']['sensitive_attribute_importance']
            pct_importance = shap_results.get('percentage', 0)
            print(f"  Sensitive attributes account for {pct_importance:.1f}% of model importance")
            
            if pct_importance < 10:
                print("    ✅ Low reliance on sensitive attributes")
            elif pct_importance < 25:
                print("    ⚠️  Moderate reliance on sensitive attributes")
            else:
                print("    🚨 High reliance on sensitive attributes")
        
        print("\n" + "="*80)


def main():
    """Main function to run bias testing from command line."""
    parser = argparse.ArgumentParser(description='Advanced Bias Testing for Fraud Detection Models')
    parser.add_argument('--train', required=True, help='Path to training CSV file')
    parser.add_argument('--test', required=True, help='Path to testing CSV file')
    parser.add_argument('--model', default='randomforest', choices=['randomforest', 'xgboost'],
                       help='Model type to train (default: randomforest)')
    parser.add_argument('--output', default='bias_report.csv',
                       help='Output file for bias report (default: bias_report.csv)')
    
    args = parser.parse_args()
    
    # Initialize framework and run analysis
    framework = BiasTestingFramework()
    results = framework.run_complete_analysis(args.train, args.test, args.model)
    
    # Save bias report
    if 'bias_report_df' in results:
        results['bias_report_df'].to_csv(args.output, index=False)
        print(f"\n📄 Bias report saved to: {args.output}")
    
    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    main()