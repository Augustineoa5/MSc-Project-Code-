# app.py (enhanced version with sequential training)
from flask import Flask, render_template, request, jsonify, send_file, session, send_from_directory
import os
import pandas as pd
from utils import clean_data, train_rf, train_if, train_dt, train_xgb, evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import pickle
import traceback
import logging
import numpy as np
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['MODELS_FOLDER'] = 'trained_models'
app.secret_key = 'your-secret-key-here'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'plots'), exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)


class ModelStorage:
    """Class to manage trained models storage and retrieval"""
    
    @staticmethod
    def get_models_file():
        return os.path.join(app.config['MODELS_FOLDER'], 'trained_models.json')
    
    @staticmethod
    def get_model_file(model_id):
        return os.path.join(app.config['MODELS_FOLDER'], f'model_{model_id}.pkl')
    
    @staticmethod
    def load_trained_models():
        """Load list of trained models"""
        try:
            models_file = ModelStorage.get_models_file()
            if os.path.exists(models_file):
                with open(models_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading trained models: {str(e)}")
            return []
    
    @staticmethod
    def save_trained_models(models_list):
        """Save list of trained models"""
        try:
            models_file = ModelStorage.get_models_file()
            with open(models_file, 'w') as f:
                json.dump(models_list, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving trained models list: {str(e)}")
            return False
    
    @staticmethod
    def add_model(model_type, trained_model, best_features, feature_subsets, train_results):
        """Add a new trained model"""
        try:
            models_list = ModelStorage.load_trained_models()
            
            # Generate unique model ID
            model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save model object
            model_file = ModelStorage.get_model_file(model_id)
            model_data = {
                'model': trained_model,
                'best_features': best_features,
                'feature_subsets': feature_subsets
            }
            
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Add model info to list
            model_info = {
                'id': model_id,
                'type': model_type,
                'timestamp': datetime.now().isoformat(),
                'best_features': best_features,
                'train_results': train_results,
                'test_results': {}  # Will store test results for different datasets
            }
            
            models_list.append(model_info)
            ModelStorage.save_trained_models(models_list)
            
            logger.info(f"Model {model_id} saved successfully")
            return model_id
            
        except Exception as e:
            logger.error(f"Error adding model: {str(e)}")
            return None
    
    @staticmethod
    def load_model(model_id):
        """Load a specific trained model"""
        try:
            model_file = ModelStorage.get_model_file(model_id)
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            return None
    
    @staticmethod
    def update_test_results(model_id, test_dataset_name, test_results):
        """Update test results for a specific model"""
        try:
            models_list = ModelStorage.load_trained_models()
            
            for model_info in models_list:
                if model_info['id'] == model_id:
                    model_info['test_results'][test_dataset_name] = {
                        'results': test_results,
                        'timestamp': datetime.now().isoformat()
                    }
                    ModelStorage.save_trained_models(models_list)
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error updating test results for {model_id}: {str(e)}")
            return False
    
    @staticmethod
    def clear_all_models():
        """Clear all trained models"""
        try:
            models_list = ModelStorage.load_trained_models()
            
            # Remove model files
            for model_info in models_list:
                model_file = ModelStorage.get_model_file(model_info['id'])
                if os.path.exists(model_file):
                    os.remove(model_file)
            
            # Clear models list
            ModelStorage.save_trained_models([])
            
            # Clear plot files
            plots_dir = os.path.join(app.config['STATIC_FOLDER'], 'plots')
            if os.path.exists(plots_dir):
                for file in os.listdir(plots_dir):
                    if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                        try:
                            os.remove(os.path.join(plots_dir, file))
                        except:
                            pass
            
            return True
        except Exception as e:
            logger.error(f"Error clearing models: {str(e)}")
            return False

    @staticmethod
    def clear_test_results():
        """Clear all test results from trained models"""
        try:
            models_list = ModelStorage.load_trained_models()
            for model_info in models_list:
                model_info['test_results'] = {}
            ModelStorage.save_trained_models(models_list)
            return True
        except Exception as e:
            logger.error(f"Error clearing test results: {str(e)}")
            return False


def safe_evaluate_model(trained_model, X_test, y_test, best_features, model_type, static_folder, plot_suffix=""):
    """Enhanced evaluate_model with unique plot naming"""
    try:
        logger.info(f"Starting model evaluation for {model_type}")
        
        # Ensure we have the right features and proper DataFrame structure
        if hasattr(X_test, 'columns'):
            available_features = [f for f in best_features if f in X_test.columns]
            if len(available_features) != len(best_features):
                missing_features = set(best_features) - set(available_features)
                logger.warning(f"Missing features: {missing_features}")
            X_test_selected = X_test[available_features] if available_features else X_test
        else:
            X_test_selected = pd.DataFrame(X_test, columns=best_features)
            available_features = best_features
        
        # Create unique plot names to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_suffix = f"{model_type}_{timestamp}{plot_suffix}"
        
        # Temporarily modify plot naming in utils (if needed)
        # This assumes your utils.evaluate_model can handle unique naming
        results = evaluate_model(trained_model, X_test_selected, y_test, available_features, model_type, static_folder)
        
        # Ensure results has all required fields
        if 'model_type' not in results:
            results['model_type'] = model_type
        if 'features_used' not in results:
            results['features_used'] = available_features
        
        # Update plot paths with unique names if they exist
        if results.get('cm_plot'):
            old_cm_path = os.path.join(static_folder, 'plots', results['cm_plot'])
            new_cm_name = f'cm_{unique_suffix}.png'
            new_cm_path = os.path.join(static_folder, 'plots', new_cm_name)
            
            if os.path.exists(old_cm_path):
                try:
                    os.rename(old_cm_path, new_cm_path)
                    results['cm_plot'] = new_cm_name
                except:
                    pass
        
        if results.get('roc_plot'):
            old_roc_path = os.path.join(static_folder, 'plots', results['roc_plot'])
            new_roc_name = f'roc_{unique_suffix}.png'
            new_roc_path = os.path.join(static_folder, 'plots', new_roc_name)
            
            if os.path.exists(old_roc_path):
                try:
                    os.rename(old_roc_path, new_roc_path)
                    results['roc_plot'] = new_roc_name
                except:
                    pass
        
        return results
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        # Fallback evaluation code (same as original)
        try:
            if hasattr(X_test, 'columns'):
                available_features = [f for f in best_features if f in X_test.columns]
                if not available_features:
                    available_features = list(X_test.columns)[:len(best_features)]
                X_test_selected = X_test[available_features]
            else:
                X_test_selected = pd.DataFrame(X_test, columns=best_features)
                available_features = best_features
            
            y_pred = trained_model.predict(X_test_selected)
            
            if model_type == 'if':
                y_pred = np.where(y_pred == -1, 1, 0)
                try:
                    y_pred_proba = -trained_model.decision_function(X_test_selected)
                    y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
                except:
                    y_pred_proba = y_pred.astype(float)
            else:
                try:
                    y_pred_proba = trained_model.predict_proba(X_test_selected)
                    if y_pred_proba.shape[1] > 1:
                        y_pred_proba = y_pred_proba[:, 1]
                except:
                    y_pred_proba = y_pred.astype(float)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            try:
                if len(np.unique(y_test)) > 1:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                else:
                    roc_auc = 0.0
            except:
                roc_auc = 0.0
            
            try:
                report_dict = classification_report(
                    y_test, y_pred, 
                    target_names=['Normal', 'Anomaly'],
                    output_dict=True, 
                    zero_division=0
                )
            except:
                report_dict = {
                    'Normal': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
                    'Anomaly': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
                    'accuracy': accuracy,
                    'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': len(y_test)},
                    'weighted avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': len(y_test)}
                }
            
            return {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'features_used': available_features,
                'report': report_dict,
                'model_type': model_type,
                'cm_plot': None,
                'roc_plot': None,
                'error': f"Fallback evaluation used. Original error: {str(e)}"
            }
            
        except Exception as fallback_error:
            logger.error(f"Fallback evaluation failed: {str(fallback_error)}")
            return {
                'accuracy': 0.0,
                'roc_auc': 0.0,
                'features_used': best_features if best_features else [],
                'report': {
                    'Normal': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
                    'Anomaly': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
                    'accuracy': 0.0,
                    'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
                    'weighted avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}
                },
                'model_type': model_type,
                'cm_plot': None,
                'roc_plot': None,
                'error': f"Complete evaluation failure: {str(e)}"
            }


def format_results_html(results, model_id=None, is_training=True, timestamp=None):
    """Enhanced format results with model identification"""
    try:
        model_type = results.get('model_type', 'Unknown').upper()
        accuracy = results.get('accuracy', 0.0)
        roc_auc = results.get('roc_auc', 0.0)
        features_used = results.get('features_used', [])
        report = results.get('report', {})
        cm_plot = results.get('cm_plot')
        roc_plot = results.get('roc_plot')
        
        # Handle error case
        error_msg = ""
        if 'error' in results:
            error_msg = f"<div class='alert alert-warning'>Warning: {results['error']}</div>"
        
        # Model identification header
        phase = "Training" if is_training else "Testing"
        model_header = f"<div class='alert alert-info'><strong>{phase} Results for Model:</strong> {model_id}"
        if timestamp:
            model_header += f" <small>({timestamp})</small>"
        model_header += "</div>"
        
        # Safe access to classification report
        class_0_data = None
        class_1_data = None
        
        for key in ['0', 'Normal']:
            if key in report:
                class_0_data = report[key]
                break
        
        for key in ['1', 'Anomaly']:
            if key in report:
                class_1_data = report[key]
                break
        
        if class_0_data is None:
            class_0_data = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}
        if class_1_data is None:
            class_1_data = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}
        
        # Create plot buttons HTML
        plot_buttons = ""
        if cm_plot and roc_plot:
            plots_dir = os.path.join(app.config['STATIC_FOLDER'], 'plots')
            cm_exists = os.path.exists(os.path.join(plots_dir, cm_plot))
            roc_exists = os.path.exists(os.path.join(plots_dir, roc_plot))
            
            if cm_exists and roc_exists:
                plot_buttons = f"""
                <div class="mt-3">
                    <a href="/plot/{cm_plot}" target="_blank" class="btn btn-secondary btn-sm">View Confusion Matrix</a>
                    <a href="/plot/{roc_plot}" target="_blank" class="btn btn-secondary btn-sm">View ROC Curve</a>
                </div>
                """
            else:
                plot_buttons = "<div class='mt-3'><small class='text-muted'>Plots are being generated...</small></div>"
        else:
            plot_buttons = "<div class='mt-3'><small class='text-muted'>Plots not available</small></div>"
        
        html = f"""
        {model_header}
        {error_msg}
        <div class="card mb-3">
            <div class="card-body">
                <h6 class="card-title">Model Performance - {model_type}</h6>
                <p><strong>Accuracy:</strong> {accuracy:.3f}</p>
                <p><strong>ROC AUC:</strong> {roc_auc:.3f}</p>
                <p><strong>Features Used:</strong> {', '.join(features_used) if features_used else 'None'}</p>
                
                <h6 class="mt-3">Classification Report</h6>
                <table class='table table-sm table-striped'>
                    <thead>
                        <tr>
                            <th>Class</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1-Score</th>
                            <th>Support</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Normal (0)</td>
                            <td>{class_0_data.get('precision', 0.0):.3f}</td>
                            <td>{class_0_data.get('recall', 0.0):.3f}</td>
                            <td>{class_0_data.get('f1-score', 0.0):.3f}</td>
                            <td>{class_0_data.get('support', 0)}</td>
                        </tr>
                        <tr>
                            <td>Anomaly (1)</td>
                            <td>{class_1_data.get('precision', 0.0):.3f}</td>
                            <td>{class_1_data.get('recall', 0.0):.3f}</td>
                            <td>{class_1_data.get('f1-score', 0.0):.3f}</td>
                            <td>{class_1_data.get('support', 0)}</td>
                        </tr>
                    </tbody>
                </table>
                {plot_buttons}
            </div>
        </div>
        """
        return html
    except Exception as e:
        logger.error(f"Error formatting results HTML: {str(e)}")
        return f"<div class='alert alert-danger'>Error displaying results: {str(e)}</div>"


def get_app_state():
    """Get current application state"""
    try:
        train_data_uploaded = os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'X_train.pkl'))
        test_data_uploaded = os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'X_external.pkl'))
        trained_models = ModelStorage.load_trained_models()
        models_trained = len(trained_models) > 0
        
        return {
            'train_data_uploaded': train_data_uploaded,
            'test_data_uploaded': test_data_uploaded,
            'models_trained': models_trained,
            'trained_models': trained_models
        }
    except Exception as e:
        logger.error(f"Error getting app state: {str(e)}")
        return {
            'train_data_uploaded': False,
            'test_data_uploaded': False,
            'models_trained': False,
            'trained_models': []
        }


def safe_load_training_preview():
    """Safely load training data preview"""
    try:
        X_train = pd.read_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'X_train.pkl'))
        y_train = pd.read_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'y_train.pkl'))
        
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'features.txt'), 'r') as f:
            available_features = eval(f.read())
        
        if not hasattr(X_train, 'columns'):
            X_train = pd.DataFrame(X_train, columns=available_features)
        
        preview_df = X_train.head().copy()
        preview_df['target'] = y_train.head().astype(int)
        return preview_df.to_html(classes='table table-striped', index=False)
    except Exception as e:
        logger.error(f"Error loading training preview: {str(e)}")
        return None


def safe_load_test_preview():
    """Safely load test data preview"""
    try:
        X_external = pd.read_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'X_external.pkl'))
        y_external = pd.read_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'y_external.pkl'))
        
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'external_features.txt'), 'r') as f:
            available_features = eval(f.read())
        
        if not hasattr(X_external, 'columns'):
            X_external = pd.DataFrame(X_external, columns=available_features)
        
        preview_df = X_external.head().copy()
        preview_df['target'] = y_external.head().astype(int)
        return preview_df.to_html(classes='table table-striped', index=False)
    except Exception as e:
        logger.error(f"Error loading test preview: {str(e)}")
        return None


@app.route('/')
def index():
    logger.info("Serving index page")
    try:
        state = get_app_state()
        
        # Load training preview if available
        train_preview = None
        if state['train_data_uploaded']:
            train_preview = safe_load_training_preview()
        
        # Load test preview if available
        test_preview = None
        if state['test_data_uploaded']:
            test_preview = safe_load_test_preview()
        
        # Create display of all trained models
        train_results_html = ""
        if state['models_trained']:
            train_results_html = "<h5>Training Results:</h5>"
            for model_info in state['trained_models']:
                model_html = format_results_html(
                    model_info['train_results'], 
                    model_info['id'], 
                    is_training=True,
                    timestamp=model_info['timestamp']
                )
                train_results_html += model_html
        
        # Create display of all test results
        test_results_html = ""
        if state['models_trained'] and state['test_data_uploaded']:
            test_results_html = "<h5>Testing Results:</h5>"
            for model_info in state['trained_models']:
                if 'external' in model_info['test_results']:
                    model_html = format_results_html(
                        model_info['test_results']['external']['results'], 
                        model_info['id'], 
                        is_training=False,
                        timestamp=model_info['test_results']['external']['timestamp']
                    )
                    test_results_html += model_html
        
        return render_template('index.html', 
                             train_data_uploaded=state['train_data_uploaded'],
                             test_data_uploaded=state['test_data_uploaded'],
                             models_trained=state['models_trained'],
                             train_preview=train_preview,
                             train_results=train_results_html if train_results_html else None,
                             test_preview=test_preview,
                             test_results=test_results_html if test_results_html else None,
                             error_message=None,
                             success_message=None)
    except Exception as e:
        logger.error(f"Error serving index page: {str(e)}")
        return render_template('index.html', 
                             train_data_uploaded=False,
                             test_data_uploaded=False,
                             models_trained=False,
                             train_preview=None,
                             train_results=None,
                             test_preview=None,
                             test_results=None,
                             error_message=f"Application error: {str(e)}",
                             success_message=None)


@app.route('/upload_train', methods=['POST'])
def upload_train():
    logger.info("Received upload_train request")
    
    try:
        if 'file' not in request.files:
            logger.error("No file provided")
            return render_template('index.html', 
                                 error_message='No file provided',
                                 train_data_uploaded=False,
                                 test_data_uploaded=False,
                                 models_trained=False,
                                 train_preview=None,
                                 train_results=None,
                                 test_preview=None,
                                 test_results=None), 400
        
        file = request.files['file']
        logger.info(f"File received: {file.filename}")
        
        if not file.filename or not file.filename.endswith('.csv'):
            logger.error("Invalid file type")
            return render_template('index.html', 
                                 error_message='File must be a CSV',
                                 train_data_uploaded=False,
                                 test_data_uploaded=False,
                                 models_trained=False,
                                 train_preview=None,
                                 train_results=None,
                                 test_preview=None,
                                 test_results=None), 400
        
        logger.info("Reading CSV")
        df = pd.read_csv(file)
        logger.info(f"CSV loaded with shape: {df.shape}")
        
        if df.empty:
            raise ValueError("The uploaded CSV file is empty")
        
        logger.info("Cleaning data")
        X, y, available_features = clean_data(df)
        logger.info(f"Data cleaned. Shape: {X.shape}, Features: {available_features}")
        
        if len(available_features) == 0:
            raise ValueError("No valid features found in the data")
        
        logger.info("Splitting data")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        logger.info(f"Split completed. X_train shape: {X_train.shape}")
        
        logger.info("Saving files")
        X_train_df = pd.DataFrame(X_train, columns=available_features) if not hasattr(X_train, 'columns') else X_train
        X_test_df = pd.DataFrame(X_test, columns=available_features) if not hasattr(X_test, 'columns') else X_test
        
        X_train_df.to_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'X_train.pkl'))
        X_test_df.to_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'X_test.pkl'))
        pd.Series(y_train).to_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'y_train.pkl'))
        pd.Series(y_test).to_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'y_test.pkl'))
        
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'features.txt'), 'w') as f:
            f.write(str(available_features))
        
        # Clear existing models when new training data is uploaded
        ModelStorage.clear_all_models()
        
        train_preview = safe_load_training_preview()
        
        logger.info("Upload successful")
        return render_template('index.html', 
                             train_data_uploaded=True,
                             test_data_uploaded=False,
                             models_trained=False,
                             train_preview=train_preview,
                             train_results=None,
                             test_preview=None,
                             test_results=None,
                             success_message='Training data uploaded successfully! Previous models cleared.',
                             error_message=None)
                             
    except Exception as e:
        logger.error(f"Error in upload_train: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('index.html', 
                             error_message=f"Error processing file: {str(e)}",
                             train_data_uploaded=False,
                             test_data_uploaded=False,
                             models_trained=False,
                             train_preview=None,
                             train_results=None,
                             test_preview=None,
                             test_results=None), 500


@app.route('/train_model', methods=['POST'])
def train_model():
    logger.info("Received train_model request")
    
    model_type = request.form.get('model_type', 'rf')
    logger.info(f"Selected model type: {model_type}")
    
    try:
        # Check if training data exists
        required_files = ['X_train.pkl', 'y_train.pkl', 'X_test.pkl', 'y_test.pkl', 'features.txt']
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], file)):
                missing_files.append(file)
        
        if missing_files:
            error_msg = f"Missing training data files. Please upload training data first."
            logger.error(error_msg)
            return render_template('index.html',
                                 error_message=error_msg,
                                 train_data_uploaded=False,
                                 test_data_uploaded=False,
                                 models_trained=False,
                                 train_preview=None,
                                 train_results=None,
                                 test_preview=None,
                                 test_results=None), 400
        
        # Load training data
        logger.info("Loading training data")
        X_train = pd.read_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'X_train.pkl'))
        y_train = pd.read_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'y_train.pkl'))
        X_test = pd.read_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'X_test.pkl'))
        y_test = pd.read_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'y_test.pkl'))
        
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'features.txt'), 'r') as f:
            available_features = eval(f.read())
            
        # Ensure data has proper column names
        if not hasattr(X_train, 'columns'):
            X_train = pd.DataFrame(X_train, columns=available_features)
            X_test = pd.DataFrame(X_test, columns=available_features)
        
        # Train model based on type
        logger.info(f"Starting training for {model_type} model")
        if model_type == 'rf':
            trained_model, best_features, feature_subsets = train_rf(X_train, y_train, available_features)
        elif model_type == 'if':
            trained_model, best_features, feature_subsets = train_if(X_train, y_train, available_features)
        elif model_type == 'dt':
            trained_model, best_features, feature_subsets = train_dt(X_train, y_train, available_features)
        elif model_type == 'xgb':
            trained_model, best_features, feature_subsets = train_xgb(X_train, y_train, available_features)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        logger.info(f"Model training completed. Best features: {best_features}")
        
        # Evaluate model on internal test set
        logger.info("Evaluating trained model on internal test data")
        train_results = safe_evaluate_model(
            trained_model, X_test, y_test, best_features, 
            model_type, app.config['STATIC_FOLDER'], "_train"
        )
        
        # Save the trained model
        model_id = ModelStorage.add_model(model_type, trained_model, best_features, feature_subsets, train_results)
        
        if not model_id:
            raise Exception("Failed to save trained model")
        
        # Load all trained models for display
        state = get_app_state()
        train_results_html = "<h5>Training Results:</h5>"
        for model_info in state['trained_models']:
            model_html = format_results_html(
                model_info['train_results'], 
                model_info['id'], 
                is_training=True,
                timestamp=model_info['timestamp']
            )
            train_results_html += model_html
        
        train_preview = safe_load_training_preview()
        test_preview = safe_load_test_preview() if state['test_data_uploaded'] else None
        
        # Build test results if any
        test_results_html = ""
        if state['test_data_uploaded']:
            test_results_html = "<h5>Testing Results:</h5>"
            for model_info in state['trained_models']:
                if 'external' in model_info['test_results']:
                    model_html = format_results_html(
                        model_info['test_results']['external']['results'], 
                        model_info['id'], 
                        is_training=False,
                        timestamp=model_info['test_results']['external']['timestamp']
                    )
                    test_results_html += model_html
        
        logger.info("Training successful")
        return render_template('index.html', 
                             train_data_uploaded=state['train_data_uploaded'],
                             test_data_uploaded=state['test_data_uploaded'],
                             models_trained=state['models_trained'],
                             train_preview=train_preview,
                             train_results=train_results_html,
                             test_preview=test_preview,
                             test_results=test_results_html if test_results_html else None,
                             success_message=f'{model_type.upper()} model trained successfully!',
                             error_message=None)
                             
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('index.html', 
                             error_message=f"Error training model: {str(e)}",
                             train_data_uploaded=False,
                             test_data_uploaded=False,
                             models_trained=False,
                             train_preview=None,
                             train_results=None,
                             test_preview=None,
                             test_results=None), 500


@app.route('/upload_test_data', methods=['POST'])
def upload_test():
    logger.info("Received upload_test request")
    
    try:
        if 'file' not in request.files:
            logger.error("No file provided")
            return render_template('index.html', 
                                 error_message='No file provided',
                                 train_data_uploaded=False,
                                 test_data_uploaded=False,
                                 models_trained=False,
                                 train_preview=None,
                                 train_results=None,
                                 test_preview=None,
                                 test_results=None), 400
        
        file = request.files['file']
        logger.info(f"File received: {file.filename}")
        
        if not file.filename or not file.filename.endswith('.csv'):
            logger.error("Invalid file type")
            return render_template('index.html', 
                                 error_message='File must be a CSV',
                                 train_data_uploaded=False,
                                 test_data_uploaded=False,
                                 models_trained=False,
                                 train_preview=None,
                                 train_results=None,
                                 test_preview=None,
                                 test_results=None), 400
        
        logger.info("Reading CSV")
        df = pd.read_csv(file)
        logger.info(f"CSV loaded with shape: {df.shape}")
        
        if df.empty:
            raise ValueError("The uploaded CSV file is empty")
        
        logger.info("Cleaning data")
        X, y, available_features = clean_data(df)
        logger.info(f"Data cleaned. Shape: {X.shape}, Features: {available_features}")
        
        if len(available_features) == 0:
            raise ValueError("No valid features found in the data")
        
        logger.info("Saving files")
        X_df = pd.DataFrame(X, columns=available_features) if not hasattr(X, 'columns') else X
        
        X_df.to_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'X_external.pkl'))
        pd.Series(y).to_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'y_external.pkl'))
        
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'external_features.txt'), 'w') as f:
            f.write(str(available_features))
        
        # Clear existing test results when new test data is uploaded
        ModelStorage.clear_test_results()
        
        test_preview = safe_load_test_preview()
        
        state = get_app_state()
        train_preview = safe_load_training_preview() if state['train_data_uploaded'] else None
        
        # Build train results
        train_results_html = ""
        if state['models_trained']:
            train_results_html = "<h5>Training Results:</h5>"
            for model_info in state['trained_models']:
                model_html = format_results_html(
                    model_info['train_results'], 
                    model_info['id'], 
                    is_training=True,
                    timestamp=model_info['timestamp']
                )
                train_results_html += model_html
        
        logger.info("Upload successful")
        return render_template('index.html', 
                             train_data_uploaded=state['train_data_uploaded'],
                             test_data_uploaded=True,
                             models_trained=state['models_trained'],
                             train_preview=train_preview,
                             train_results=train_results_html if train_results_html else None,
                             test_preview=test_preview,
                             test_results=None,
                             success_message='Test data uploaded successfully! Previous test results cleared.',
                             error_message=None)
                             
    except Exception as e:
        logger.error(f"Error in upload_test: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('index.html', 
                             error_message=f"Error processing file: {str(e)}",
                             train_data_uploaded=False,
                             test_data_uploaded=False,
                             models_trained=False,
                             train_preview=None,
                             train_results=None,
                             test_preview=None,
                             test_results=None), 500


@app.route('/test_model', methods=['POST'])
def test_model():
    logger.info("Received test_model request")
    
    model_id = request.form.get('model_id')
    if not model_id:
        return render_template('index.html', 
                             error_message='No model ID provided',
                             train_data_uploaded=False,
                             test_data_uploaded=False,
                             models_trained=False,
                             train_preview=None,
                             train_results=None,
                             test_preview=None,
                             test_results=None), 400
    
    logger.info(f"Selected model ID: {model_id}")
    
    try:
        # Check if test data exists
        required_files = ['X_external.pkl', 'y_external.pkl', 'external_features.txt']
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], file)):
                missing_files.append(file)
        
        if missing_files:
            error_msg = f"Missing test data files. Please upload test data first."
            logger.error(error_msg)
            return render_template('index.html',
                                 error_message=error_msg,
                                 train_data_uploaded=False,
                                 test_data_uploaded=False,
                                 models_trained=False,
                                 train_preview=None,
                                 train_results=None,
                                 test_preview=None,
                                 test_results=None), 400
        
        # Load test data
        logger.info("Loading test data")
        X_external = pd.read_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'X_external.pkl'))
        y_external = pd.read_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'y_external.pkl'))
        
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'external_features.txt'), 'r') as f:
            available_features = eval(f.read())
            
        # Ensure data has proper column names
        if not hasattr(X_external, 'columns'):
            X_external = pd.DataFrame(X_external, columns=available_features)
        
        # Load the model
        model_data = ModelStorage.load_model(model_id)
        if not model_data:
            raise Exception(f"Failed to load model {model_id}")
        
        trained_model = model_data['model']
        best_features = model_data['best_features']
        
        # Get model type from trained models list
        trained_models = ModelStorage.load_trained_models()
        model_type = next((m['type'] for m in trained_models if m['id'] == model_id), 'unknown')
        
        # Evaluate model on external test set
        logger.info(f"Evaluating model {model_id} on external test data")
        test_results = safe_evaluate_model(
            trained_model, X_external, y_external, best_features, 
            model_type, app.config['STATIC_FOLDER'], "_test"
        )
        
        # Update test results
        updated = ModelStorage.update_test_results(model_id, 'external', test_results)
        if not updated:
            raise Exception("Failed to update test results")
        
        # Load state and build HTML
        state = get_app_state()
        train_preview = safe_load_training_preview() if state['train_data_uploaded'] else None
        test_preview = safe_load_test_preview() if state['test_data_uploaded'] else None
        
        train_results_html = ""
        if state['models_trained']:
            train_results_html = "<h5>Training Results:</h5>"
            for model_info in state['trained_models']:
                model_html = format_results_html(
                    model_info['train_results'], 
                    model_info['id'], 
                    is_training=True,
                    timestamp=model_info['timestamp']
                )
                train_results_html += model_html
        
        test_results_html = "<h5>Testing Results:</h5>"
        for model_info in state['trained_models']:
            if 'external' in model_info['test_results']:
                model_html = format_results_html(
                    model_info['test_results']['external']['results'], 
                    model_info['id'], 
                    is_training=False,
                    timestamp=model_info['test_results']['external']['timestamp']
                )
                test_results_html += model_html
        
        logger.info("Testing successful")
        return render_template('index.html', 
                             train_data_uploaded=state['train_data_uploaded'],
                             test_data_uploaded=state['test_data_uploaded'],
                             models_trained=state['models_trained'],
                             train_preview=train_preview,
                             train_results=train_results_html,
                             test_preview=test_preview,
                             test_results=test_results_html,
                             success_message=f'Model {model_id} tested successfully!',
                             error_message=None)
                             
    except Exception as e:
        logger.error(f"Error in test_model: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('index.html', 
                             error_message=f"Error testing model: {str(e)}",
                             train_data_uploaded=False,
                             test_data_uploaded=False,
                             models_trained=False,
                             train_preview=None,
                             train_results=None,
                             test_preview=None,
                             test_results=None), 500

@app.route('/test_all_models', methods=['POST'])
def test_all_models():
    logger.info("Received test_all_models request")
    
    try:
        # Check if test data exists
        required_files = ['X_external.pkl', 'y_external.pkl', 'external_features.txt']
        missing_files = [file for file in required_files if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], file))]
        
        if missing_files:
            error_msg = "Missing test data files. Please upload test data first."
            logger.error(error_msg)
            return render_template('index.html',
                                 error_message=error_msg,
                                 train_data_uploaded=False,
                                 test_data_uploaded=False,
                                 models_trained=False,
                                 train_preview=None,
                                 train_results=None,
                                 test_preview=None,
                                 test_results=None), 400
        
        # Load test data
        logger.info("Loading test data")
        X_external = pd.read_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'X_external.pkl'))
        y_external = pd.read_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'y_external.pkl'))
        
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'external_features.txt'), 'r') as f:
            available_features = eval(f.read())
            
        if not hasattr(X_external, 'columns'):
            X_external = pd.DataFrame(X_external, columns=available_features)
        
        # Load all trained models
        trained_models = ModelStorage.load_trained_models()
        if not trained_models:
            raise Exception("No trained models available")
        
        # Test each model
        for model_info in trained_models:
            model_id = model_info['id']
            model_type = model_info['type']
            
            model_data = ModelStorage.load_model(model_id)
            if not model_data:
                logger.warning(f"Failed to load model {model_id}, skipping")
                continue
            
            trained_model = model_data['model']
            best_features = model_data['best_features']
            
            logger.info(f"Evaluating model {model_id} on external test data")
            test_results = safe_evaluate_model(
                trained_model, X_external, y_external, best_features, 
                model_type, app.config['STATIC_FOLDER'], "_test"
            )
            
            # Update test results
            updated = ModelStorage.update_test_results(model_id, 'external', test_results)
            if not updated:
                logger.warning(f"Failed to update test results for {model_id}")
        
        # Load state and build HTML
        state = get_app_state()
        train_preview = safe_load_training_preview() if state['train_data_uploaded'] else None
        test_preview = safe_load_test_preview() if state['test_data_uploaded'] else None
        
        train_results_html = ""
        if state['models_trained']:
            train_results_html = "<h5>Training Results:</h5>"
            for model_info in state['trained_models']:
                model_html = format_results_html(
                    model_info['train_results'], 
                    model_info['id'], 
                    is_training=True,
                    timestamp=model_info['timestamp']
                )
                train_results_html += model_html
        
        test_results_html = ""
        if state['models_trained'] and state['test_data_uploaded']:
            test_results_html = "<h5>Testing Results:</h5>"
            for model_info in state['trained_models']:
                if 'external' in model_info['test_results']:
                    model_html = format_results_html(
                        model_info['test_results']['external']['results'], 
                        model_info['id'], 
                        is_training=False,
                        timestamp=model_info['test_results']['external']['timestamp']
                    )
                    test_results_html += model_html
        
        logger.info("All models tested successfully")
        return render_template('index.html', 
                             train_data_uploaded=state['train_data_uploaded'],
                             test_data_uploaded=state['test_data_uploaded'],
                             models_trained=state['models_trained'],
                             train_preview=train_preview,
                             train_results=train_results_html if train_results_html else None,
                             test_preview=test_preview,
                             test_results=test_results_html if test_results_html else None,
                             success_message='All models tested successfully!',
                             error_message=None)
                             
    except Exception as e:
        logger.error(f"Error in test_all_models: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('index.html', 
                             error_message=f"Error testing all models: {str(e)}",
                             train_data_uploaded=False,
                             test_data_uploaded=False,
                             models_trained=False,
                             train_preview=None,
                             train_results=None,
                             test_preview=None,
                             test_results=None), 500

    except Exception as e:
        logger.error(f"Error in test_all_models: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('index.html', 
                             error_message=f"Error testing all models: {str(e)}",
                             train_data_uploaded=False,
                             test_data_uploaded=False,
                             models_trained=False,
                             train_preview=None,
                             train_results=None,
                             test_preview=None,
                             test_results=None), 500

@app.route('/reset', methods=['POST'])
def reset():
    try:
        # Clear all trained models and their files
        success_models = ModelStorage.clear_all_models()
        if not success_models:
            raise Exception("Failed to clear trained models")

        # Clear all test results
        success_test = ModelStorage.clear_test_results()
        if not success_test:
            raise Exception("Failed to clear test results")

        # Clear uploaded files
        upload_dir = app.config['UPLOAD_FOLDER']
        for file in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {str(e)}")

        logger.info("Application reset successfully")
        return jsonify({'success': True, 'redirect': '/'})
    except Exception as e:
        logger.error(f"Error in reset: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/plot/<filename>')
def plot(filename):
    return send_from_directory(os.path.join(app.config['STATIC_FOLDER'], 'plots'), filename)

if __name__ == '__main__':
    app.run(debug=True)