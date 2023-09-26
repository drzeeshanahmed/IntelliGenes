# (Packages/Libraries) Matrix Manipulation
import pandas as pd
import numpy as np

# (Packages/Libraries) Statistical Analysis & Machine Learning
import sklearn
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# (Packages/Libraries) Miscellaneous
import os
from datetime import datetime
import argparse
import warnings

warnings.filterwarnings("ignore", message = "No data for colormapping provided via 'c'. Parameters 'vmin', 'vmax' will be ignored")

class DiseasePrediction:
    
    # Initialize DiseasePrediction Class 
    def __init__(self: 'DiseasePrediction', cgit_file: str,
                                            features_file: str, 
                                            output_dir: str,
                                            voting: str,
                                            random_state: 42,
                                            test_size: 0.3,
                                            n_splits: 5,
                                            use_rf = True,
                                            use_svm = True,
                                            use_xgb = True,
                                            use_knn = True,
                                            use_mlp = True,
                                            use_tuning = False,
                                            use_normalization = False,
                                            use_igenes = True,
                                            use_visualization = False):
        
        self.cgit_file = cgit_file
        self.features_file = features_file
        self.output_dir = output_dir
        self.voting = voting
        self.random_state = random_state
        self.test_size = test_size
        self.n_splits = n_splits
        self.use_rf = use_rf
        self.use_svm = use_svm
        self.use_xgb = use_xgb
        self.use_knn = use_knn
        self.use_mlp = use_mlp
        self.use_tuning = use_tuning
        self.use_normalization = use_normalization
        self.use_igenes = use_igenes
        self. use_visualization = use_visualization

        self.df = pd.read_csv(self.cgit_file)
        self.features = pd.read_csv(self.features_file)['Features'].values.flatten().tolist()
        if not self.features or self.features[0] == "Features":
            raise ValueError("Features not included.")
        
        self.y = self.df['Type']
        self.X = self.df[self.features]
        
        self.kfold = KFold(n_splits = self.n_splits, shuffle = True, random_state = self.random_state)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = self.test_size, random_state = self.random_state)
        
        self.classifiers = [] 
            
    # Random Forest Classifier 
    def rf_classifier(self: 'DiseasePrediction'):
        if self.use_rf:         
            print('Random Forest...')
            if self.use_normalization:
                pipeline = Pipeline([('scaling', StandardScaler()), ('rf_clf', RandomForestClassifier(random_state = self.random_state))])
            else:
                pipeline = Pipeline([('rf_clf', RandomForestClassifier(random_state = self.random_state))])
            
            if self.use_tuning:
                rf_parameters = {
                    'rf_clf__max_features': np.arange(1, self.X_train.shape[1] + 1),
                    'rf_clf__min_samples_split': np.arange(2, 11),
                    'rf_clf__min_samples_leaf': np.concatenate([np.arange(1, 11), [100, 150]])  
                }
                
                rf_clf = GridSearchCV(pipeline, param_grid = rf_parameters, cv = self.kfold, scoring = 'accuracy', n_jobs = -1, verbose = 0).fit(self.X_train, self.y_train)
            else: 
                rf_clf = pipeline.fit(self.X_train, self.y_train)
                
            if self.use_igenes or self.use_visualization:
                rf_importances = shap.TreeExplainer(rf_clf.named_steps['rf_clf']).shap_values(self.X_test)
            
            if self.use_igenes:
                rf_importances_extracted = np.mean(rf_importances[0], axis = 0)
                rf_importances_normalized = rf_importances_extracted / np.max(np.abs(rf_importances_extracted))
                rf_hhi = ((np.abs(rf_importances_normalized) / np.sum(rf_importances_normalized))**2)
                
            if self.use_visualization:
               shap.summary_plot(rf_importances[0], self.X_test, plot_type = "dot", show = False)
               
               plt.title("Random Forest Feature Importances", fontsize = 16)
               plt.xlabel("SHAP Value", fontsize = 14)
               plt.ylabel("Feature", fontsize = 14)
               plt.tight_layout()
               
               file_name = f"{self.cgit_file}_{datetime.now().strftime('%m-%d-%Y-%I-%M-%S-%p')}_RF-SHAP.png"
               rf_plot = os.path.join(self.output_dir, file_name)
               
               if not os.path.exists(self.output_dir):
                   os.makedirs(self.output_dir)
                   
               plt.savefig(rf_plot)
               plt.close()
               
            y_pred = rf_clf.predict(self.X_test)
            y_proba = rf_clf.predict_proba(self.X_test)[:, 1]
            
            rf_accuracy = accuracy_score(self.y_test, y_pred)
            rf_roc_auc = roc_auc_score(self.y_test, y_proba)
            rf_f1 = f1_score(self.y_test, y_pred, average = 'weighted')
            
            if self.use_igenes:
                return rf_clf, rf_accuracy, rf_roc_auc, rf_f1, rf_importances_normalized, rf_hhi
            else:
                return rf_clf, rf_accuracy, rf_roc_auc, rf_f1                  
        
        return None
    
    # Support Vector Machine Classifier 
    def svm_classifier(self: 'DiseasePrediction'):
        if self.use_svm:
            print('Support Vector Machine...')
            if self.use_normalization:
                pipeline = Pipeline([('scaling', StandardScaler()), ('svm_clf', SVC(random_state = self.random_state, kernel = 'linear', probability = True))])
            else:
                pipeline = Pipeline([('svm_clf', SVC(random_state = self.random_state, kernel = 'linear', probability = True))])
                
            if self.use_tuning:
                svm_parameters = {
                    "svm_clf__kernel": ['linear'],
                    'svm_clf__gamma': [0.001, 0.01, 0.1, 1, 10],
                    'svm_clf__C': [0.01, 0.1, 1, 10, 50, 100, 500, 1000, 5000, 10000]
                }
                
                svm_clf = GridSearchCV(pipeline, param_grid = svm_parameters, cv = self.kfold, scoring = 'accuracy', n_jobs = -1, verbose = 0).fit(self.X_train, self.y_train)
            else: 
                svm_clf = pipeline.fit(self.X_train, self.y_train)
                
            if self.use_igenes or self.use_visualization:
                svm_importances = shap.LinearExplainer(svm_clf.named_steps['svm_clf'], masker = shap.maskers.Independent(self.X_train)).shap_values(self.X_test)
            
            if self.use_igenes:
                svm_importances_extracted = svm_importances[0]
                svm_importances_normalized = svm_importances_extracted / np.max(np.abs(svm_importances_extracted))
                svm_hhi = ((np.abs(svm_importances_normalized) / np.sum(svm_importances_normalized))**2)
                
            if self.use_visualization:
                shap.summary_plot(svm_importances, self.X_test, plot_type = "dot", show = False)
                
                plt.title("Support Vector Machine Feature Importances", fontsize = 16)
                plt.xlabel("SHAP Value", fontsize = 14)
                plt.ylabel("Feature", fontsize = 14)
                plt.tight_layout()
                
                file_name = f"{self.cgit_file}_{datetime.now().strftime('%m-%d-%Y-%I-%M-%S-%p')}_SVM-SHAP.png"
                svm_plot = os.path.join(self.output_dir, file_name)
                
                if not os.path.exists(self.output_dir):
                   os.makedirs(self.output_dir)
                   
                plt.savefig(svm_plot)
                plt.close()
            
            y_pred = svm_clf.predict(self.X_test)
            y_proba = svm_clf.predict_proba(self.X_test)[:, 1]
            
            svm_accuracy = accuracy_score(self.y_test, y_pred)
            svm_roc_auc = roc_auc_score(self.y_test, y_proba)
            svm_f1 = f1_score(self.y_test, y_pred, average = 'weighted')
            
            if self.use_igenes:
                return svm_clf, svm_accuracy, svm_roc_auc, svm_f1, svm_importances_normalized, svm_hhi
            else: 
                return svm_clf, svm_accuracy, svm_roc_auc
            
        return None
    
    # XGBoost Classifier 
    def xgb_classifier(self: 'DiseasePrediction'):
        if self.use_xgb: 
            print('XGBoost...')
            if self.use_normalization:
                pipeline = Pipeline([('scaling', StandardScaler()), ('xgb_clf', xgb.XGBClassifier(random_state = self.random_state, objective = "reg:squarederror"))])
            else:
                pipeline = Pipeline([('xgb_clf', xgb.XGBClassifier(random_state = self.random_state, objective = "binary:logistic"))])
                
            if self.use_tuning:
                xgb_parameters = {
                    "xgb_clf__n_estimators": [int(x) for x in np.linspace(100, 500, 5)],
                    "xgb_clf__max_depth": [int(x) for x in np.linspace(3, 9, 4)],
                    "xgb_clf__gamma": [0.01, 0.1],
                    "xgb_clf__learning_rate": [0.001, 0.01, 0.1, 1]
                }
                
                xgb_clf = GridSearchCV(pipeline, param_grid = xgb_parameters, cv = self.kfold, scoring = 'accuracy', n_jobs = -1, verbose = 0).fit(self.X_train, self.y_train)
            else: 
                xgb_clf = pipeline.fit(self.X_train, self.y_train)
                
            if self.use_igenes or self.use_visualization:
                xgb_importances = shap.TreeExplainer(xgb_clf.named_steps['xgb_clf']).shap_values(self.X_test)
            
            if self.use_igenes:
                xgb_importances_extracted = xgb_importances[0]
                xgb_importances_normalized = xgb_importances_extracted / np.max(np.abs(xgb_importances_extracted))
                xgb_hhi = ((np.abs(xgb_importances_normalized) / np.sum(xgb_importances_normalized))**2)
                
            if self.use_visualization:
                shap.summary_plot(xgb_importances, self.X_test, plot_type = "dot", show = False)
                
                plt.title("XGBoost Feature Importances", fontsize = 16)
                plt.xlabel("SHAP Value", fontsize = 14)
                plt.ylabel("Feature", fontsize = 14)
                plt.tight_layout()
                
                file_name = f"{self.cgit_file}_{datetime.now().strftime('%m-%d-%Y-%I-%M-%S-%p')}_XGB-SHAP.png"
                xgb_plot = os.path.join(self.output_dir, file_name)
                
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                
                plt.savefig(xgb_plot)
                plt.close()
                
            y_pred = xgb_clf.predict(self.X_test)
            y_proba = xgb_clf.predict_proba(self.X_test)[:, 1]
            
            xgb_accuracy = accuracy_score(self.y_test, y_pred)
            xgb_roc_auc = roc_auc_score(self.y_test, y_proba)
            xgb_f1 = f1_score(self.y_test, y_pred, average = 'weighted')
            
            if self.use_igenes:
                return xgb_clf, xgb_accuracy, xgb_roc_auc, xgb_f1, xgb_importances_normalized, xgb_hhi
            else: 
                return xgb_clf, xgb_accuracy, xgb_roc_auc, xgb_f1
            
        return None
    
    # k-Nearest Neighbors Classifier
    def knn_classifier(self: 'DiseasePrediction'):
        if self.use_knn: 
            print('k-Nearest Neighbors...')
            if self.use_normalization:
                pipeline = Pipeline([('scaling', StandardScaler()), ('knn_clf', sklearn.neighbors.KNeighborsClassifier())])
            else:
                pipeline = Pipeline([('knn_clf', sklearn.neighbors.KNeighborsClassifier())])
                
            if self.use_tuning:
                knn_parameters = {
                    "knn_clf__leaf_size": list(range(1, 50)),
                    "knn_clf__n_neighbors": list(range(1, min(len(self.X_train) - 1, 30) + 1)),
                    "knn_clf__p": [1, 2]
                }
                
                knn_clf = GridSearchCV(pipeline, param_grid = knn_parameters, cv = self.kfold, scoring = 'accuracy', n_jobs = -1, verbose = 0).fit(self.X_train, self.y_train)
            else:
                knn_clf = pipeline.fit(self.X_train, self.y_train)
                
            if self.use_igenes or self.use_visualization:
                knn_importances = shap.KernelExplainer(knn_clf.named_steps['knn_clf'].predict, shap.sample(self.X_train, 1000)).shap_values(self.X_test)
            
            if self.use_igenes:
                knn_importances_extracted = knn_importances[0]
                knn_importances_normalized = knn_importances_extracted / np.max(np.abs(knn_importances_extracted))
                knn_hhi = ((np.abs(knn_importances_normalized) / np.sum(knn_importances_normalized))**2)
            
            if self.use_visualization:
                shap.summary_plot(knn_importances, self.X_test, plot_type = "dot", show = False)
                
                plt.title("k-Nearest Neighbors Feature Importances", fontsize = 16)
                plt.xlabel("SHAP Value", fontsize = 14)
                plt.ylabel("Feature", fontsize = 14)
                plt.tight_layout()
                
                file_name = f"{self.cgit_file}_{datetime.now().strftime('%m-%d-%Y-%I-%M-%S-%p')}_kNN-SHAP.png"
                knn_plot = os.path.join(self.output_dir, file_name)
                
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                    
                plt.savefig(knn_plot)
                plt.close()
            
            y_pred = knn_clf.predict(self.X_test)
            y_proba = knn_clf.predict_proba(self.X_test)[:, 1]
            
            knn_accuracy = accuracy_score(self.y_test, y_pred)
            knn_roc_auc = roc_auc_score(self.y_test, y_proba)
            knn_f1 = f1_score(self.y_test, y_pred, average = 'weighted')
            
            if self.use_igenes:
                return knn_clf, knn_accuracy, knn_roc_auc, knn_f1, knn_importances_normalized, knn_hhi
            else: 
                return knn_clf, knn_accuracy, knn_roc_auc, knn_f1
            
        return None
    
    # Multi-Layer Perceptron Classifier
    def mlp_classifier(self: 'DiseasePrediction'):
        if self.use_mlp:
            print('Multi-Layer Perceptron...')
            if self.use_normalization:
                pipeline = Pipeline([('scaling', StandardScaler()), ('mlp_clf', MLPClassifier(random_state = self.random_state, max_iter = 3000))])
            else:
                pipeline = Pipeline([('mlp_clf', MLPClassifier(random_state = self.random_state, max_iter = 2000))])
            
            if self.use_tuning:
                mlp_parameters = {
                    'mlp_clf__hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,), (100, 100,), (100, 100, 100)],
                    'mlp_clf__activation': ['tanh', 'relu'],
                    'mlp_clf__solver': ['sgd', 'adam'],
                    'mlp_clf__alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
                    'mlp_clf__learning_rate': ['constant', 'adaptive'],
                    'mlp_clf__learning_rate_init': [0.001, 0.01, 0.1],
                    'mlp_clf__momentum': [0.1, 0.2, 0.5, 0.9],
                }
                
                mlp_clf = GridSearchCV(pipeline, param_grid = mlp_parameters, cv = self.kfold, scoring = 'accuracy', n_jobs = -1, verbose = 0).fit(self.X_train, self.y_train)
            else:
                mlp_clf = pipeline.fit(self.X_train, self.y_train)
                
            if self.use_igenes or self.use_visualization:
                mlp_importances =  shap.KernelExplainer(mlp_clf.named_steps['mlp_clf'].predict, shap.sample(self.X_train, 1000)).shap_values(self.X_test)
                
            if self.use_igenes:
                mlp_importances_extracted = mlp_importances[0]
                mlp_importances_normalized = mlp_importances_extracted / np.max(np.abs(mlp_importances_extracted))
                mlp_hhi = ((np.abs(mlp_importances_normalized) / np.sum(mlp_importances_normalized))**2)
                
            if self.use_visualization:
                shap.summary_plot(mlp_importances, self.X_test, plot_type = "dot", show = False)
                
                plt.title("Multi-Layer Perceptron Feature Importances", fontsize = 16)
                plt.xlabel("SHAP Value", fontsize = 14)
                plt.ylabel("Feature", fontsize = 14)
                plt.tight_layout()
                
                file_name = f"{self.cgit_file}_{datetime.now().strftime('%m-%d-%Y-%I-%M-%S-%p')}_MLP-SHAP.png"
                mlp_plot = os.path.join(self.output_dir, file_name)
                
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                    
                plt.savefig(mlp_plot)
                plt.close()
                
            y_pred = mlp_clf.predict(self.X_test)
            y_proba = mlp_clf.predict_proba(self.X_test)[:, 1]
            
            mlp_accuracy = accuracy_score(self.y_test, y_pred)
            mlp_roc_auc = roc_auc_score(self.y_test, y_proba)
            mlp_f1 = f1_score(self.y_test, y_pred, average = 'weighted')           
            
            if self.use_igenes:
                return mlp_clf, mlp_accuracy, mlp_roc_auc, mlp_f1, mlp_importances_normalized, mlp_hhi
            else:
                return mlp_clf, mlp_accuracy, mlp_roc_auc, mlp_f1
            
        return None
    
    # Voting Classifier
    def voting_classifier(self: 'DiseasePrediction'):
        if self.classifiers:
            print('Voting Classifier...')
            voting_clf = VotingClassifier(estimators = [(clf[0], clf[1]["Classifier"]) for clf in self.classifiers], voting = self.voting).fit(self.X_train, self.y_train)
            
            y_pred = voting_clf.predict(self.X_test)
            y_proba = voting_clf.predict_proba(self.X_test)[:, 1]
            
            voting_accuracy = accuracy_score(self.y_test, y_pred) 
            voting_roc_auc = roc_auc_score(self.y_test, y_proba) 
            voting_f1 = f1_score(self.y_test, y_pred, average = 'weighted')
            
            return voting_clf, voting_accuracy, voting_roc_auc, voting_f1
        
        return None
    
    # Execute Classifiers
    def execute_classifiers(self: 'DiseasePrediction'):
        classifiers = [
            ("Random Forest", self.rf_classifier),
            ("Support Vector Machine", self.svm_classifier),
            ("XGBoost", self.xgb_classifier),
            ("k-Nearest Neighbors", self.knn_classifier),
            ("Multi-Layer Perceptron", self.mlp_classifier)
        ]

        self.classifiers = []
        for name, classifier in classifiers:
            metrics_tuple = classifier()
            if metrics_tuple is not None:
                clf, accuracy, roc_auc, f1, *rest = metrics_tuple

                classifier_info = {
                    "Classifier": clf,
                    "Accuracy": accuracy,
                    "ROC-AUC": roc_auc,
                    "F1": f1
                }

                if len(rest) >= 2:  
                    classifier_info["Importances"] = rest[0]
                    classifier_info["HHI"] = rest[1]

                self.classifiers.append((name, classifier_info))
                
        voting_metrics_tuple = self.voting_classifier()
        if voting_metrics_tuple:
            voting_clf, voting_accuracy, voting_roc_auc, voting_f1 = voting_metrics_tuple
            self.classifiers.append(("Voting Classifier", {
                "Classifier": voting_clf,
                "Accuracy": voting_accuracy,
                "ROC-AUC": voting_roc_auc,
                "F1": voting_f1
            }))
            
    # Generate Classifier Metrics
    def classifier_metrics(self: 'DiseasePrediction'):   
        if hasattr(self, 'classifiers') and self.classifiers:
            metrics_df = pd.DataFrame({
                'Classifier': [clf[0] for clf in self.classifiers],
                'Accuracy': [clf[1]['Accuracy'] for clf in self.classifiers],
                'ROC-AUC': [clf[1]['ROC-AUC'] for clf in self.classifiers],
                'F1': [clf[1]['F1'] for clf in self.classifiers]
            })
        
            return metrics_df
    
    # Generate I-Genes Profile
    def expression_direction(self: 'DiseasePrediction', *scores):
        positive_count = sum(1 for score in scores if score > 0)
        negative_count = sum(1 for score in scores if score < 0)
        
        if positive_count > negative_count:
            return "Overexpressed"
        elif negative_count > positive_count:
            return "Underexpressed"
        else:
            return "Inconclusive"
    
    def igenes_scores(self: 'DiseasePrediction'):
        if self.use_igenes:
            print('Generating I-Genes Scores...')
            if self.classifiers:
                normalized_importances = []
                herfindahl_hirschman_indices = []
                
                for name, metrics in self.classifiers:
                    if name == "Voting Classifier":
                        continue
                    
                    if "Importances" in metrics:
                        normalized_importances.append(metrics["Importances"])
                        herfindahl_hirschman_indices.append(metrics["HHI"])
                    
                if not normalized_importances or not herfindahl_hirschman_indices:
                    return
                    
                num_classifiers = len(self.classifiers)
                hhi_weights = (np.array(herfindahl_hirschman_indices) / np.sum(herfindahl_hirschman_indices))
                initial_weights = (1 / (num_classifiers + 1))
                final_weights = (initial_weights + (initial_weights * hhi_weights))
                
                igenes_scores = np.zeros_like(normalized_importances[0])
                for weight, importance in zip(final_weights, normalized_importances):
                    igenes_scores += weight * np.abs(importance)
                                
                igenes_df = pd.DataFrame({
                    'Feature': self.X_train.columns,
                    'I-Genes Score': igenes_scores,
                })
                
                normalized_importances_list = [metrics["Importances"] for _, metrics in self.classifiers if "Importances" in metrics]
                transposed_importances_list = list(zip(*normalized_importances_list))
                directions = [self.expression_direction(*scores) for scores in transposed_importances_list]
                igenes_df['Direction'] = directions
                
                igenes_df['I-Genes Rankings'] = igenes_df['I-Genes Score'].rank(ascending=False).astype(int)
                igenes_df = igenes_df.sort_values(by = 'I-Genes Rankings')
                
                return igenes_df
    
# Execute DiseasePrediction Class
def main():
    print("IntelliGenes Disease Prediction and I-Genes Scores...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--cgit_file', required = True)
    parser.add_argument('-f', '--features_file', required = True)
    parser.add_argument('-o', '--output_dir', required = True)
    parser.add_argument('--voting', type = str, default = 'soft')
    parser.add_argument('--random_state', type = int, default = 42)
    parser.add_argument('--test_size', type = float, default = 0.3)
    parser.add_argument('--n_splits', type = int, default = 5)
    parser.add_argument('--no_rf', action = 'store_true')
    parser.add_argument('--no_svm', action = 'store_true')
    parser.add_argument('--no_xgb', action = 'store_true')
    parser.add_argument('--no_knn', action = 'store_true')
    parser.add_argument('--no_mlp', action = 'store_true')
    parser.add_argument('--tune', action = 'store_true')
    parser.add_argument('--normalize', action = 'store_true')
    parser.add_argument('--no_igenes', action = 'store_true')
    parser.add_argument('--no_visualizations', action = 'store_true')


    args = parser.parse_args()

    pipeline = DiseasePrediction(cgit_file = args.cgit_file,
                                features_file = args.features_file,
                                output_dir = args.output_dir,
                                voting = args.voting,
                                random_state = args.random_state,
                                test_size = args.test_size,
                                n_splits = args.n_splits,
                                use_rf = not args.no_rf,
                                use_svm = not args.no_svm,
                                use_xgb = not args.no_xgb,
                                use_knn = not args.no_knn,
                                use_mlp = not args.no_mlp,
                                use_tuning = args.tune,
                                use_normalization = args.normalize,
                                use_igenes = not args.no_igenes,
                                use_visualization = not args.no_visualizations)
    
    pipeline.execute_classifiers()
    metrics_df = pipeline.classifier_metrics()
    igenes_df = pipeline.igenes_scores()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    metrics_name = f"{args.cgit_file}_{datetime.now().strftime('%m-%d-%Y-%I-%M-%S-%p')}_Classifier-Metrics.csv"
    metrics_file = os.path.join(args.output_dir, metrics_name)
    
    metrics_df.to_csv(metrics_file, index = False)
    print("\n Clasifier Metrics:", metrics_file, "\n")
    
    if args.no_igenes is False: 
        igenes_name = f"{args.cgit_file}_{datetime.now().strftime('%m-%d-%Y-%I-%M-%S-%p')}_I-Genes-Score.csv"
        igenes_file = os.path.join(args.output_dir, igenes_name)
        
        igenes_df.to_csv(igenes_file, index = False)
        print("\n I-Genes Scores:", igenes_file, "\n")
                       
if __name__ == '__main__':
    main()