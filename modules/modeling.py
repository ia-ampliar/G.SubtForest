import os
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score
)


class RandomForestKFoldRunner:
    """
    Executa treinamento e validação cruzada (K-Fold) de Random Forest com
    otimização de hiperparâmetros, geração de relatórios, métricas e SHAP.

    Args:
        n_splits (int): Número de folds (default: 10)
        random_state (int): Semente para reprodutibilidade (default: 42)
        output_dir (str): Diretório para salvar resultados (default: 'outputs')
        n_iter_search (int): Número de iterações do RandomizedSearchCV (default: 10)
    """

    def __init__(self, n_splits=10, random_state=42, output_dir="outputs", n_iter_search=10):
        self.n_splits = n_splits
        self.random_state = random_state
        self.output_dir = output_dir
        self.n_iter_search = n_iter_search

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "kfold_random_forest_results"), exist_ok=True)

        # Parâmetros padrão para Random Search
        self.param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

    # -------------------------------------------------------------------------
    def create_result_table(self, y_true, y_pred, y_prob, rf_model, fold, sample_ids):
        """Cria e salva a tabela de resultados de um fold."""
        results_df = pd.DataFrame({
            'Sample ID': sample_ids,
            'true_label': y_true.values,
            'predicted_label': y_pred,
            'predicted_probability': [max(prob) for prob in y_prob],
            'probability_vector': list(y_prob)
        })

        result_path = os.path.join(
            self.output_dir, "kfold_random_forest_results",
            f"rf_test_df_fold{fold}_results.csv"
        )
        results_df.to_csv(result_path, index=False)
        print(f"Resultados do fold {fold} salvos em: {result_path}")

    # -------------------------------------------------------------------------
    def run(self, X, y, data_df):
        """
        Executa o K-Fold com Random Forest e retorna os resultados.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Labels.
            data_df (pd.DataFrame): DataFrame contendo 'Sample ID'.
        """
        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        models, best_params_per_fold = [], []
        val_indices_per_fold = []
        shap_values_folds = []
        accuracies, f1_scores, precisions, recalls = [], [], [], []
        reports = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"\n🚀 Executando Fold {fold_idx + 1}/{self.n_splits}")

            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            val_indices_per_fold.append(val_idx)

            # Modelo base
            rf = RandomForestClassifier(
                random_state=self.random_state, n_jobs=-1, class_weight='balanced'
            )

            # Otimização de hiperparâmetros
            random_search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=self.param_dist,
                n_iter=self.n_iter_search,
                cv=3,
                n_jobs=-1,
                random_state=self.random_state
            )
            random_search.fit(X_tr, y_tr)

            best_rf_model = random_search.best_estimator_
            best_params_per_fold.append(random_search.best_params_)
            models.append(best_rf_model)

            # Previsões
            y_val_pred = best_rf_model.predict(X_val)
            y_val_prob = best_rf_model.predict_proba(X_val)

            # Tabela de resultados
            sample_ids = data_df['Sample ID'].iloc[val_idx]
            self.create_result_table(y_val, y_val_pred, y_val_prob, best_rf_model, fold_idx, sample_ids)

            # Classification Report
            report = classification_report(
                y_val, y_val_pred,
                target_names=best_rf_model.classes_,
                output_dict=True, zero_division=0
            )
            report_df = pd.DataFrame(report).transpose().round(2)
            report_path = os.path.join(self.output_dir, f"cr_val_fold{fold_idx}.csv")
            report_df.to_csv(report_path, index=False)
            reports.append(report_df)

            print(f"\nClassification Report Fold {fold_idx}:")
            print(report_df)

            # SHAP
            explainer = shap.TreeExplainer(best_rf_model)
            shap_vals = explainer.shap_values(X_val, check_additivity=False)
            shap_values_folds.append(shap_vals)

            # Métricas
            accuracies.append(accuracy_score(y_val, y_val_pred))
            f1_scores.append(f1_score(y_val, y_val_pred, average='macro'))
            precisions.append(precision_score(y_val, y_val_pred, average='macro', zero_division=0))
            recalls.append(recall_score(y_val, y_val_pred, average='macro', zero_division=0))

        # ---------------------------------------------------------------------
        # Resultados agregados
        metrics_df = self._save_aggregate_metrics(accuracies, f1_scores, precisions, recalls)
        shap_values_folds_mean_per_class = self._save_shap_importances(models, shap_values_folds, X)

        # Índices de validação
        val_indices_df = pd.DataFrame({
            'Fold': list(range(self.n_splits)),
            'Val Indices': [list(indices) for indices in val_indices_per_fold]
        })
        val_indices_df.to_csv(os.path.join(self.output_dir, 'val_indices_per_fold.csv'), index=False)

        print("\n✅ Execução finalizada com sucesso!")

        return {
            'models': models,
            'best_params_per_fold': best_params_per_fold,
            'val_indices_per_fold': val_indices_per_fold,
            'accuracies': accuracies,
            'f1_scores': f1_scores,
            'precisions': precisions,
            'recalls': recalls,
            'reports': reports,
            'shap_values_folds': shap_values_folds,
            'shap_values_folds_mean_per_class': shap_values_folds_mean_per_class,
            'metrics_df': metrics_df,
            'val_indices_df': val_indices_df
        }

    # -------------------------------------------------------------------------
    def _save_aggregate_metrics(self, accuracies, f1_scores, precisions, recalls):
        """Calcula e salva métricas médias do K-Fold."""
        print("\n📊 Resultados médios (validação cruzada):")
        print(f"Acurácia média: {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")
        print(f"F1-score médio: {np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}")
        print(f"Precisão média: {np.mean(precisions):.2f} ± {np.std(precisions):.2f}")
        print(f"Recall médio: {np.mean(recalls):.2f} ± {np.std(recalls):.2f}")

        metrics_df = pd.DataFrame({
            'Acurácia Média': [np.mean(accuracies)],
            'Acurácia Std': [np.std(accuracies)],
            'F1-Score Médio': [np.mean(f1_scores)],
            'F1-Score Std': [np.std(f1_scores)],
            'Precisão Média': [np.mean(precisions)],
            'Precisão Std': [np.std(precisions)],
            'Recall Médio': [np.mean(recalls)],
            'Recall Std': [np.std(recalls)]
        })

        metrics_path = os.path.join(self.output_dir, 'metricas_media_kfold_hyper.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Métricas médias salvas em: {metrics_path}")
        return metrics_df

    # -------------------------------------------------------------------------
    def _save_shap_importances(self, models, shap_values_folds, X):
        """Calcula e salva tabelas de importância SHAP (geral e por subtipo)."""
        classes = list(models[0].classes_)
        shap_values_folds_mean_per_class = {cls: [] for cls in classes}

        for shap_vals in shap_values_folds:
            for cls_idx, cls in enumerate(classes):
                shap_mean_cls = np.abs(shap_vals[cls_idx]).mean(axis=0)
                shap_values_folds_mean_per_class[cls].append(shap_mean_cls)

        # Importância SHAP por subtipo
        shap_importance_per_class = []
        for cls in classes:
            mean_shap_cls = np.mean(shap_values_folds_mean_per_class[cls], axis=0)
            top_indices = np.argsort(mean_shap_cls)[::-1]
            class_df = pd.DataFrame({
                'Gene': X.columns[top_indices],
                f'SHAP Importance ({cls})': mean_shap_cls[top_indices],
                'Subtipo': cls
            })
            shap_importance_per_class.append(class_df)

        shap_importance_df = pd.concat(shap_importance_per_class, ignore_index=True)
        shap_importance_df.to_csv(os.path.join(self.output_dir, 'gene_importance_shap_per_subtype.csv'), index=False)

        # Importância SHAP geral
        shap_values_folds_mean = [
            np.mean([np.abs(s).mean(axis=0) for s in shap_vals], axis=0)
            for shap_vals in shap_values_folds
        ]
        mean_shap_folds = np.mean(shap_values_folds_mean, axis=0)
        top_indices = np.argsort(mean_shap_folds)[::-1]

        gene_importance_df = pd.DataFrame({
            'Gene': X.columns[top_indices],
            'SHAP Importance': mean_shap_folds[top_indices]
        })
        gene_importance_df.to_csv(os.path.join(self.output_dir, 'gene_importance_shap.csv'), index=False)

        print("\nSHAP importances salvas com sucesso.")
        return shap_values_folds_mean_per_class
