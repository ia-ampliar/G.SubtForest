"""
Módulo para avaliação de modelos e geração de métricas.
Unifica as funções dos notebooks etapa2 e etapa3.
"""
import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import shap

class ModelEvaluator:
    """
    Classe para avaliação de modelos Random Forest com SHAP e métricas.
    """
    
    def __init__(self, output_dir='outputs'):
        """
        Inicializa o ModelEvaluator com o diretório de saída.
        
        Args:
            output_dir (str): Caminho para o diretório de saída
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def create_result_table(self, y_true, y_pred, y_prob, rf_model, fold, sample_ids, path=None):
        """
        Cria tabela de resultados para um fold específico.
        Função unificada dos notebooks etapa2 e etapa3.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            y_prob: Probabilidades preditas
            rf_model: Modelo Random Forest
            fold: Número do fold
            sample_ids: IDs das amostras
            path: Caminho para salvar (se None, usa self.output_dir)
        """
        if path is None:
            path = self.output_dir
            
        # Verifica se y_prob é uma lista de dicionários (etapa3) ou array numpy (etapa2)
        if isinstance(y_prob, list) and len(y_prob) > 0 and isinstance(y_prob[0], dict):
            # Formato etapa3 (lista de dicts)
            predicted_probability = [max(prob.values()) for prob in y_prob]
            probability_vector = y_prob
        else:
            # Formato etapa2 (array numpy)
            predicted_probability = [max(prob) for prob in y_prob]
            probability_vector = list(y_prob)
        
        results_df = pd.DataFrame({
            'Sample ID': sample_ids,
            'true_label': y_true.values if hasattr(y_true, 'values') else y_true,
            'predicted_label': y_pred,
            'predicted_probability': predicted_probability,
            'probability_vector': probability_vector
        })

        # Criar diretório se não existir (usar nomes consistentes)
        results_dir = os.path.join(path, 'kfold_random_forest_results')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        results_df.to_csv(os.path.join(results_dir, f'rf_test_df_fold{fold}_results.csv'), index=False)
    
    def average_classification_report(self, reports):
        """
        Calcula a média dos classification reports.
        Função do notebook etapa3.
        
        Args:
            reports: Lista de classification reports
            
        Returns:
            dict: Relatório médio
        """
        # Aceita lista de dicts gerados por sklearn.classification_report(output_dict=True)
        n_reports = len(reports)
        if n_reports == 0:
            return {}

        metrics_sum = defaultdict(lambda: defaultdict(float))
        accuracy_sum = 0.0

        for report in reports:
            # report deve ser um dict
            for label, metrics in report.items():
                if label == 'accuracy':
                    accuracy_sum += float(metrics)
                    continue
                # métricas por classe/macros
                for metric_name, value in metrics.items():
                    # alguns valores podem ser NaN; trate como 0
                    try:
                        metrics_sum[label][metric_name] += float(value)
                    except Exception:
                        pass

        # Média
        avg_report = {}
        for label, metrics in metrics_sum.items():
            avg_report[label] = {}
            for metric_name, value in metrics.items():
                avg_report[label][metric_name] = value / n_reports
        avg_report['accuracy'] = accuracy_sum / n_reports
        return avg_report

    def format_classification_report(self, avg_report, digits=2):
        """
        Formata o relatório como texto.
        Função do notebook etapa3.
        
        Args:
            avg_report: Relatório médio
            digits: Número de casas decimais
            
        Returns:
            str: Relatório formatado
        """
        classes = ['cin', 'ebv', 'gs', 'msi']
        metrics = ['precision', 'recall', 'f1-score', 'support']

        headers = ['precision', 'recall', 'f1-score', 'support']
        lines = []

        # Cabeçalho
        lines.append(f"{'':14}" + "  ".join(f"{h:>10}" for h in headers))

        # Linhas por classe
        for cls in classes:
            if cls not in avg_report:
                continue
            values = [
                f"{avg_report[cls][metric]:.{digits}f}" if metric != 'support' else f"{int(avg_report[cls][metric])}"
                for metric in metrics
            ]
            lines.append(f"{cls:<14}" + "  ".join(f"{v:>10}" for v in values))

        # Linha em branco
        lines.append("")

        # Macro avg
        if 'macro avg' in avg_report:
            macro_values = [
                f"{avg_report['macro avg'][metric]:.{digits}f}" if metric != 'support' else f"{int(avg_report['macro avg'][metric])}"
                for metric in metrics
            ]
            lines.append(f"{'macro avg':<14}" + "  ".join(f"{v:>10}" for v in macro_values))

        # Weighted avg
        if 'weighted avg' in avg_report:
            weighted_values = [
                f"{avg_report['weighted avg'][metric]:.{digits}f}" if metric != 'support' else f"{int(avg_report['weighted avg'][metric])}"
                for metric in metrics
            ]
            lines.append(f"{'weighted avg':<14}" + "  ".join(f"{v:>10}" for v in weighted_values))

        # Accuracy
        if 'accuracy' in avg_report:
            accuracy = f"{avg_report['accuracy']:.{digits}f}"
            lines.append(f"{'accuracy':<14}" + f"{'':>12}{'':>12}{accuracy:>12}{'':>12}")

        return "\n".join(lines)
    
    def rf_kfold_exe(self, X_train_val, y_train_val, train_val_df, mode='validation'):
        """
        Executa k-fold com Random Forest, otimização de hiperparâmetros e SHAP.
        Função unificada dos notebooks etapa2 e etapa3.
        
        Args:
            X_train_val: Features de treino/validação
            y_train_val: Labels de treino/validação
            train_val_df: DataFrame com Sample ID para validação
            mode: 'validation' para etapa2 ou 'test' para etapa3
            
        Returns:
            dict: Dicionário com resultados
        """
        # Parâmetros para otimização do Random Forest
        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        # K-fold
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        # Para coletar resultados
        shap_values_folds = []
        models = []
        best_params_per_fold = []
        val_indices_per_fold = []
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        reports = []

        # Iniciar RandomizedSearchCV
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_val, y_train_val)):
            X_tr, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
            y_tr, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

            # Salvar índices de validação
            val_indices_per_fold.append(val_idx)

            # Instanciar modelo
            rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

            # Otimizar hiperparâmetros
            random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                              n_iter=10, cv=3, n_jobs=-1, random_state=42)
            random_search.fit(X_tr, y_tr)

            # Salvar melhores parâmetros
            best_params_per_fold.append(random_search.best_params_)
            best_rf_model = random_search.best_estimator_
            models.append(best_rf_model)

            # Previsões
            y_val_pred = best_rf_model.predict(X_val)
            y_val_prob = best_rf_model.predict_proba(X_val)

            # Tabela de resultados
            sample_ids = train_val_df['Sample ID'].iloc[val_idx]
            self.create_result_table(y_val, y_val_pred, y_val_prob, best_rf_model, fold_idx, sample_ids)

            # Classification report
            report = classification_report(y_val, y_val_pred, target_names=best_rf_model.classes_, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose().round(2)
            report_df.to_csv(f"{self.output_dir}/cr_val_fold{fold_idx}.csv", index=False)
            reports.append(report)
            print(f"\nClassification Report fold{fold_idx}:")
            print(report_df)

            # SHAP para o fold (apenas no modo validation)
            if mode == 'validation':
                explainer = shap.TreeExplainer(best_rf_model)
                shap_vals = explainer.shap_values(X_val, check_additivity=False)
                shap_values_folds.append(shap_vals)

            # Calcular métricas
            accuracies.append(accuracy_score(y_val, y_val_pred))
            f1_scores.append(f1_score(y_val, y_val_pred, average='macro'))
            precisions.append(precision_score(y_val, y_val_pred, average='macro', zero_division=0))
            recalls.append(recall_score(y_val, y_val_pred, average='macro', zero_division=0))

        # Calcular métricas médias
        print("Resultados dos modelos dos 10-fold com Hyperp. Otimiz. no conjunto de validação")
        print(f"Média da Acurácia: {np.mean(accuracies):.2f} +- {np.std(accuracies):.2f}")
        print(f"Média do F1-Score: {np.mean(f1_scores):.2f} +- {np.std(f1_scores):.2f}")
        print(f"Média da Precisão: {np.mean(precisions):.2f} +- {np.std(precisions):.2f}")
        print(f"Média do Recall: {np.mean(recalls):.2f} +- {np.std(recalls):.2f}")

        # Salvar métricas médias
        metrics_df = pd.DataFrame({
            'Acurácia Média': [np.mean(accuracies)],
            'Acurácia Std (+-)': [np.std(accuracies)],
            'F1-Score Médio': [np.mean(f1_scores)],
            'F1-Score Std (+-)': [np.std(f1_scores)],
            'Precisão Média': [np.mean(precisions)],
            'Precisão Std (+-)': [np.std(precisions)],
            'Recall Médio': [np.mean(recalls)],
            'Recall Std': [np.std(recalls)]
        })
        metrics_df.to_csv(f'{self.output_dir}/metricas_media_kfold_hyper.csv', index=False)

        # Processar SHAP apenas no modo validation
        shap_values_folds_mean_per_class = {}
        if mode == 'validation' and shap_values_folds:
            # Calcular SHAP médio por subtipo
            classes = list(models[0].classes_)
            shap_values_folds_mean_per_class = {cls: [] for cls in classes}

            for shap_vals in shap_values_folds:
                for cls_idx, cls in enumerate(classes):
                    shap_mean_cls = np.abs(shap_vals[cls_idx]).mean(axis=0)
                    shap_values_folds_mean_per_class[cls].append(shap_mean_cls)

            # Gerar tabela de importância SHAP por subtipo
            shap_importance_per_class = []
            for cls in classes:
                mean_shap_cls = np.mean(shap_values_folds_mean_per_class[cls], axis=0)
                top_indices = np.argsort(mean_shap_cls)[::-1]
                class_df = pd.DataFrame({
                    'Gene': X_train_val.columns[top_indices],
                    f'SHAP Importance ({cls})': mean_shap_cls[top_indices]
                })
                class_df['Subtipo'] = cls
                shap_importance_per_class.append(class_df)

            # Concatenar e salvar tabela de importância por subtipo
            shap_importance_df = pd.concat(shap_importance_per_class, ignore_index=True)
            shap_importance_df.to_csv(f'{self.output_dir}/gene_importance_shap_per_subtype.csv', index=False)
            print("\nTabela de Importância SHAP por Subtipo salva em 'gene_importance_shap_per_subtype.csv'")
            print(shap_importance_df.head(20))

            # Tabela de importância SHAP geral
            shap_values_folds_mean = [np.mean([np.abs(s).mean(axis=0) for s in shap_vals], axis=0) for shap_vals in shap_values_folds]
            mean_shap_folds = np.mean(shap_values_folds_mean, axis=0)
            top_indices = np.argsort(mean_shap_folds)[::-1]
            gene_importance_df = pd.DataFrame({
                'Gene': X_train_val.columns[top_indices],
                'SHAP Importance': mean_shap_folds[top_indices]
            })
            gene_importance_df.to_csv(f'{self.output_dir}/gene_importance_shap.csv', index=False)
            print("\nTabela de Importância SHAP Geral salva em 'gene_importance_shap.csv'")

            # Salvar índices de validação
            val_indices_df = pd.DataFrame({
                'Fold': list(range(10)),
                'Val Indices': [list(indices) for indices in val_indices_per_fold]
            })
            val_indices_df.to_csv(f'{self.output_dir}/val_indices_per_fold.csv', index=False)
            print("\nÍndices de Validação salvos em 'val_indices_per_fold.csv'")

        results_dict = {
            'shap_values_folds': shap_values_folds,
            'models': models,
            'best_params_per_fold': best_params_per_fold,
            'val_indices_per_fold': val_indices_per_fold,
            'accuracies': accuracies,
            'f1_scores': f1_scores,
            'precisions': precisions,
            'recalls': recalls,
            'reports': reports,
            'shap_values_folds_mean_per_class': shap_values_folds_mean_per_class,
            'metrics_df': metrics_df
        }

        return results_dict
    
    def test_rf_kfold_exe(self, X_train_val, y_train_val, X_test, y_test, test_df, path=None):
        """
        Executa k-fold para teste com painéis genéticos.
        Função do notebook etapa3.
        
        Args:
            X_train_val: Features de treino/validação
            y_train_val: Labels de treino/validação
            X_test: Features de teste
            y_test: Labels de teste
            test_df: DataFrame de teste
            path: Caminho para salvar resultados
            
        Returns:
            dict: Dicionário com resultados
        """
        if path is None:
            path = self.output_dir
            
        # Parâmetros para otimização do Random Forest
        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        # K-fold
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        # Para coletar resultados
        models = []
        best_params_per_fold = []
        val_indices_per_fold = []
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        reports = []

        # Iniciar RandomizedSearchCV
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_val, y_train_val)):
            X_tr, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
            y_tr, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

            # Salvar índices de validação
            val_indices_per_fold.append(val_idx)

            # Instanciar modelo
            rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

            # Otimizar hiperparâmetros
            random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                              n_iter=10, cv=3, n_jobs=-1, random_state=42)
            random_search.fit(X_tr, y_tr)

            # Salvar melhores parâmetros
            best_params_per_fold.append(random_search.best_params_)
            best_rf_model = random_search.best_estimator_
            models.append(best_rf_model)

            # Previsões no conjunto de teste
            y_test_pred = best_rf_model.predict(X_test)
            y_test_prob = best_rf_model.predict_proba(X_test)
            
            # Converter probabilidades para formato de dicionário
            columns = best_rf_model.classes_
            y_test_prob = pd.DataFrame(y_test_prob, columns=columns).round(4).to_dict(orient='records')

            # Tabela de resultados
            sample_ids = test_df['Sample ID']
            self.create_result_table(y_test, y_test_pred, y_test_prob, best_rf_model, fold_idx, sample_ids, path)

            # Classification report
            report = classification_report(y_test, y_test_pred, target_names=best_rf_model.classes_, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose().round(2)
            reports.append(report)
            
            # Salvar classification report por fold
            cr_dir = os.path.join(path, 'test_classification_report_per_fold')
            if not os.path.isdir(cr_dir):
                os.makedirs(cr_dir)
            report_df.to_csv(f'{cr_dir}/cr_test_fold{fold_idx}.csv')

            # Calcular métricas
            accuracies.append(accuracy_score(y_test, y_test_pred))
            f1_scores.append(f1_score(y_test, y_test_pred, average='macro'))
            precisions.append(precision_score(y_test, y_test_pred, average='macro', zero_division=0))
            recalls.append(recall_score(y_test, y_test_pred, average='macro', zero_division=0))

        # Calcular métricas médias
        print("Resultados dos modelos dos 10-fold com Hyperp. Otimiz. no conjunto de teste")
        print(f"Média da Acurácia: {np.mean(accuracies):.2f} +- {np.std(accuracies):.2f}")
        print(f"Média do F1-Score: {np.mean(f1_scores):.2f} +- {np.std(f1_scores):.2f}")
        print(f"Média da Precisão: {np.mean(precisions):.2f} +- {np.std(precisions):.2f}")
        print(f"Média do Recall: {np.mean(recalls):.2f} +- {np.std(recalls):.2f}")

        # Calcular o relatório médio
        avg_report = self.average_classification_report(reports)
        print("Classification Report (Average):")
        print(self.format_classification_report(avg_report))

        # Salvar métricas médias
        metrics_df = pd.DataFrame({
            'Acurácia Média': [np.mean(accuracies)],
            'Acurácia Std (+-)': [np.std(accuracies)],
            'F1-Score Médio': [np.mean(f1_scores)],
            'F1-Score Std (+-)': [np.std(f1_scores)],
            'Precisão Média': [np.mean(precisions)],
            'Precisão Std (+-)': [np.std(precisions)],
            'Recall Médio': [np.mean(recalls)],
            'Recall Std': [np.std(recalls)]
        })
        metrics_df.to_csv(f'{path}/metricas_media_kfold_hyper.csv', index=False)

        results_dict = {
            'models': models,
            'best_params_per_fold': best_params_per_fold,
            'val_indices_per_fold': val_indices_per_fold,
            'accuracies': accuracies,
            'f1_scores': f1_scores,
            'precisions': precisions,
            'recalls': recalls,
            'reports': reports,
            'metrics_df': metrics_df
        }

        return results_dict
    
    def metrics_from_json(self, df_final, panels, panel_name, path=None, test_list=None):
        """
        Gera métricas para um painel específico a partir de arquivo JSON.
        Função do notebook etapa3.
        
        Args:
            df_final: DataFrame completo
            panels: Dicionário com painéis genéticos
            panel_name: Nome do painel
            path: Caminho para salvar resultados
        """
        if path is None:
            path = self.output_dir

        if test_list is None:
            raise ValueError("É necessário fornecer 'test_list' com os IDs de amostra para o conjunto de teste.")

        from .preprocessor import Preprocessor
        preprocessor = Preprocessor()
        
        panel_genes = panels[panel_name] 
        # Filtra apenas genes que existem na tabela final
        panel_genes_existing = [g for g in panel_genes if g in df_final.columns]
        df_p1 = df_final[['Sample ID', 'Subtype'] + panel_genes_existing]  # Subset do dataset
        
        print('Dividindo dataset por painel')
        train_val_df, test_df = preprocessor.create_data_set(df_p1, test_list)
        
        print('Separando X e y')
        X_train_val, y_train_val, X_test, y_test, sample_ids = preprocessor.X_y_df_split(train_val_df, test_df)
        
        print('Gerando resultados')
        result_dict = self.test_rf_kfold_exe(X_train_val, y_train_val, X_test, y_test, test_df, path)
        
        return result_dict