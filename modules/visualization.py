"""
Módulo para visualização de dados e resultados.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import shap

class Visualization:
    """
    Classe para visualização de dados e resultados.
    """
    
    def __init__(self, output_dir='outputs'):
        """
        Inicializa o Visualization com o diretório de saída.
        
        Args:
            output_dir (str): Caminho para o diretório de saída
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Configurações padrão para visualizações
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette('viridis')
    
    def plot_subtype_distribution(self, clinical_data, title='Distribuição de Subtipos Moleculares', figsize=(10, 6)):
        """
        Plota a distribuição de subtipos moleculares.
        
        Args:
            clinical_data (pandas.DataFrame): DataFrame com dados clínicos
            title (str): Título do gráfico
            figsize (tuple): Tamanho da figura
            
        Returns:
            matplotlib.figure.Figure: Figura gerada
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Contar subtipos
        subtype_counts = clinical_data['Subtype'].value_counts()
        
        # Calcular percentagens
        total = len(clinical_data)
        percentages = (subtype_counts / total * 100).round(1)
        
        # Criar rótulos com contagem e percentagem
        labels = [f"{idx} ({count}, {pct}%)" for idx, count, pct in 
                 zip(subtype_counts.index, subtype_counts, percentages)]
        
        # Plotar gráfico de barras
        sns.barplot(x=subtype_counts.index, y=subtype_counts.values, ax=ax)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylabel('Número de Amostras')
        ax.set_xlabel('Subtipo Molecular')
        
        # Adicionar valores no topo das barras
        for i, v in enumerate(subtype_counts.values):
            ax.text(i, v + 5, str(v), ha='center')
        
        plt.tight_layout()
        return fig
    
    def plot_gene_frequency(self, gene_matrix, top_n=20, title='Frequência dos Genes Mais Comuns', figsize=(12, 8)):
        """
        Plota a frequência dos genes mais comuns.
        
        Args:
            gene_matrix (pandas.DataFrame): Matriz de genes
            top_n (int): Número de genes a mostrar
            title (str): Título do gráfico
            figsize (tuple): Tamanho da figura
            
        Returns:
            matplotlib.figure.Figure: Figura gerada
        """
        # Selecionar colunas de genes (excluindo Sample ID e Subtype)
        gene_cols = [col for col in gene_matrix.columns if col not in ['Sample ID', 'Subtype']]
        
        # Calcular frequência de cada gene
        gene_freq = gene_matrix[gene_cols].mean().sort_values(ascending=False)
        
        # Selecionar os top_n genes
        top_genes = gene_freq.head(top_n)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plotar gráfico de barras
        sns.barplot(x=top_genes.values, y=top_genes.index, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Frequência')
        ax.set_ylabel('Gene')
        
        # Adicionar valores nas barras
        for i, v in enumerate(top_genes.values):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center')
        
        plt.tight_layout()
        return fig
    
    def plot_gene_subtype_heatmap(self, gene_matrix, top_n=20, title='Heatmap de Genes por Subtipo', figsize=(14, 10)):
        """
        Plota um heatmap de genes por subtipo.
        
        Args:
            gene_matrix (pandas.DataFrame): Matriz de genes
            top_n (int): Número de genes a mostrar
            title (str): Título do gráfico
            figsize (tuple): Tamanho da figura
            
        Returns:
            matplotlib.figure.Figure: Figura gerada
        """
        # Selecionar colunas de genes (excluindo Sample ID e Subtype)
        gene_cols = [col for col in gene_matrix.columns if col not in ['Sample ID', 'Subtype']]
        
        # Calcular frequência de cada gene por subtipo
        subtype_gene_freq = gene_matrix.groupby('Subtype')[gene_cols].mean()
        
        # Calcular frequência média total para ordenar genes
        gene_freq_total = gene_matrix[gene_cols].mean().sort_values(ascending=False)
        
        # Selecionar os top_n genes
        top_genes = gene_freq_total.head(top_n).index.tolist()
        
        # Criar figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plotar heatmap
        sns.heatmap(subtype_gene_freq[top_genes], annot=True, fmt='.2f', cmap='viridis', ax=ax)
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_importance, top_n=20, title='Importância das Features', figsize=(12, 8)):
        """
        Plota a importância das features.
        
        Args:
            feature_importance (pandas.DataFrame): DataFrame com importância das features
            top_n (int): Número de features a mostrar
            title (str): Título do gráfico
            figsize (tuple): Tamanho da figura
            
        Returns:
            matplotlib.figure.Figure: Figura gerada
        """
        # Selecionar as top_n features
        top_features = feature_importance.head(top_n)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plotar gráfico de barras
        sns.barplot(x=top_features['importance'], y=top_features['feature'], ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Importância')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        return fig
    
    def plot_shap_summary(self, shap_values, X, max_display=20, title='SHAP Summary Plot', figsize=(12, 10)):
        """
        Plota um resumo dos valores SHAP.
        
        Args:
            shap_values (shap.Explanation): Valores SHAP
            X (pandas.DataFrame): Features
            max_display (int): Número máximo de features a mostrar
            title (str): Título do gráfico
            figsize (tuple): Tamanho da figura
            
        Returns:
            matplotlib.figure.Figure: Figura gerada
        """
        # Criar figura
        plt.figure(figsize=figsize)
        
        # Plotar resumo SHAP
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        plt.title(title)
        
        fig = plt.gcf()
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, classes=None, title='Matriz de Confusão', figsize=(8, 6)):
        """
        Plota uma matriz de confusão.
        
        Args:
            y_true (array-like): Valores verdadeiros
            y_pred (array-like): Valores preditos
            classes (list, optional): Lista de classes
            title (str): Título do gráfico
            figsize (tuple): Tamanho da figura
            
        Returns:
            matplotlib.figure.Figure: Figura gerada
        """
        from sklearn.metrics import confusion_matrix
        
        # Calcular matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalizar
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Criar figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plotar heatmap
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', ax=ax)
        
        # Configurar eixos
        if classes is not None:
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
        
        ax.set_xlabel('Predito')
        ax.set_ylabel('Verdadeiro')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_comparison(self, metrics_df, title='Comparação de Métricas', figsize=(10, 6)):
        """
        Plota uma comparação de métricas para diferentes modelos.
        
        Args:
            metrics_df (pandas.DataFrame): DataFrame com métricas
            title (str): Título do gráfico
            figsize (tuple): Tamanho da figura
            
        Returns:
            matplotlib.figure.Figure: Figura gerada
        """
        # Criar figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plotar gráfico de barras
        metrics_df.plot(kind='bar', ax=ax)
        
        ax.set_title(title)
        ax.set_ylabel('Valor')
        ax.set_xlabel('Modelo')
        ax.legend(loc='best')
        
        plt.tight_layout()
        return fig
    
    def plot_shap_values_per_subtype(self, shap_values, X, y, subtype, max_display=10, title=None, figsize=(12, 8)):
        """
        Plota valores SHAP para um subtipo específico.
        
        Args:
            shap_values (shap.Explanation): Valores SHAP
            X (pandas.DataFrame): Features
            y (pandas.Series): Target
            subtype (str): Subtipo a analisar
            max_display (int): Número máximo de features a mostrar
            title (str, optional): Título do gráfico
            figsize (tuple): Tamanho da figura
            
        Returns:
            matplotlib.figure.Figure: Figura gerada
        """
        # Filtrar amostras do subtipo
        mask = y == subtype
        X_subtype = X[mask]
        shap_values_subtype = shap.Explanation(
            values=shap_values.values[mask],
            base_values=shap_values.base_values[mask],
            data=X_subtype.values,
            feature_names=X.columns
        )
        
        # Criar figura
        plt.figure(figsize=figsize)
        
        # Plotar resumo SHAP
        shap.summary_plot(shap_values_subtype, X_subtype, max_display=max_display, show=False)
        
        if title is None:
            title = f'SHAP Values para Subtipo {subtype}'
        plt.title(title)
        
        fig = plt.gcf()
        plt.tight_layout()
        return fig
    
    def plot_shap_values_comparison(self, shap_values, X, y, subtypes=None, max_display=10, figsize=(15, 10)):
        """
        Plota uma comparação de valores SHAP entre diferentes subtipos.
        
        Args:
            shap_values (shap.Explanation): Valores SHAP
            X (pandas.DataFrame): Features
            y (pandas.Series): Target
            subtypes (list, optional): Lista de subtipos a comparar
            max_display (int): Número máximo de features a mostrar
            figsize (tuple): Tamanho da figura
            
        Returns:
            matplotlib.figure.Figure: Figura gerada
        """
        if subtypes is None:
            subtypes = y.unique()
        
        # Criar figura com subplots
        fig, axes = plt.subplots(len(subtypes), 1, figsize=figsize)
        
        # Para cada subtipo
        for i, subtype in enumerate(subtypes):
            # Filtrar amostras do subtipo
            mask = y == subtype
            X_subtype = X[mask]
            shap_values_subtype = shap.Explanation(
                values=shap_values.values[mask],
                base_values=shap_values.base_values[mask],
                data=X_subtype.values,
                feature_names=X.columns
            )
            
            # Plotar resumo SHAP
            plt.sca(axes[i])
            shap.summary_plot(shap_values_subtype, X_subtype, max_display=max_display, show=False)
            plt.title(f'SHAP Values para Subtipo {subtype}')
        
        plt.tight_layout()
        return fig
    
    def plot_shap_dependence(self, shap_values, X, feature, interaction_feature=None, title=None, figsize=(10, 6)):
        """
        Plota a dependência SHAP para uma feature específica.
        
        Args:
            shap_values (shap.Explanation): Valores SHAP
            X (pandas.DataFrame): Features
            feature (str): Feature a analisar
            interaction_feature (str, optional): Feature de interação
            title (str, optional): Título do gráfico
            figsize (tuple): Tamanho da figura
            
        Returns:
            matplotlib.figure.Figure: Figura gerada
        """
        # Criar figura
        plt.figure(figsize=figsize)
        
        # Plotar dependência SHAP
        shap.dependence_plot(feature, shap_values.values, X, interaction_index=interaction_feature, show=False)
        
        if title is None:
            title = f'SHAP Dependence para {feature}'
            if interaction_feature is not None:
                title += f' com interação de {interaction_feature}'
        plt.title(title)
        
        fig = plt.gcf()
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig, filename, path=None, formats=None, dpi=300):
        """
        Salva uma figura em vários formatos.
        
        Args:
            fig (matplotlib.figure.Figure): Figura a salvar
            filename (str): Nome do arquivo (sem extensão)
            path (str, optional): Caminho para salvar
            formats (list, optional): Lista de formatos
            dpi (int): Resolução em DPI
        """
        if path is None:
            path = os.path.join(self.output_dir, 'figures')
            
        if not os.path.exists(path):
            os.makedirs(path)
            
        if formats is None:
            formats = ['png', 'pdf']
            
        for fmt in formats:
            fig.savefig(os.path.join(path, f"{filename}.{fmt}"), dpi=dpi, bbox_inches='tight')
            
        plt.close(fig)