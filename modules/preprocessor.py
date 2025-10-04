"""
Módulo para pré-processamento de dados clínicos e genéticos.
"""
import pandas as pd
import numpy as np
import os

class Preprocessor:
    """
    Classe para pré-processamento de dados clínicos e genéticos.
    """
    
    def __init__(self, output_dir='outputs'):
        """
        Inicializa o Preprocessor com o diretório de saída.
        
        Args:
            output_dir (str): Caminho para o diretório de saída
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def standardize_sample_ids(self, datas):
        """
        Padroniza os IDs de amostra nos dados clínicos.
        
        Args:
            datas (pandas.DataFrame): DataFrame com dados clínicos ou somáticos
            
        Returns:
            pandas.DataFrame: DataFrame com IDs padronizados
        """
        # Cópia para não modificar o original
        datas = datas.copy()
        # 🔹 Renomeia para padronizar os nomes de colunas antes de manipular
        if 'Sample_ID' in datas.columns:
            datas = datas.rename(columns={'Sample_ID': 'Sample ID'})
        datas['Sample ID'] = datas['Sample ID'].str[:12]
        
        return datas
    
    
    def merge_clinical_and_somatic(self, clinical_data, somatic_data):
        """
        Mescla dados clínicos e somáticos com base na coluna 'Sample ID'.
        
        Args:
            clinical_data (pandas.DataFrame): DataFrame com dados clínicos
            somatic_data (pandas.DataFrame): DataFrame com dados somáticos
            
        Returns:
            pandas.DataFrame: DataFrame mesclado
        """
        # Padroniza os IDs de amostra
        clinical_data = self.standardize_sample_ids(clinical_data)
        somatic_data = self.standardize_sample_ids(somatic_data)
        
        # Mescla os DataFrames
        merged_df = pd.merge(clinical_data, somatic_data, on='Sample ID', how='inner')
        
        return merged_df
        
    def create_gene_matrix(self, somatic_data, merged_data, output_dir="outputs"):
        """
        Cria uma matriz gene x subtipo a partir do DataFrame mesclado (dados clínicos + somáticos).
        
        Etapas:
        1. Remove variantes 'synonymous_variant'
        2. Cria uma matriz onde genes viram colunas (1 = gene mutado, 0 = não mutado)
        3. Remove casos POLE (STAD_POLE)
        4. Renomeia subtipos (para minúsculo e sem prefixo 'STAD_')
        5. Salva o resultado em CSV
        
        Args:
            somatic_data (pd.DataFrame): DataFrame com dados somáticos.
            merged_data (pd.DataFrame): DataFrame resultante da fusão clínica + somática.
            output_dir (str): Diretório onde o CSV final será salvo (padrão: 'outputs')
        
        Returns:
            pd.DataFrame: DataFrame final (Sample ID, Subtype, genes binários)
        """

        # 🔹 1. Remover variantes sinônimas
        snp_subtype_missense_data = merged_data[merged_data['effect'] != 'synonymous_variant']
        somatic_data = self.standardize_sample_ids(somatic_data)
        excluidas = len(somatic_data['Sample ID']) - len(snp_subtype_missense_data['Sample ID'])
        print(f"Foram excluídas: {excluidas} amostras")
        print(f"Restando:")
        print(f"- Casos: {len(snp_subtype_missense_data['Sample ID'].unique())}")
        print(f"- Amostras: {len(snp_subtype_missense_data['Sample ID'])}")

        # 🔹 2. Remover a coluna 'effect'
        df = snp_subtype_missense_data.drop(columns=['effect'])

        # 🔹 3. Criar matriz pivotada (genes como colunas)
        df_pivot = df.pivot_table(index=['Sample ID', 'Subtype'], columns='gene', aggfunc=lambda x: 1, fill_value=0)

        # Redefinir o índice para transformar em um DataFrame simples
        df_final = df_pivot.reset_index()

        # 🔹 5. Remover casos STAD_POLE
        df_final = df_final[df_final['Subtype'] != 'STAD_POLE']
        print("\nDistribuição dos subtipos após remoção de POLE:")
        print(df_final['Subtype'].value_counts())
        print(f"\nDimensões finais do dataset: {df_final.shape}")


        # Renomeando subtipos para minúsculo e retirando 'STAD_'
        df_final['Subtype'] = df_final['Subtype'].apply((lambda x: x.split('_')[1].lower()))

        # 🔹 6. Salvar o CSV no diretório especificado
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'gene_subtype_table.csv')
        df_final.to_csv(output_path, index=False)

        print(f"\n✅ Arquivo salvo em: {output_path}")

        return df_final

    def create_data_set(self, df_final, test_list):
        test = df_final['Sample ID'].isin(test_list)
        test_df = df_final[test]
        print(f'Dimensções do data frame de teste: {test_df.shape}')

        train_val = ~test
        train_val_df = df_final[train_val]
        print(f'Dimensções do data frame de de teste e validação: {train_val_df.shape}')
        
        return train_val_df, test_df

    def X_y_df_split(self, train_val_df, test_df):
        # Ler dados:
        X_train_val = train_val_df.drop(["Subtype", "Sample ID"], axis=1)
        y_train_val = train_val_df["Subtype"] #.values

        X_test      = test_df.drop(["Subtype", "Sample ID"], axis=1)
        y_test      = test_df["Subtype"] #.values
        sample_ids  = test_df["Sample ID"] # .values
        
        return X_train_val, y_train_val, X_test, y_test, sample_ids
    

    def create_train_test_split(self, test_list, df_final):
        """
        Divide o DataFrame em conjuntos de treinamento e teste.
        
        Args:
            df_final (pd.DataFrame): DataFrame final com genes e subtipos.
            test_size (float): Proporção do conjunto de teste (padrão: 0.2).
            random_state (int): Seed para reproducibilidade (padrão: 42).
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """

        print("Casos que estão na lista dos casos usados para o teste do ensemble mas não estão nos casos restantes das tabelas mesclada de subtipos e genes")
        print([x for x in test_list if x not in df_final['Sample ID'].unique()])

        test_list.remove('TCGA-RD-A8N2')
        print(f"Quantidade de casos que serão separados para o dataset de teste: {len(test_list)}")

        train_val_df, test_df = self.create_data_set(df_final, test_list)
        print("Distribuição no dataset de teste:")
        print(test_df['Subtype'].value_counts())
        X_train_val, y_train_val, X_test, y_test, sample_ids = self.X_y_df_split(train_val_df, test_df)

        return X_train_val, y_train_val, X_test, y_test, sample_ids, train_val_df, test_df

