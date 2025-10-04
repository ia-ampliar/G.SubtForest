"""
Módulo para carregamento de dados clínicos e genéticos.
"""
import pandas as pd
import os

class DataLoader:
    """
    Classe para carregamento e manipulação de dados clínicos e genéticos.
    """
    
    def __init__(self, data_dir='datas'):
        """
        Inicializa o DataLoader com o diretório de dados.
        
        Args:
            data_dir (str): Caminho para o diretório de dados
        """
        self.data_dir = data_dir
        
    def load_clinical_data(self, file_path=None):
        """
        Carrega dados clínicos com colunas 'Sample ID' e 'Subtype'.
        
        Args:
            file_path (str, optional): Caminho para o arquivo de dados clínicos.
                Se não fornecido, usa o caminho padrão.
                
        Returns:
            pandas.DataFrame: DataFrame com dados clínicos
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, 'Dados STAD clínico e patológico TCGA doutorado Julio.xlsx - data.csv')
        
        clinical_data = pd.read_csv(file_path, usecols=['Sample ID', 'Subtype'])
        clinical_data = clinical_data.dropna()
        return clinical_data
    
    def load_somatic_data(self, file_path=None):
        """
        Carrega dados somáticos com colunas 'Sample_ID', 'gene' e 'effect'.
        
        Args:
            file_path (str, optional): Caminho para o arquivo de dados somáticos.
                Se não fornecido, usa o caminho padrão.
                
        Returns:
            pandas.DataFrame: DataFrame com dados somáticos
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, 'TCGA-STAD.varscan2_snv - TCGA-STAD.varscan2_snv.csv')
        
        somatic_data = pd.read_csv(file_path, usecols=['Sample_ID', 'gene', 'effect'])
        return somatic_data

    def get_subtype_statistics(self, clinical_data):
        """
        Calcula estatísticas sobre os subtipos moleculares.
        
        Args:
            clinical_data (pandas.DataFrame): DataFrame com dados clínicos
            
        Returns:
            dict: Dicionário com estatísticas dos subtipos
        """
        n_cases = int(len(clinical_data['Sample ID'].unique()))
        print(f"Quantidade total de casos: {n_cases}")
        subtype_counts = clinical_data['Subtype'].value_counts()
        print("\nQuantidade de casos por subtipo:")
        print(subtype_counts)

        vc = clinical_data['Subtype'].value_counts()
        n_cin_cases = vc.get('STAD_CIN', default=0)
        n_msi_cases = vc.get('STAD_MSI', default=0)
        n_gs_cases = vc.get('STAD_GS', default=0)
        n_ebv_cases = vc.get('STAD_EBV', default=0)
        n_pole_cases = vc.get('STAD_POLE', default=0)
        print("Porcentagem por classe:")
        if n_cases > 0:
            print(f"CIN: {n_cin_cases / n_cases * 100:.2f}%")
            print(f"MSI: {n_msi_cases / n_cases * 100:.2f}%")
            print(f"GS: {n_gs_cases / n_cases * 100:.2f}%")
            print(f"EBV: {n_ebv_cases / n_cases * 100:.2f}%")
            print(f"POLE: {n_pole_cases / n_cases * 100:.2f}%")
            
    
    def load_gene_subtype_table(self, file_path=None):
        """
        Carrega a tabela final com genes e subtipos.
        
        Args:
            file_path (str, optional): Caminho para o arquivo da tabela.
                Se não fornecido, usa o caminho padrão.
                
        Returns:
            pandas.DataFrame: DataFrame com a tabela de genes e subtipos
        """
        if file_path is None:
            file_path = 'outputs/gene_subtype_table.csv'
            
        df_final = pd.read_csv(file_path)
        return df_final
    
    def load_gene_panels(self, file_path=None):
        """
        Carrega painéis genéticos a partir de um arquivo JSON.
        
        Args:
            file_path (str, optional): Caminho para o arquivo JSON.
                Se não fornecido, usa o caminho padrão.
                
        Returns:
            dict: Dicionário com painéis genéticos
        """
        import json
        
        if file_path is None:
            file_path = 'outputs/gene_panels.json'
            
        with open(file_path, 'r') as f:
            panels = json.load(f)
            
        return panels