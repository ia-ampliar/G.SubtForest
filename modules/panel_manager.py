"""
Módulo para gerenciamento de painéis genéticos.
"""
import json
import os

class PanelManager:
    """
    Classe para gerenciar painéis genéticos e carregar dados JSON.
    """
    
    def __init__(self):
        """
        Inicializa o PanelManager.
        """
        pass
    
    def load_panels_from_json(self, json_path):
        """
        Carrega painéis genéticos de um arquivo JSON.
        
        Args:
            json_path (str): Caminho para o arquivo JSON
            
        Returns:
            dict: Dicionário com os painéis genéticos
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Arquivo JSON não encontrado: {json_path}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            panels = json.load(f)
            
        return panels
    
    def get_panel_genes(self, panels, panel_name):
        """
        Obtém os genes de um painel específico.
        
        Args:
            panels (dict): Dicionário com painéis
            panel_name (str): Nome do painel
            
        Returns:
            list: Lista de genes do painel
        """
        if panel_name not in panels:
            raise KeyError(f"Painel '{panel_name}' não encontrado nos painéis disponíveis: {list(panels.keys())}")
            
        return panels[panel_name]
    
    def list_available_panels(self, panels):
        """
        Lista os painéis disponíveis.
        
        Args:
            panels (dict): Dicionário com painéis
            
        Returns:
            list: Lista com nomes dos painéis
        """
        return list(panels.keys())
    
    def get_panel_info(self, panels, panel_name):
        """
        Obtém informações sobre um painel específico.
        
        Args:
            panels (dict): Dicionário com painéis
            panel_name (str): Nome do painel
            
        Returns:
            dict: Informações do painel (nome, número de genes, lista de genes)
        """
        genes = self.get_panel_genes(panels, panel_name)
        
        return {
            'name': panel_name,
            'num_genes': len(genes),
            'genes': genes
        }