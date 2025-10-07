"""
Módulo para gerenciamento de painéis genéticos.
"""
import json
import os

class PanelManager:
    """
    Classe para gerenciar painéis genéticos e carregar dados JSON.
    """
    
    def __init__(self, default_json_path=None):
        """
        Inicializa o PanelManager.

        Args:
            default_json_path (str, optional): Caminho padrão para genetic_panels.json. Se None, usa o arquivo na raiz do projeto.
        """
        if default_json_path is None:
            # Usa genetic_panels.json na raiz do projeto (um nível acima de modules)
            project_root = os.path.dirname(os.path.dirname(__file__))
            default_json_path = os.path.join(project_root, 'genetic_panels.json')
        self.default_json_path = default_json_path
    
    def load_panels_from_json(self, json_path=None):
        """
        Carrega painéis genéticos de um arquivo JSON.
        
        Args:
            json_path (str, optional): Caminho para o arquivo JSON. Se None, usa o caminho padrão.
            
        Returns:
            dict: Dicionário com os painéis genéticos
        """
        if json_path is None:
            json_path = self.default_json_path

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Arquivo JSON não encontrado: {json_path}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            panels = json.load(f)
            
        return panels

    def load_default_panels(self):
        """
        Carrega os painéis usando o caminho padrão configurado no manager.
        """
        return self.load_panels_from_json(self.default_json_path)
    
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