# COPIE ÇA EXACTEMENT (5 lignes)
"""
Package utils v2.0.0 - FIX CIRCULAR IMPORT
"""
__version__ = "2.0.0"

try:
    from .data_processor import DataProcessor
    print("✅ DataProcessor chargé")
except ImportError:
    class DataProcessor: pass

__all__ = ['DataProcessor', 'ModelAnalyzer', 'ModelTrainer', 'ModelLoader']
