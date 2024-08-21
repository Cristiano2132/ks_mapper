import unittest
import pandas as pd
import numpy as np
from mapper import DevPrdModelPerformanceMapper

class TestDevPrdModelPerformanceMapper(unittest.TestCase):

    def setUp(self):
        # Configuração básica de dataframes para usar em todos os testes
        self.df1 = pd.DataFrame({'id': [1, 2, 3, 4, 5], 'col1': [1, 2, 3, 4, 5], 'col2': [1, 2, 3, 4, 5]})
        self.df2 = pd.DataFrame({'id': [1, 2, 3, 4, 5], 'col1': [1, 2, 3, 6, 7], 'col2': [6, 7, 8, 9, 10]})

    def test_no_problems(self):
        """Testa se o método retorna corretamente um dataframe com KS e batimento quando não há problemas"""
        mapper = DevPrdModelPerformanceMapper(df1=self.df1, df2=self.df2, features_modelo=['col1', 'col2'])
        res = mapper.get_ks_features(features_to_map=['col1', 'col2'], prefix1='df1_', prefix2='df2_')
        self.assertIn('feature', res.columns)
        self.assertIn('ks', res.columns)
        self.assertIn('batimento', res.columns)
        self.assertEqual(len(res), 2)

    def test_different_id_columns(self):
        """Testa se um erro é levantado quando as colunas de ID são diferentes"""
        erro_esperado = 'Erro: coluna id não está presente em algum dos dataframes'
        df1 = self.df1.copy().rename(columns={'id': 'id_df1'})
        mapper = DevPrdModelPerformanceMapper(df1=df1, df2=self.df2, features_modelo=['col1', 'col2'])
        with self.assertRaises(ValueError) as context:
            mapper.get_ks_features(features_to_map=['col1', 'col2'], prefix1='df1_', prefix2='df2_')
        erro_observado = str(context.exception)
        self.assertTrue(erro_esperado == erro_observado)

    def test_different_id_types(self):
        """Testa se um erro é levantado quando as colunas de ID têm tipos diferentes"""
        erro_esperado = 'Erro: colunas id não são do mesmo tipo em ambos dataframes'
        df1 = self.df1.copy()
        df1['id'] = df1['id'].astype(str)
        mapper = DevPrdModelPerformanceMapper(df1=df1, df2=self.df2, features_modelo=['col1', 'col2'])
        with self.assertRaises(ValueError) as context:
            mapper.get_ks_features(features_to_map=['col1', 'col2'], prefix1='df1_', prefix2='df2_')
        erro_observado = str(context.exception)
        self.assertTrue(erro_esperado == erro_observado)
        
    def test_different_feature_types(self):
        """Testa se um erro é levantado quando as features têm tipos diferentes"""
        erro_esperado = "Erro: colunas ['col1', 'col2'] não são do mesmo tipo em ambos dataframes (df1 e df2)"
        df1 = self.df1.copy()
        df1['col1'] = df1['col1'].astype(str)
        mapper = DevPrdModelPerformanceMapper(df1=df1, df2=self.df2, features_modelo=['col1', 'col2'])
        with self.assertRaises(ValueError) as context:
            mapper.get_ks_features(features_to_map=['col1', 'col2'], prefix1='df1_', prefix2='df2_')
        erro_observado = str(context.exception)
        self.assertTrue(erro_esperado == erro_observado)

    def test_missing_features(self):
        """Testa se um erro é levantado quando alguma feature não está presente em ambos os dataframes"""
        erro_esperado = "Erro: as variáveis ['col3'] não estão presentes no dataframe"
        mapper = DevPrdModelPerformanceMapper(df1=self.df1, df2=self.df2, features_modelo=['col1', 'col2'])
        with self.assertRaises(ValueError) as context:
            mapper.get_ks_features(features_to_map=['col1', 'col2', 'col3'], prefix1='df1_', prefix2='df2_')
        erro_observado = str(context.exception)
        self.assertTrue(erro_esperado == erro_observado)
        
    def test_null_feature(self):
        """Testa se o método ignora corretamente features com valores nulos"""
        df2 = self.df2.copy()
        df2['col2'] = np.nan
        mapper = DevPrdModelPerformanceMapper(df1=self.df1, df2=df2, features_modelo=['col1', 'col2'])
        res = mapper.get_ks_features(features_to_map=['col1', 'col2'], prefix1='df1_', prefix2='df2_')
        features_mapped = res['feature'].values
        self.assertNotIn('col2', features_mapped)

    def test_plot_distribution(self):
        """Testa se o método de plot funciona corretamente"""
        df1 = pd.DataFrame({'target': np.random.normal(0, 1, 1000), 'id': np.arange(1000)})
        df2 = pd.DataFrame({'target': np.random.normal(0, 1, 1000), 'id': np.arange(1000)})
        mapper = DevPrdModelPerformanceMapper(df1=df1, df2=df2, features_modelo=['target'])
        fig, ax = mapper.plot_distribution(var='target', prefix1='df1_', prefix2='df2_')
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

if __name__ == '__main__':
    unittest.main()