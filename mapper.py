import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
from scipy import stats
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from model_loader import ModelLoader

class DevPrdModelPerformanceMapper:
    def __init__(self, df1: pd.DataFrame, df2: pd.DataFrame, features_modelo: list):
        self.df1 = df1
        self.df2 = df2
        self.features_modelo = features_modelo

    @staticmethod
    def __style_ax(ax):
        # Axis styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.tick_params(bottom=False, left=False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#EEEEEE')
        ax.xaxis.grid(False)

    def __check_columns_types_math(self, columns: list[str])->bool:
        types_df1 = [self.df1[c].dtype for c in columns]
        types_df2 = [self.df2[c].dtype for c in columns]
        not_matches = [c1!=c2 for c1, c2 in zip(types_df1, types_df2)]
        if sum(not_matches) == 0:
            return True
        print(f'Erro: os tipos das colunas {[c for i,c in enumerate(columns) if not_matches[i]]} são diferentes')
        return False

    def __check_vars_are_in_df(self, vars: list, df: pd.DataFrame)->bool:
        if not all(x in df.columns for x in vars):
            not_found = [x for x in vars if x not in df.columns]
            if len(not_found) > 0:
                raise ValueError(f'Erro: as variáveis {[x for x in not_found]} não estão presentes no dataframe')
        return True 

    def get_ks_2samp(self, var: str, prefix1: str, prefix2: str, id_column: str = 'id', return_df_combined: bool = False)->tuple[float]:
        features = [var, id_column]
        df_drift = self.df1[features].set_index(id_column).add_prefix(prefix1).join(self.df2[features].set_index(id_column).add_prefix(prefix2), how='inner')
        feature1 = prefix1+var
        feature2 = prefix2+var    
        batimento = sum(df_drift[feature1] == df_drift[feature2])/len(df_drift)
        ks = stats.ks_2samp(df_drift[feature1], df_drift[feature2]).statistic
        if return_df_combined:
            return ks, batimento, df_drift
        return ks, batimento

    def plot_distribution(self, var: str, prefix1: str, prefix2: str, id_column: str = 'id', title:str = None)->tuple[plt.Figure, plt.Axes]:
        
        ks, batimento, df_combined = self.get_ks_2samp(var = var, prefix1 = prefix1, prefix2 = prefix2, id_column = id_column, return_df_combined = True)
        df_res_pivoted = df_combined.melt(value_vars=[f'{prefix1}{var}', f'{prefix2}{var}'], var_name='source', value_name=var)
        fig, ax = plt.subplots(1,1, figsize=(10,6))
        sns.histplot(
            data=df_res_pivoted,
            x=var,
            hue='source',
            kde=True,
            stat='density',
            ax =ax,
            common_norm=False,
            alpha = 0.3
        )
        self.__style_ax(ax)
        
        if title is None:
            ax.set_title(f'Distribuição de probabilidades: ks = {ks:.4f}')
        else:
            ax.set_title(title + f': ks = {ks:.4f}')
        fig.tight_layout()
        return fig, ax

    def get_ks_features(self, features_to_map: list[str], prefix1: str, prefix2: str, id_column: str = 'id')->Union[pd.DataFrame, None]:

        features_are_indf1_and_df2 = (self.__check_vars_are_in_df(features_to_map, self.df1) and self.__check_vars_are_in_df(features_to_map, self.df2))
        # remove de features_to_map a feature se ela é nula em algum dos dataframes
        features_to_map = [f for f in features_to_map if self.df1[f].isna().sum() == 0 and self.df2[f].isna().sum() == 0]
        if not features_are_indf1_and_df2:
            raise ValueError('Erro: existe alguma feature em features_to_map não presente em ambos dataframes')
            
            
        if id_column not in self.df1.columns or id_column not in self.df2.columns:
            raise ValueError(f'Erro: coluna {id_column} não está presente em algum dos dataframes')
        id_columns_type_mattch = self.__check_columns_types_math(columns = [id_column])
        if not id_columns_type_mattch:
            raise ValueError(f'Erro: colunas {id_column} não são do mesmo tipo em ambos dataframes')

        features_to_map_types_match = self.__check_columns_types_math(columns = features_to_map)
        if not features_to_map_types_match:
            raise ValueError(f'Erro: colunas {features_to_map} não são do mesmo tipo em ambos dataframes (df1 e df2)')

        res_ks = []
        for i, feature in enumerate(features_to_map):
            if self.df1[feature].isna().sum() == len(self.df1) or self.df2[feature].isna().sum() == len(self.df2):
                continue
            ks, batimento = self.get_ks_2samp(var = feature,  prefix1 = prefix1, prefix2 = prefix2, id_column = id_column)  
            res_ks.append({'feature': feature, 'batimento': batimento, 'ks': ks})
        res_ks = pd.DataFrame(res_ks)
        return res_ks
    
    def get_ks_variation_with_feature_swap(self, model_path: str, features_model: list[str], features_to_map: list[str], id_column: str = 'id', copy_df1_to_df2: bool = True, model_predict_method: str = 'predict')->Union[pd.DataFrame, None]:
        model_loader = ModelLoader()
        model = model_loader.load_model(model_path=model_path, predict_method = model_predict_method)

        if copy_df1_to_df2:
            data = {'1': self.df1, '2': self.df2}
        else:
            data = {'1': self.df2, '2': self.df1}
        ks = []
        feature_substituida = []
        target1 = model.predict(data.get('1').set_index(id_column)[features_model])
        target2 = model.predict(data.get('2').set_index(id_column)[features_model])
        ks_original = stats.ks_2samp(target1, target2).statistic
        ks.append(ks_original)
        feature_substituida.append('original')
    
        for feature in features_to_map:
            df = data.get('2').copy()
            ids = df[id_column].values
            df[feature] = data.get('1').set_index(id_column).loc[ids][feature].values
            target2 = model.predict(df.set_index(id_column)[features_model])
            ks_after_swap = stats.ks_2samp(target1, target2).statistic
            ks.append(ks_after_swap)
            feature_substituida.append(feature)
        res = pd.DataFrame({'feature_substituida': feature_substituida, 'ks': ks})
        return res

if __name__ == '__main__':

    model_path = 'random_forest_model.pkl'
    
    # simular 2k observaçoes de 6 features A, B, C, D, E, F e uma variável resposta Y tal que Y = A + 2B -8C + D + E/2 + F + erro onde o erro segue uma distribuição normal com media 5 e desvio padrão 2
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(2000, 6), columns = ['A', 'B', 'C', 'D', 'E', 'F'])
    df['id'] = range(2000)
    df['y'] = df['A'] + 2*df['B'] - 8*df['C'] + df['D'] + df['E']/2 + df['F'] + np.random.normal(0, 1, 2000)
    
 
    X_train, X_test, y_train, y_test = train_test_split(df[['id', 'A', 'B', 'C', 'D', 'E', 'F']], df[['id','y']], test_size=0.25, random_state=42)
    
    model = RandomForestRegressor()
    model.fit(X_train[['A', 'B', 'C', 'D', 'E', 'F']], y_train['y'])
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    df1 = X_test.copy()
    df1['y'] = model.predict(df1[['A', 'B', 'C', 'D', 'E', 'F']])
    df2 = X_test.copy()
    
    # adicionar alguns ruidos nas features para simular drift
    df2['A'] = df2['A'] + np.random.normal(0, 0.1, len(df2))
    df2['B'] = df2['B'] + np.random.normal(0, 0.5, len(df2))
    df2['C'] = df2['C'] + np.random.normal(0, 0.7, len(df2))
    df2['D'] = df2['D'] + np.random.normal(0, 1, len(df2))
    df2['y'] = model.predict(df2[['A', 'B', 'C', 'D', 'E', 'F']])
    
    mapper = DevPrdModelPerformanceMapper(df1=df1, df2=df2, features_modelo=['A', 'B', 'C', 'D', 'E', 'F'])
    # print(mapper.get_ks_2samp(var = 'A', prefix1 = 'df1_', prefix2 = 'df2_', id_column = 'id'))
    # print(mapper.get_ks_features(features_to_map = ['A', 'B', 'C', 'D', 'E', 'F'], prefix1 = 'df1_', prefix2 = 'df2_', id_column = 'id'))
    print(mapper.get_ks_variation_with_feature_swap(model_path = model_path, features_model = ['A', 'B', 'C', 'D', 'E', 'F'], features_to_map = ['A', 'B', 'C', 'D', 'E', 'F'], id_column = 'id', copy_df1_to_df2 = True, model_predict_method = 'predict'))
    

    
    # mapper = DevPrdModelPerformanceMapper(df1=df1, df2=df2, features_modelo=[f'feature_{i+1}' for i in range(5)])
    # print(mapper.get_ks_features(features_to_map = [f'feature_{i+1}' for i in range(5)], prefix1 = 'df1_', prefix2 = 'df2_', id_column = 'ID'))
    # mapper.get_ks_variation_with_feature_swap(model_path = model_path, features_model = [f'feature_{i+1}' for i in range(5)], features_to_map = [f'feature_{i+1}' for i in range(5)], id_column = 'ID', copy_df1_to_df2 = True, model = 'sklearn_rf')
    
    
    
    
        