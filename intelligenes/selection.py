# (Packages/Libraries) Matrix Manipulation
import pandas as pd 

# (Packages/Libraries) Statistical Analysis & Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import pearsonr

# (Packages/Libraries) Miscellaneous
import argparse
import warnings
from sklearn.exceptions import ConvergenceWarning
import os
from datetime import datetime

class FeatureSelection:
    
    def __init__(self: 'FeatureSelection', cgit_file: str, output_dir: str, random_state: 42, test_size: 0.3, use_rfe = True, use_pearson = True, use_chi2 = True, use_anova = True, use_normalization = False): 
        self.cgit_file = cgit_file
        self.output_dir = output_dir
        self.random_state = random_state
        self.test_size = test_size
        self.use_rfe = use_rfe
        self.use_pearson = use_pearson
        self.use_chi2 = use_chi2
        self.use_anova = use_anova
        self.use_normalization = use_normalization
        
        self.df = pd.read_csv(self.cgit_file)
        
        self.y = self.df['Type']
        self.X = self.df.drop(['Type', 'ID'], axis = 1)
        
        if self.use_normalization:
            self.X = pd.DataFrame(MinMaxScaler().fit_transform(self.X), columns = self.X.columns)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = self.test_size, random_state = self.random_state)
        
        self.selectors = []        

    def rfe_selector(self: 'FeatureSelection'):
        if self.use_rfe:
            print("Recursive Feature Elimination...") 
            rfe_selection = RFE(estimator = DecisionTreeClassifier(random_state = self.random_state), n_features_to_select = 1).fit(self.X_train, self.y_train)
            rfe_df = pd.DataFrame({'attributes': self.X_train.columns,
                                   'rfe_rankings': rfe_selection.ranking_})
            
            rfe_df = rfe_df.sort_values(by = 'rfe_rankings').loc[rfe_df['rfe_rankings'] <= int((self.df.shape[1] - 2) * .10)]
            return rfe_df 
        return None

    def pearson_selector(self: 'FeatureSelection'):
        if self.use_pearson:
            print("Pearson's Correlation...") 
            pearson_selection = [pearsonr(self.X_train[column], self.y_train) for column in self.X.columns]
            pearson_df = pd.DataFrame({'attributes': self.X_train.columns,
                                       'pearson_p-value': [corr[1] for corr in pearson_selection]})
        
            pearson_df = pearson_df[pearson_df['pearson_p-value'] < 0.05]
            return pearson_df
        return None
    
    def chi2_selector(self: 'FeatureSelection'):
        if self.use_chi2:
            print("Chi-Square Test...") 
            chi2_selection = SelectKBest(score_func = chi2, k = 10).fit(self.X_train, self.y_train)
            chi2_df = pd.DataFrame({'attributes': self.X_train.columns, 
                                    'chi2_p-value': chi2_selection.pvalues_})
            
            chi2_df = chi2_df[chi2_df['chi2_p-value'] < 0.05]
            return chi2_df
        return None
              
    def anova_selector(self: 'FeatureSelection'):
        if self.use_anova:
            print("ANOVA...")
            anova_selection = SelectKBest(score_func = f_classif, k = 10).fit(self.X_train, self.y_train)
            anova_df = pd.DataFrame({'attributes': self.X_train.columns, 
                                     'anova_p-value': anova_selection.pvalues_})
 
            anova_df = anova_df[anova_df['anova_p-value'] < 0.05]
            return anova_df
        return None
    
    def execute_selectors(self: 'FeatureSelection'):
        self.selectors = [self.rfe_selector(), 
                          self.pearson_selector(), 
                          self.chi2_selector(), 
                          self.anova_selector()]
        
        self.selectors = [df for df in self.selectors if df is not None]
        
    def selected_attributes(self: 'FeatureSelection'):
        selected_attributes = pd.DataFrame({'attributes': self.X_train.columns})
        for df in self.selectors:
            selected_attributes = selected_attributes.merge(df, how = 'inner', on = 'attributes')

        selector_cols = ['rfe_rankings', 'pearson_p-value', 'chi2_p-value', 'anova_p-value']
        selectors_used = [col for col in selector_cols if col in selected_attributes.columns]
        if any(not self.__dict__[f"use_{selector.split('_')[0]}"] for selector in selectors_used):
            selected_attributes = selected_attributes.dropna(subset = selectors_used, how = 'any')
            
        selected_attributes = selected_attributes.rename(columns={
            'attributes': 'Features',
            'rfe_rankings': 'RFE Rankings',
            'pearson_p-value': "Pearson's Correlation (p-value)",
            'chi2_p-value': 'Chi-Square Test (p-value)',
            'anova_p-value': 'ANOVA (p-value)'
        })

        return selected_attributes
    
def main():
    print("\n")
    print("IntelliGenes Feature Selection/Biomarker Location...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--cgit_file', required = True)
    parser.add_argument('-o', '--output_dir', required = True)
    parser.add_argument('--random_state', type = int, default = 42)
    parser.add_argument('--test_size', type = float, default = 0.3)
    parser.add_argument('--no_rfe', action = 'store_true')
    parser.add_argument('--no_pearson', action = 'store_true')
    parser.add_argument('--no_chi2', action = 'store_true')
    parser.add_argument('--no_anova', action = 'store_true')
    parser.add_argument('--normalize', action = 'store_true')
    args = parser.parse_args()

    pipeline = FeatureSelection(
        cgit_file  = args.cgit_file, 
        output_dir = args.output_dir, 
        random_state = args.random_state, 
        test_size = args.test_size, 
        use_rfe = not args.no_rfe, 
        use_pearson = not args.no_pearson, 
        use_chi2 = not args.no_chi2, 
        use_anova = not args.no_anova, 
        use_normalization = args.normalize
    )
    
    pipeline.execute_selectors()
    features_df = pipeline.selected_attributes()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    features_name = f"{args.cgit_file}_{datetime.now().strftime('%m-%d-%Y-%I-%M-%S-%p')}_Selected-Features.csv"
    features_file = os.path.join(args.output_dir, features_name)
    
    features_df.to_csv(features_file, index = False)
    print("\n Selected Features:", features_file, "\n")

if __name__ == '__main__':
    main()
