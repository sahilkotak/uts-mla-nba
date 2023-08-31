# from sklearn.preprocessing import PolynomialFeatures


# class FeatureEngineer:
#     def __init__(self, dataframe, target_column=None):
#         """
#         Initialize the FeatureEngineer class with a dataframe.

#         Parameters:
#         - dataframe: The dataset to be used for feature engineering.
#         - target_column: The target column if available. Used for polynomial features.
#         """
#         self.df = dataframe.copy()
#         if target_column:
#             self.target = dataframe[target_column]
#         else:
#             self.target = None

#     def player_efficiency(self):
#         """Compute the player efficiency metric."""
#         # Compute total rebounds as the sum of offensive and defensive rebounds
#         self.df['total_rebounds'] = self.df['oreb'] + self.df['dreb']

#         self.df['usage_efficiency'] = (self.df['pts'] + self.df['total_rebounds'] + self.df['ast']
#                                        + self.df['stl'] + self.df['blk']
#                                        - (self.df['fta'] - self.df['ft'])
#                                        - (self.df['fga'] - self.df['fg'])
#                                        - self.df['to']) / self.df['mp']
#         return self

#     def defense_to_offense_ratio(self):
#         """Compute the defense to offense ratio."""
#         self.df['def_to_off_ratio'] = (
#             self.df['stl'] + self.df['blk']) / (self.df['pts'] + self.df['ast'])
#         # Handle infinite values due to division by zero
#         self.df['def_to_off_ratio'].replace([np.inf, -np.inf], 0, inplace=True)
#         return self

#     def rebound_efficiency(self):
#         """Compute the rebound efficiency."""
#         self.df['total_rebounds'] = self.df['oreb'] + self.df['dreb']
#         return self

#     def playermaker_ability(self):
#         """Compute the playmaker ability."""
#         # This column already exists in the dataset, so we will not recompute it.
#         pass

#     def defensive_metric(self):
#         """Compute the defensive metric."""
#         self.df['defensive_metric'] = (
#             self.df['stl'] + self.df['blk']) / self.df['mp']
#         return self

#     def offensive_metric(self):
#         """Compute the offensive metric."""
#         self.df['offensive_metric'] = (
#             self.df['pts'] + self.df['ast']) / self.df['mp']
#         return self

#     def scoring_efficiency(self):
#         """Compute the scoring efficiency."""
#         self.df['scoring_efficiency'] = self.df['pts'] / \
#             (self.df['fga'] + 0.44 * self.df['fta'])
#         return self

#     def polynomial_features(self):
#         """Generate polynomial features for efficiency related columns."""
#         if self.target is not None:
#             # Extracting main columns for polynomial features
#             columns_to_poly = ['usage_efficiency', 'scoring_efficiency', 'def_to_off_ratio', 'total_rebounds',
#                                'defensive_metric', 'offensive_metric']
#             poly = PolynomialFeatures(2, interaction_only=True)
#             interaction_terms = poly.fit_transform(self.df[columns_to_poly])

#             # Extracting the feature names and appending to the dataframe
#             feature_names = poly.get_feature_names_out(columns_to_poly)
#             interaction_df = pd.DataFrame(
#                 interaction_terms, columns=feature_names)

#             # Dropping original columns and 1's column from polynomial features
#             interaction_df.drop(columns=columns_to_poly + ['1'], inplace=True)

#             # Merging the polynomial features back to the main dataframe
#             self.df = pd.concat([self.df, interaction_df], axis=1)
#         return self

#     def binning_features(self):
#         """Binning various metrics into categories."""
#         # Binning scoring efficiency
#         self.df['scoring_efficiency_bin'] = pd.qcut(self.df['scoring_efficiency'],
#                                                     q=3, labels=["low", "medium", "high"])

#         # Binning defensive metric
#         self.df['defensive_metric_bin'] = pd.qcut(self.df['defensive_metric'],
#                                                   q=3, labels=["low", "medium", "high"])

#         # Binning offensive metric
#         self.df['offensive_metric_bin'] = pd.qcut(self.df['offensive_metric'],
#                                                   q=3, labels=["low", "medium", "high"])
#         return self
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
import pandas as pd
import numpy as np


# Modify the FeatureEngineer class to conditionally create new features

class FeatureEngineer:
    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def three_point_preference(self):
        if '3PA' in self.df.columns and 'FGA' in self.df.columns:
            self.df['three_point_preference'] = self.df['3PA'] / self.df['FGA']
        return self

    def inside_scoring_ability(self):
        if '2P' in self.df.columns and 'FGA' in self.df.columns:
            self.df['inside_scoring_ability'] = self.df['2P'] / self.df['FGA']
        return self

    def playmaking_ability(self):
        if 'ast' in self.df.columns and 'pts' in self.df.columns:
            # Prevent division by zero by replacing zero 'pts' with a small number
            non_zero_pts = self.df['pts'].replace(0, np.finfo(float).eps)
            self.df['playmaking_ability'] = self.df['ast'] / non_zero_pts
        return self

    def defensive_index(self):
        if 'stl' in self.df.columns and 'blk' in self.df.columns and 'GP' in self.df.columns:
            self.df['defensive_metric'] = (self.df['stl'] + self.df['blk']) / self.df['GP']
        return self

    def versatility_index(self):
        if all(col in self.df.columns for col in ['pts', 'ast', 'stl', 'blk']):
            self.df['versatility_index'] = self.df['pts'] * self.df['ast'] * self.df['stl'] * self.df['blk']
        return self

    def polynomial_features(self):
        # Select only numeric columns (excluding 'drafted')
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        if 'drafted' in numeric_cols:
            numeric_cols.remove('drafted')
            
        if 'pts' in numeric_cols:
            self.df = self.df.join(
                self.df[numeric_cols].apply(lambda x: x * self.df['pts'], axis=0).add_suffix('_pts'),
                rsuffix='_pts'
            )
        if 'ast' in numeric_cols:
            self.df = self.df.join(
                self.df[numeric_cols].apply(lambda x: x * self.df['ast'], axis=0).add_suffix('_ast'),
                rsuffix='_ast'
            )
        if 'stl' in numeric_cols:
            self.df = self.df.join(
                self.df[numeric_cols].apply(lambda x: x * self.df['stl'], axis=0).add_suffix('_stl'),
                rsuffix='_stl'
            )
        if 'blk' in numeric_cols:
            self.df = self.df.join(
                self.df[numeric_cols].apply(lambda x: x * self.df['blk'], axis=0).add_suffix('_blk'),
                rsuffix='_blk'
            )
        return self

    def binning_features(self):
        if 'defensive_metric' in self.df.columns:
            self.df['defensive_metric_bin'] = pd.cut(self.df['defensive_metric'], bins=3, labels=['Low', 'Medium', 'High'])
        return self

    def get_dataframe(self):
        return self.df
