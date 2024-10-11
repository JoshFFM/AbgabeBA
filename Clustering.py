import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import inspect

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,
    Normalizer, QuantileTransformer, PowerTransformer
)
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, OPTICS, DBSCAN, AffinityPropagation,
    MeanShift, SpectralClustering, Birch, estimate_bandwidth
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_samples, silhouette_score, calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.cm as cm
import hdbscan

from itertools import cycle, islice
import umap

from analyse import anova, ancova, ancova_mit_transformationen, rang_ancova, glmm_poisson, glmm_negative_binomial, \
    zinb_model, chi_square_tests_and_plots, posthoc_dunn_test, assign_covariates_and_impute_participant_group, \
    simple_stat_tests, permutation_test_with_covariate, permutation_test_without_covariate, permutation_test_with_and_without_covariate
from process_rf_columns_and_generate_heatmaps import process_rf_columns, process_and_save_heatmaps, \
    RF_COLUMNS_EINS_BIS_VIER, RF_ZEITVORGABE, RF_AUFGABENSTELLUNG, RF_NOTE

# Für Kmeans
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["R_HOME"] = "C:/Program Files/R/R-4.4.1"

# Warnungen ignorieren
warnings.filterwarnings("ignore", category=DeprecationWarning)

########################################################################################################################
# 1. Daten laden
########################################################################################################################
# Feature Header laden
# csv_file_path = 'feature_header.csv'
# df = pd.read_csv(csv_file_path)
# # Filtere die Zeilen, bei denen der Wert in der Spalte 'include' True ist
# feature_columns = df[df['include'] == True]['feature'].tolist()
feature_dict = feature_dict = {
    'date_of_last_access': False,
    'datetime': False,
    'duration': False,
    'RF_aufg_interessant': True,  # RF werden nicht im clustering als featurevektoren mit einbezogen. durchgeben für df
    'RF_aufg_motivierend': True,
    'RF_aufgabenstellung': True,
    'RF_aufgabenstellung_text': False,
    'RF_aufmerksam': True,
    'RF_entspannt': True,
    'RF_gereizt': True,
    'RF_gestresst': True,
    'RF_informiertjetzt': True,
    'RF_interessiert': True,
    'RF_kennefakten': True,
    'RF_nervos': True,
    'RF_note': True,
    'RF_weissviel': True,
    'RF_wusstevorher': True,
    'RF_zeitvorgabe': True,
    'AssessmentPhase': False,
    'UserId': False,
    'sessionID': False,
    'copy_ai_tool_string_len': True,  # bei boxplot unten
    'copy_ai_tool_string_len_describe_25': False,
    'copy_ai_tool_string_len_describe_50': True,
    'copy_ai_tool_string_len_describe_75': False,
    'copy_ai_tool_string_len_describe_count': True,  # sollte ggf. raus
    'copy_ai_tool_string_len_describe_max': False,
    'copy_ai_tool_string_len_describe_mean': False,
    'copy_ai_tool_string_len_describe_min': False,
    'copy_ai_tool_string_len_describe_std': False,
    'copy_ai_tool_word_count': False,
    'copy_ai_tool_word_count_describe_25': False,
    'copy_ai_tool_word_count_describe_50': False,
    'copy_ai_tool_word_count_describe_75': False,
    'copy_ai_tool_word_count_describe_count': False,
    'copy_ai_tool_word_count_describe_max': False,
    'copy_ai_tool_word_count_describe_mean': False,
    'copy_ai_tool_word_count_describe_min': False,
    'copy_ai_tool_word_count_describe_std': False,
    'copy_search_string_len': False,
    'copy_search_string_len_describe_25': False,
    'copy_search_string_len_describe_50': False,
    'copy_search_string_len_describe_75': False,
    'copy_search_string_len_describe_count': False,
    'copy_search_string_len_describe_max': False,
    'copy_search_string_len_describe_mean': False,
    'copy_search_string_len_describe_min': False,
    'copy_search_string_len_describe_std': False,
    'copy_search_word_count': False,
    'copy_search_word_count_describe_25': False,
    'copy_search_word_count_describe_50': False,
    'copy_search_word_count_describe_75': False,
    'copy_search_word_count_describe_count': False,
    'copy_search_word_count_describe_max': False,
    'copy_search_word_count_describe_mean': False,
    'copy_search_word_count_describe_min': False,
    'copy_search_word_count_describe_std': False,
    'copy_total_string_len': False,
    'copy_total_string_len_describe_25': False,
    'copy_total_string_len_describe_50': False,
    'copy_total_string_len_describe_75': False,
    'copy_total_string_len_describe_count': False,
    'copy_total_string_len_describe_max': False,
    'copy_total_string_len_describe_mean': False,
    'copy_total_string_len_describe_min': False,
    'copy_total_string_len_describe_std': False,
    'copy_total_word_count': False,
    'copy_total_word_count_describe_25': False,
    'copy_total_word_count_describe_50': False,
    'copy_total_word_count_describe_75': False,
    'copy_total_word_count_describe_count': False,
    'copy_total_word_count_describe_max': False,
    'copy_total_word_count_describe_mean': False,
    'copy_total_word_count_describe_min': False,
    'copy_total_word_count_describe_std': False,
    'copy_unipark_string_len': False,
    'copy_unipark_string_len_describe_25': False,
    'copy_unipark_string_len_describe_50': False,
    'copy_unipark_string_len_describe_75': False,
    'copy_unipark_string_len_describe_count': False,
    'copy_unipark_string_len_describe_max': False,
    'copy_unipark_string_len_describe_mean': False,
    'copy_unipark_string_len_describe_min': False,
    'copy_unipark_string_len_describe_std': False,
    'copy_unipark_word_count': False,
    'copy_unipark_word_count_describe_25': False,
    'copy_unipark_word_count_describe_50': False,
    'copy_unipark_word_count_describe_75': False,
    'copy_unipark_word_count_describe_count': False,
    'copy_unipark_word_count_describe_max': False,
    'copy_unipark_word_count_describe_mean': False,
    'copy_unipark_word_count_describe_min': False,
    'copy_unipark_word_count_describe_std': False,
    'paste_ai_tool_string_len': True,
    'paste_ai_tool_string_len_describe_25': False,
    'paste_ai_tool_string_len_describe_50': True,
    'paste_ai_tool_string_len_describe_75': False,
    'paste_ai_tool_string_len_describe_count': True,
    'paste_ai_tool_string_len_describe_max': False,
    'paste_ai_tool_string_len_describe_mean': False,
    'paste_ai_tool_string_len_describe_min': False,
    'paste_ai_tool_string_len_describe_std': False,
    'paste_ai_tool_word_count': False,
    'paste_ai_tool_word_count_describe_25': False,
    'paste_ai_tool_word_count_describe_50': False,
    'paste_ai_tool_word_count_describe_75': False,
    'paste_ai_tool_word_count_describe_count': False,
    'paste_ai_tool_word_count_describe_max': False,
    'paste_ai_tool_word_count_describe_mean': False,
    'paste_ai_tool_word_count_describe_min': False,
    'paste_ai_tool_word_count_describe_std': False,
    'paste_search_string_len': False,
    'paste_search_string_len_describe_25': False,
    'paste_search_string_len_describe_50': False,
    'paste_search_string_len_describe_75': False,
    'paste_search_string_len_describe_count': False,
    'paste_search_string_len_describe_max': False,
    'paste_search_string_len_describe_mean': False,
    'paste_search_string_len_describe_min': False,
    'paste_search_string_len_describe_std': False,
    'paste_search_word_count': False,
    'paste_search_word_count_describe_25': False,
    'paste_search_word_count_describe_50': False,
    'paste_search_word_count_describe_75': False,
    'paste_search_word_count_describe_count': False,
    'paste_search_word_count_describe_max': False,
    'paste_search_word_count_describe_mean': False,
    'paste_search_word_count_describe_min': False,
    'paste_search_word_count_describe_std': False,
    'paste_total_string_len': False,
    'paste_total_string_len_describe_25': False,
    'paste_total_string_len_describe_50': False,
    'paste_total_string_len_describe_75': False,
    'paste_total_string_len_describe_count': False,
    'paste_total_string_len_describe_max': False,
    'paste_total_string_len_describe_mean': False,
    'paste_total_string_len_describe_min': False,
    'paste_total_string_len_describe_std': False,
    'paste_total_word_count': False,
    'paste_total_word_count_describe_25': False,
    'paste_total_word_count_describe_50': False,
    'paste_total_word_count_describe_75': False,
    'paste_total_word_count_describe_count': False,
    'paste_total_word_count_describe_max': False,
    'paste_total_word_count_describe_mean': False,
    'paste_total_word_count_describe_min': False,
    'paste_total_word_count_describe_std': False,
    'paste_unipark_string_len': True,
    'paste_unipark_string_len_describe_25': False,
    'paste_unipark_string_len_describe_50': True,
    'paste_unipark_string_len_describe_75': False,
    'paste_unipark_string_len_describe_count': True,
    'paste_unipark_string_len_describe_max': False,
    'paste_unipark_string_len_describe_mean': False,
    'paste_unipark_string_len_describe_min': False,
    'paste_unipark_string_len_describe_std': False,
    'paste_unipark_word_count': False,
    'paste_unipark_word_count_describe_25': False,
    'paste_unipark_word_count_describe_50': False,
    'paste_unipark_word_count_describe_75': False,
    'paste_unipark_word_count_describe_count': False,
    'paste_unipark_word_count_describe_max': False,
    'paste_unipark_word_count_describe_mean': False,
    'paste_unipark_word_count_describe_min': False,
    'paste_unipark_word_count_describe_std': False,
    'keystrokes_total': False,
    'text_keystrokes_total': False,
    'keystrokes_unipark': True,
    'text_keystrokes_unipark': False,
    'keystrokes_search': False,
    'text_keystrokes_search': False,
    'keystrokes_ai': True,
    'text_keystrokes_ai': False
}
# Filtere die Features, bei denen 'include' True ist
feature_columns = [feature for feature, include in feature_dict.items() if include]

merge_columns = ['AssessmentPhase', 'UserId', 'sessionID']

# 2. Datenimport
print('1. Datenimport')
input_folder = 'merged_files'
dfs = {}


# Erstelle den Basisordner mit dem aktuellen Datum und der Uhrzeit
output_base = f"clustering_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# Überprüfe, ob der Basisordner existiert, und erstelle ihn, falls nicht
os.makedirs(output_base, exist_ok=True)

removed_rows = []  # Liste, um alle entfernten Zeilen zu speichern

for file in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file)
    df = pd.read_csv(file_path)

    # Bedingungen zum Filtern
    condition = (df['keystrokes_total'] != 0) & ((df['keystrokes_total'] >= 100) | (df['paste_unipark_string_len'] >= 100))

    # Zeilen, die entfernt werden
    removed_df = df[~condition]
    removed_rows.append(removed_df)  # Entfernte Zeilen sammeln

    # Gefilterte Daten
    df_filtered = df[condition]

    # Namen des Fragebogens extrahieren
    questionnaire_name = file.split('/')[-1].split('.')[0]

    # Nur die ausgewählten Spalten speichern
    dfs[questionnaire_name] = df_filtered[feature_columns + merge_columns]

    print(f'{questionnaire_name} added to dfs with filtering applied.')

# Entfernte Zeilen in eine Datei schreiben
if removed_rows:  # Nur wenn es tatsächlich entfernte Zeilen gibt
    removed_data = pd.concat(removed_rows)
    removed_data.to_csv(os.path.join(output_base, 'removed_rows.csv'), index=False)
    print(f'Removed rows saved to "removed_rows.csv".')


########################################################################################################################


########################################################################################################################
# 2. Columntransformer konfigurieren
########################################################################################################################

# 3. Feature-Liste erstellen

header = feature_columns

rf_columns = [item for item in header if 'RF_' in item]
# rf_columns.remove('RF_aufgabenstellung_text')

# Definiert die describe-Werte die nicht mitaufgenommen werden sollen.
# describe_excluded = [a + b for a in ['describe_'] for b in ['count', 'max', 'min', '25', '50', '75', 'std']]

#
string_len_columns = [item for item in header if
                      ('string_len' in item)]  # and not any(substring in item for substring in describe_excluded))]

#
word_count_columns = [item for item in header if
                      ('word_count' in item)]  # and not any(substring in item for substring in describe_excluded))]
#
keystroke_columns = [item for item in header if ('keystrokes' in item and 'timestamp' not in item)]

# Anzahl der copy- und paste-events. _len und _count
# 'XXX_string_len_describe_count' und 'XXX_word_count_describe_count' sind identisch
describe_count_columns = [item for item in header if '_count_describe_count' in item]

ai_columns = [item for item in header if ('ai' in item)]

# feature_columns = word_count_columns + keystroke_columns + describe_count_columns + rf_columns
# [col for col in header if col not in excluded_columns]

############################################################################################################

# other_excluded_columns = ['date_of_last_access', 'datetime', 'duration','RF_aufgabenstellung_text']
# excluded_columns = merge_columns + other_excluded_columns + describe_columns + rf_columns

# 4. Transformer-Liste erstellen
transformers = []
passthrough_columns = merge_columns
for col in feature_columns:
    if col in ai_columns:
        #transformers.append((f'scaler_{col}', RobustScaler(), [col]))
        if 'keystrokes' in col:
            transformers.append((f'scaler_{col}', MaxAbsScaler(), [col]))
        else:
            transformers.append((f'scaler_{col}', MaxAbsScaler(), [col]))
    elif col in describe_count_columns:
        transformers.append((f'scaler_{col}', MaxAbsScaler(), [col]))
    elif col in string_len_columns:
        if '_count' in col:
            transformers.append((f'scaler_{col}', MaxAbsScaler(), [col]))
        else:
            transformers.append((f'scaler_{col}', MaxAbsScaler(), [col]))
    # elif col in word_count_columns:
    #     if '_std' in col:
    #         transformers.append((f'scaler_{col}', RobustScaler(), [col]))
    #     else:
    #         transformers.append((f'scaler_{col}', RobustScaler(), [col]))
    elif col in keystroke_columns:
        transformers.append((f'scaler_{col}', StandardScaler(), [col]))
    elif col in rf_columns:
        passthrough_columns.append(col)
    else:
        print(f'{col} hat keinen transformer zugwiesen bekommen ')

transformers.append(('passthrough', 'passthrough', passthrough_columns))

# Definiere den ColumnTransformer, wobei die merge_columns durchgereicht werden
preprocessor = ColumnTransformer(transformers=transformers)


########################################################################################################################
# 2. Transformer zum zusammenführen und skalieren der DFs
########################################################################################################################


# Custom Transformer für das Merging nach den ersten beiden Buchstaben des Index
class GroupMerger(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        grouped_dfs = {}
        for name, df in X.items():
            group_key = name[:2]  # Nutze die ersten zwei Buchstaben als Schlüssel
            if group_key not in grouped_dfs:
                grouped_dfs[group_key] = []
            grouped_dfs[group_key].append(df)

        # Kombiniere die DataFrames in jeder Gruppe (vertikal)
        for key in grouped_dfs:
            grouped_dfs[key] = pd.concat(grouped_dfs[key], axis=0, ignore_index=True)
        # print('GroupMerger returned:')
        # [print(f'{name}:\t{len(df)} rows x {len(df.columns)} columns') for name, df in grouped_dfs.items()]
        return grouped_dfs


# Custom Transformer zum Zusammenführen der Gruppen ME und EC
class GroupCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, group1, group2, new_group_name):
        self.group1 = group1  # Name der ersten Gruppe (z.B. 'ME')
        self.group2 = group2  # Name der zweiten Gruppe (z.B. 'EC')
        self.new_group_name = new_group_name  # Name der neuen kombinierten Gruppe

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        combined_groups = {}

        # Falls beide Gruppen existieren, werden sie kombiniert
        if self.group1 in X and self.group2 in X:
            # Kombiniere die beiden Gruppen vertikal
            combined_df = pd.concat([X[self.group1], X[self.group2]], axis=0, ignore_index=True)
            combined_groups[self.new_group_name] = combined_df

            # Entferne die alten Gruppen
            del X[self.group1]
            del X[self.group2]

        # Füge die restlichen Gruppen hinzu
        combined_groups.update(X)

        # print('GroupCombiner returned:')
        # [print(f'{name}:\t{len(df)} rows x {len(df.columns)} columns') for name, df in combined_groups.items()]
        return combined_groups


# Custom Transformer zum Anwenden des ColumnTransformers pro Gruppe
class DictionaryScaler(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def fit(self, X, y=None):
        # Fit auf jedem DataFrame (Gruppe) in X
        for key, df in X.items():
            self.preprocessor.fit(df)
        return self

    def transform(self, X):
        scaled_groups = {}
        for key, df in X.items():
            # Transformiere die Daten
            transformed_data = self.preprocessor.transform(df)

            # Hole die Spaltennamen nach der Transformation
            feature_names = self.preprocessor.get_feature_names_out()

            # Entferne doppelte Präfixe: Nur den eigentlichen Spaltennamen behalten
            cleaned_feature_names = [name.split('__')[-1] for name in feature_names]

            # Erstelle einen DataFrame mit den bereinigten Spaltennamen
            scaled_part = pd.DataFrame(transformed_data, columns=cleaned_feature_names)

            scaled_groups[key] = scaled_part
        return scaled_groups


# Custom Transformer zum Zusammenführen von doppelten UserIDs
# Custom Transformer zum Zusammenführen von doppelten UserIDs
class DuplicateUserMerger(BaseEstimator, TransformerMixin):
    def __init__(self, id_column, merge_columns):
        self.id_column = id_column  # Die Spalte, in der UserIDs gespeichert sind
        self.merge_columns = merge_columns  # Spalten, die zu Listen zusammengeführt werden sollen

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        merged_groups = {}
        for key, df in X.items():
            # Identifiziere doppelte UserIDs und gruppiere nach der UserID
            grouped_df = df.groupby(self.id_column).agg(
                {
                    **{
                        col: lambda x: self._merge_strings_or_lists(x) for col in self.merge_columns
                        # Listen für merge_columns
                    },
                    **{
                        col: 'mean' for col in df.columns if col not in self.merge_columns and col != self.id_column
                        # Mittelwert für numerische Spalten
                    }
                }
            ).reset_index()

            merged_groups[key] = grouped_df

        # print('DuplicateUserMerger returned:')
        # [print(f'{name}:\t{len(df)} rows x {len(df.columns)} columns') for name, df in merged_groups.items()]
        return merged_groups

    def _merge_strings_or_lists(self, series):
        """
        Hilfsfunktion zum Zusammenführen von Strings und/oder Listen:
        - Falls nur Strings vorhanden sind, werden sie in eine Liste umgewandelt und zusammengeführt.
        - Falls Listen und Strings gemischt sind, wird alles in eine Liste zusammengeführt.
        - Falls schon Listen vorhanden sind, werden sie kombiniert.
        """
        merged = []
        for item in series:
            if isinstance(item, list):
                merged.extend(item)  # Füge Listenelemente zur Liste hinzu
            else:
                merged.append(item)  # Füge den String zur Liste hinzu
        return merged  # Optional: `set(merged)` verwenden, um Duplikate zu entfernen


# Custom Transformer zum horizontalen Zusammenführen von DO und GE
class HorizontalMerger(BaseEstimator, TransformerMixin):
    def __init__(self, group1, group2, id_column, merge_columns, suffixes=('_DO', '_GE')):
        self.group1 = group1  # Name der ersten Gruppe (z.B. 'DO')
        self.group2 = group2  # Name der zweiten Gruppe (z.B. 'GE')
        self.id_column = id_column  # Die Spalte, in der UserIDs gespeichert sind
        self.merge_columns = merge_columns  # Spalten wie 'sessionID' und 'AssessmentPhase', die in beiden vorkommen
        self.suffixes = suffixes  # Suffixe für die Spalten

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Überprüfe, ob beide Gruppen existieren
        if self.group1 in X and self.group2 in X:
            # Merge der beiden Gruppen über UserId
            merged_df = pd.merge(
                X[self.group1], X[self.group2],
                on=self.id_column,
                how='outer',  # Outer Join, damit alle UserIDs aus beiden Gruppen erhalten bleiben
                suffixes=self.suffixes  # Suffixe für Spaltenkonflikte
            )

            # print('HorizontalMerger returned:')
            # print(f'ALL:\t{len(merged_df)} rows x {len(merged_df.columns)} columns')
            return merged_df
        else:
            raise ValueError(f"Gruppen {self.group1} und {self.group2} müssen beide vorhanden sein.")


# Custom Transformer zur Berechnung des Mittelwerts der unbekannten Spalten
class SimilarityImputer(BaseEstimator, TransformerMixin):
    def __init__(self, suffix_do='_DO', suffix_ge='_GE', merge_columns=[], k=5):
        self.suffix_do = suffix_do  # Suffix für DO-Spalten
        self.suffix_ge = suffix_ge  # Suffix für GE-Spalten
        self.merge_columns = merge_columns  # Spalten, die ausgeschlossen werden sollen
        self.k = k  # Anzahl der nächsten Nachbarn

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Kopiere das Original-DataFrame, um es nicht zu verändern
        X_filled = X.copy()

        # Identifiziere alle DO- und GE-Spalten basierend auf den Suffixen
        do_columns = [col for col in X_filled.columns if col.endswith(self.suffix_do)]
        ge_columns = [col for col in X_filled.columns if col.endswith(self.suffix_ge)]

        # Entferne merge_columns, die in den Spaltennamen vorkommen, da sie nicht-numerische Daten enthalten
        do_columns = [col for col in do_columns if not any(merge_col in col for merge_col in self.merge_columns)]
        ge_columns = [col for col in ge_columns if not any(merge_col in col for merge_col in self.merge_columns)]

        # Kombiniere die DO- und GE-Spalten zur Berechnung der Ähnlichkeit
        known_columns = do_columns + ge_columns

        # DataFrame mit bekannten Spalten (nur numerische Werte) erstellen
        known_df = X_filled[known_columns].dropna()

        # KNN zur Berechnung der Ähnlichkeit basierend auf allen bekannten Spalten
        nn = NearestNeighbors(n_neighbors=self.k)
        nn.fit(known_df)

        # Iteriere über die Benutzer, die fehlende Werte in DO- oder GE-Spalten haben
        for i in X_filled.index:
            # Überprüfe, ob der Benutzer fehlende Werte in DO- oder GE-Spalten hat
            missing_do = X_filled.loc[i, do_columns].isnull().any()
            missing_ge = X_filled.loc[i, ge_columns].isnull().any()

            if missing_do or missing_ge:
                # Bereinige die Zeile von NaN-Werten: Ersetze NaN-Werte durch den Spaltenmittelwert
                row_values = X_filled.loc[i, known_columns].fillna(X_filled[known_columns].mean()).to_frame().T

                # Finde die ähnlichsten Benutzer basierend auf allen DO- und GE-Spalten
                distances, neighbors = nn.kneighbors(row_values, return_distance=True)
                similar_users = X_filled.iloc[neighbors.flatten()]

                # Mittelwert für fehlende DO-Spalten berechnen
                if missing_do:
                    for col in do_columns:
                        if pd.isnull(X_filled.loc[i, col]):
                            X_filled.loc[i, col] = similar_users[col].mean()

                # Mittelwert für fehlende GE-Spalten berechnen
                if missing_ge:
                    for col in ge_columns:
                        if pd.isnull(X_filled.loc[i, col]):
                            X_filled.loc[i, col] = similar_users[col].mean()

        return X_filled


# Custom Transformer, der aus einem Dictionary mit einem DataFrame nur den DataFrame zurückgibt
class DictionaryToDataFrame(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, dict):
            # Verifiziere, dass das Dictionary genau einen DataFrame enthält
            if len(X) == 1 and isinstance(list(X.values())[0], pd.DataFrame):
                # Gib den DataFrame zurück
                return list(X.values())[0]
            else:
                raise ValueError("Das Dictionary muss genau einen DataFrame enthalten.")
        else:
            raise ValueError("Eingabe muss ein Dictionary mit einem DataFrame sein.")


pipeline_to_ge_and_do_merged_horizontal = Pipeline([
    ('dictonary_scaler', DictionaryScaler(preprocessor)),
    ('group_merger', GroupMerger()),
    ('group_scaler', DictionaryScaler(preprocessor)),
    ('duplicate_user_merger',
     DuplicateUserMerger(id_column='UserId', merge_columns=['AssessmentPhase', 'sessionID'] + rf_columns)),
    ('group_combiner', GroupCombiner(group1='ME', group2='EC', new_group_name='DO')),
    ('group_scaler2', DictionaryScaler(preprocessor)),
    ('horizontal_merger', HorizontalMerger(group1='DO', group2='GE', id_column='UserId',
                                           merge_columns=['AssessmentPhase', 'sessionID'] + rf_columns)),
    ('similarity_imputer', SimilarityImputer(suffix_do='_DO', suffix_ge='_GE', merge_columns=merge_columns, k=2))
])

pipeline_to_ge_and_do_merged_vertical = Pipeline(
    pipeline_to_ge_and_do_merged_horizontal.steps[0:6] +
    [
        ('group_combiner2', GroupCombiner(group1='DO', group2='GE', new_group_name='ALL')),
        ('group_scaler3', DictionaryScaler(preprocessor)),
        ('duplicate_user_merger2',
         DuplicateUserMerger(id_column='UserId', merge_columns=['AssessmentPhase', 'sessionID'] + rf_columns)),
        ('group_scaler4', DictionaryScaler(preprocessor)),
        ('dictonary_to_dataframe', DictionaryToDataFrame())
    ])

pipeline_to_ge_and_ec_and_me = Pipeline([
    ('dictonary_scaler', DictionaryScaler(preprocessor)),
    ('group_merger', GroupMerger()),
    ('group_scaler', DictionaryScaler(preprocessor)),
    ('duplicate_user_merger',
     DuplicateUserMerger(id_column='UserId', merge_columns=['AssessmentPhase', 'sessionID'] + rf_columns)),
    ('group_scaler5', DictionaryScaler(preprocessor))
])

pipeline_to_ge_and_ec_and_me_one_scaling = Pipeline([
    ('group_merger', GroupMerger()),
    ('duplicate_user_merger',
     DuplicateUserMerger(id_column='UserId', merge_columns=['AssessmentPhase', 'sessionID'] + rf_columns)),
    ('group_scaler5', DictionaryScaler(preprocessor))
])

pipeline_to_ge_and_ec_and_me_one_scaling_begin = Pipeline([
    ('dictonary_scaler', DictionaryScaler(preprocessor)),
    ('group_merger', GroupMerger()),
    ('duplicate_user_merger',
     DuplicateUserMerger(id_column='UserId', merge_columns=['AssessmentPhase', 'sessionID'] + rf_columns))
])

pipeline_vertical_one_scaling = Pipeline([
    ('group_merger', GroupMerger()),
    ('group_combiner', GroupCombiner(group1='ME', group2='EC', new_group_name='DO')),
    ('group_combiner2', GroupCombiner(group1='DO', group2='GE', new_group_name='ALL')),
    ('duplicate_user_merger',
     DuplicateUserMerger(id_column='UserId', merge_columns=['AssessmentPhase', 'sessionID'] + rf_columns)),
    ('group_scaler', DictionaryScaler(preprocessor)),
    ('dictonary_to_dataframe', DictionaryToDataFrame())
])

pipeline_vertical_one_scaling_begin = Pipeline([
    ('dictonary_scaler', DictionaryScaler(preprocessor)),
    ('group_merger', GroupMerger()),
    ('group_combiner', GroupCombiner(group1='ME', group2='EC', new_group_name='DO')),
    ('group_combiner2', GroupCombiner(group1='DO', group2='GE', new_group_name='ALL')),
    ('duplicate_user_merger',
     DuplicateUserMerger(id_column='UserId', merge_columns=['AssessmentPhase', 'sessionID'] + rf_columns)),
    ('dictonary_to_dataframe', DictionaryToDataFrame())
])

pipeline_vertical_with_duplicates_dictscaled = Pipeline([
    ('dictonary_scaler', DictionaryScaler(preprocessor)),
    ('group_merger', GroupMerger()),
    ('group_combiner', GroupCombiner(group1='ME', group2='EC', new_group_name='DO')),
    ('group_combiner2', GroupCombiner(group1='DO', group2='GE', new_group_name='ALL')),
    ('dictonary_to_dataframe', DictionaryToDataFrame())
])
pipeline_vertical_with_duplicates_groupscaled = Pipeline([
    ('group_merger', GroupMerger()),

    ('group_combiner', GroupCombiner(group1='ME', group2='EC', new_group_name='DO')),
    ('dictonary_scaler', DictionaryScaler(preprocessor)),
    ('group_combiner2', GroupCombiner(group1='DO', group2='GE', new_group_name='ALL')),
    ('dictonary_to_dataframe', DictionaryToDataFrame())
])

pipeline_to_ge_and_ec_and_me_without_user_merger = Pipeline([
    ('group_merger', GroupMerger()),
    ('group_scaler', DictionaryScaler(preprocessor))
])

pipeline_to_questionaries = Pipeline([
    ('dictonary_scaler', DictionaryScaler(preprocessor)),
])

pipeline_vertical_with_duplicates_notscaled = Pipeline([
    ('group_merger', GroupMerger()),
    ('group_combiner', GroupCombiner(group1='ME', group2='EC', new_group_name='DO')),
    ('group_combiner2', GroupCombiner(group1='DO', group2='GE', new_group_name='ALL')),
    ('dictonary_to_dataframe', DictionaryToDataFrame())
])

########################################################################################################################
# 3. Clustering
########################################################################################################################



# Definiere die Spalten, die ausgeschlossen werden sollen
merge_columns = ['AssessmentPhase', 'UserId', 'sessionID']
merge_columns_2 = ['UserId', 'AssessmentPhase_DO', 'AssessmentPhase_GE', 'sessionID_DO', 'sessionID_GE']

# groups_dfs_with_duplicates = pipeline_to_ge_and_ec_and_me_without_user_merger.fit_transform(dfs)


# Deine DataFrame-Auswahl
chosen_dfs_index = {
    # 'merged_horizontal': pipeline_to_ge_and_do_merged_horizontal.fit_transform(dfs),
    # 'merged_vertical': pipeline_to_ge_and_do_merged_vertical.fit_transform(dfs),
    # 'merged_vertical_scaled_once': pipeline_vertical_one_scaling.fit_transform(dfs),
    # 'merged_vertical_scaled_once_begin': pipeline_vertical_one_scaling_begin.fit_transform(dfs),
    # "groups_dfs['EC']": pipeline_to_ge_and_ec_and_me.fit_transform(dfs)['EC'],
    # "groups_scaled_once_dfs['EC']": pipeline_to_ge_and_ec_and_me_one_scaling.fit_transform(dfs)['EC'],
    # "groups_scaled_once_begin_dfs['EC']": pipeline_to_ge_and_ec_and_me_one_scaling_begin.fit_transform(dfs)['EC'],
    # "groups_dfs['GE']": pipeline_to_ge_and_ec_and_me.fit_transform(dfs)['GE'],
    # "groups_scaled_once_dfs['GE']": pipeline_to_ge_and_ec_and_me_one_scaling.fit_transform(dfs)['GE'],
    # "groups_scaled_once_begin_dfs['GE']": pipeline_to_ge_and_ec_and_me_one_scaling_begin.fit_transform(dfs)['GE'],
    # "groups_dfs['ME']": pipeline_to_ge_and_ec_and_me.fit_transform(dfs)['ME'],
    # "groups_scaled_once_dfs['ME']": pipeline_to_ge_and_ec_and_me_one_scaling.fit_transform(dfs)['ME'],
    # "groups_scaled_once_begin_dfs['ME']": pipeline_to_ge_and_ec_and_me_one_scaling_begin.fit_transform(dfs)['ME'],
    # "questionaries_dfs['EC_NU']": questionaries_dfs['EC_NU'],
    # "questionaries_dfs['GEN-COR_FM']": questionaries_dfs['GEN-COR_FM'],
    # "questionaries_dfs['ME_AT']": questionaries_dfs['ME_AT'],
    # "groups_dfs_with_duplicates['EC']": pipeline_to_ge_and_ec_and_me_without_user_merger.fit_transform(dfs)['EC'],
    # "groups_dfs_with_duplicates['ME']": pipeline_to_ge_and_ec_and_me_without_user_merger.fit_transform(dfs)['ME'],
    # "groups_dfs_with_duplicates['GE']": pipeline_to_ge_and_ec_and_me_without_user_merger.fit_transform(dfs)['GE'],
    "all_with_duplicates_groupscaled": pipeline_vertical_with_duplicates_groupscaled.fit_transform(dfs),
    # "vertical_with_duplicates_dictscaled": vpipeline_vertical_with_duplicates_dictscaled.fit_transform(dfs),

    # Dynamically added elements
    # **{key: value for key, value in pipeline_to_questionaries.fit_transform(dfs).items()}
}

# Cluster Algorithmen
algorithms = {
    'KMeans0': KMeans(random_state=42),
    # 'KMeans1': KMeans(random_state=42, init='k-means++', algorithm='lloyd'),  # default konfig full=lloyd beste k=2
    'KMeans2': KMeans(random_state=42, init='k-means++', algorithm='elkan'), #dasselbe

    'GMM1': GaussianMixture(random_state=42, init_params='kmeans', covariance_type='full'),
    'GMM2': GaussianMixture(random_state=42, init_params='kmeans', covariance_type='tied'),
    'GMM3': GaussianMixture(random_state=42, init_params='kmeans', covariance_type='diag'),
    'GMM4': GaussianMixture(random_state=42, init_params='kmeans', covariance_type='spherical'),
    'GMM5': GaussianMixture(random_state=42, init_params='kmeans', covariance_type='full', n_init=5),
    'GMM6': GaussianMixture(random_state=42, init_params='random', covariance_type='full'),
    'GMM7': GaussianMixture(random_state=42, init_params='kmeans', covariance_type='tied', max_iter=500),
    'GMM8': GaussianMixture(random_state=42, init_params='kmeans', covariance_type='diag', tol=1e-5),
    'GMM9': GaussianMixture(random_state=42, init_params='kmeans', covariance_type='spherical', n_components=3),
    'GMM10': GaussianMixture(random_state=42, init_params='kmeans', covariance_type='spherical', n_components=2),
    'GMM11': GaussianMixture(random_state=42, init_params='kmeans', covariance_type='full', reg_covar=1e-6),
    'GMM13': GaussianMixture(random_state=42, init_params='k-means++', covariance_type='full'),
    'GMM14': GaussianMixture(random_state=42, init_params='k-means++', covariance_type='tied'),
    'GMM15': GaussianMixture(random_state=42, init_params='k-means++', covariance_type='diag',  reg_covar=1e-4), #  reg_covar=1e-4 wegen schlechter definierter kovarianzenmatritzen
    'GMM16': GaussianMixture(random_state=42, init_params='k-means++', covariance_type='spherical',  reg_covar=1e-4), # beste ohne #  reg_covar=1e-4 wegen schlechter definierter kovarianzenmatritzen
    'GMM17': GaussianMixture(random_state=42, init_params='k-means++', covariance_type='full', n_init=5),
    'GMM19': GaussianMixture(random_state=42, init_params='k-means++', covariance_type='tied', max_iter=500),
    'GMM20': GaussianMixture(random_state=42, init_params='k-means++', covariance_type='diag', tol=1e-5,  reg_covar=1e-4),
    'GMM21': GaussianMixture(random_state=42, init_params='k-means++', covariance_type='spherical', n_components=3,  reg_covar=1e-4), #  reg_covar=1e-4 wegen schlechter definierter kovarianzenmatritzen
    'GMM22': GaussianMixture(random_state=42, init_params='k-means++', covariance_type='spherical', n_components=2,  reg_covar=1e-4), #  reg_covar=1e-4 wegen schlechter definierter kovarianzenmatritzen
    'GMM23': GaussianMixture(random_state=42, init_params='k-means++', covariance_type='full'),
    'GMM24': GaussianMixture(random_state=42, init_params='random_from_data', covariance_type='full',  reg_covar=1e-4), #  reg_covar=1e-4 wegen schlechter definierter kovarianzenmatritzen
    'GMM25': GaussianMixture(random_state=42, init_params='random_from_data', covariance_type='tied',  reg_covar=1e-4), #  reg_covar=1e-4 wegen schlechter definierter kovarianzenmatritzen
    'GMM26': GaussianMixture(random_state=42, init_params='random_from_data', covariance_type='diag',  reg_covar=1e-4), #  reg_covar=1e-4 wegen schlechter definierter kovarianzenmatritzen
    'GMM27': GaussianMixture(random_state=42, init_params='random_from_data', covariance_type='spherical',  reg_covar=1e-4), #  reg_covar=1e-4 wegen schlechter definierter kovarianzenmatritzen

    'AffinityPropagation1': AffinityPropagation(random_state=42, damping=0.8, preference=-10, max_iter=1000),
    'AffinityPropagation2': AffinityPropagation(random_state=42, damping=0.8, preference=-20, max_iter=1000),
    'AffinityPropagation3': AffinityPropagation(random_state=42, damping=0.8, preference=-40, max_iter=1000),
    'AffinityPropagation4': AffinityPropagation(random_state=42, damping=0.8, preference=-80, max_iter=1000),
    'AffinityPropagation5': AffinityPropagation(random_state=42, damping=0.8, preference=-200, max_iter=1000),
    'AffinityPropagation6': AffinityPropagation(random_state=42, damping=0.8, preference=-240, max_iter=1000),
    'AffinityPropagation7': AffinityPropagation(random_state=42, damping=0.8, preference=-280, max_iter=1000),
    'AffinityPropagation8': AffinityPropagation(random_state=42, damping=0.8, preference=-300, max_iter=1000),
    'AffinityPropagation9': AffinityPropagation(random_state=42, damping=0.95, preference=-100, max_iter=1000), ###################
    'AffinityPropagation10': AffinityPropagation(random_state=42, damping=0.95, preference=-20, max_iter=1000),
    'AffinityPropagation11': AffinityPropagation(random_state=42, damping=0.95, preference=-50, max_iter=1000),
    'AffinityPropagation12': AffinityPropagation(random_state=42, damping=0.95, preference=-160, max_iter=1000),
    'AffinityPropagation13': AffinityPropagation(random_state=42, damping=0.95, preference=-180, max_iter=1000),
    'AffinityPropagation14': AffinityPropagation(random_state=42, damping=0.95, preference=-200, max_iter=1000),
    'AffinityPropagation15': AffinityPropagation(random_state=42, damping=0.95, preference=-220, max_iter=1000),
    'AffinityPropagation16': AffinityPropagation(random_state=42, damping=0.95, preference=-300, max_iter=1000),
    'AffinityPropagation17': AffinityPropagation(random_state=42, damping=0.7, preference=-100, max_iter=1000),
    'AffinityPropagation18': AffinityPropagation(random_state=42, damping=0.7, preference=-120, max_iter=1000),
    'AffinityPropagation19': AffinityPropagation(random_state=42, damping=0.7, preference=-140, max_iter=1000),
    'AffinityPropagation20': AffinityPropagation(random_state=42, damping=0.7, preference=-160, max_iter=1000),
    'AffinityPropagation21': AffinityPropagation(random_state=42, damping=0.7, preference=-180, max_iter=1000),
    'AffinityPropagation22': AffinityPropagation(random_state=42, damping=0.7, preference=-200, max_iter=1000),
    'AffinityPropagation23': AffinityPropagation(random_state=42, damping=0.7, preference=-220, max_iter=1000),
    'AffinityPropagation24': AffinityPropagation(random_state=42, damping=0.7, preference=-240, max_iter=1000),
    'AffinityPropagation25': AffinityPropagation(random_state=42, damping=0.8, preference=-100, max_iter=1000), ###################
    'AffinityPropagation26': AffinityPropagation(random_state=42, damping=0.8, preference=-120, max_iter=1000),
    'AffinityPropagation27': AffinityPropagation(random_state=42, damping=0.8, preference=-140, max_iter=1000),
    'AffinityPropagation28': AffinityPropagation(random_state=42, damping=0.8, preference=-160, max_iter=1000),
    'AffinityPropagation29': AffinityPropagation(random_state=42, damping=0.8, preference=-180, max_iter=1000),
    'AffinityPropagation30': AffinityPropagation(random_state=42, damping=0.8, preference=-200, max_iter=1000),
    'AffinityPropagation31': AffinityPropagation(random_state=42, damping=0.8, preference=-220, max_iter=1000),
    'AffinityPropagation32': AffinityPropagation(random_state=42, damping=0.8, preference=-240, max_iter=1000),
    'AffinityPropagation33': AffinityPropagation(random_state=42, damping=0.9, preference=-100, max_iter=1000),
    'AffinityPropagation34': AffinityPropagation(random_state=42, damping=0.9, preference=-120, max_iter=1000),
    'AffinityPropagation35': AffinityPropagation(random_state=42, damping=0.9, preference=-140, max_iter=1000),
    'AffinityPropagation36': AffinityPropagation(random_state=42, damping=0.9, preference=-160, max_iter=1000),
    'AffinityPropagation37': AffinityPropagation(random_state=42, damping=0.9, preference=-180, max_iter=1000),
    'AffinityPropagation38': AffinityPropagation(random_state=42, damping=0.9, preference=-200, max_iter=1000),
    'AffinityPropagation39': AffinityPropagation(random_state=42, damping=0.9, preference=-220, max_iter=1000),
    'AffinityPropagation40': AffinityPropagation(random_state=42, damping=0.9, preference=-240, max_iter=1000),

    'MeanShift0': MeanShift(),

    'SpectralClustering0': SpectralClustering(random_state=42),
    'SpectralClustering1': SpectralClustering(assign_labels='kmeans', random_state=42, affinity='rbf'), # Standard-Einstellung mit RBF-Kernel
    'SpectralClustering2': SpectralClustering(assign_labels='kmeans', random_state=42, affinity='nearest_neighbors'),
    'SpectralClustering4': SpectralClustering(assign_labels='discretize', random_state=42, affinity='rbf'),
    'SpectralClustering5': SpectralClustering(assign_labels='kmeans', random_state=42, affinity='nearest_neighbors', n_neighbors=15),
    'SpectralClustering6': SpectralClustering(assign_labels='discretize', random_state=42, affinity='nearest_neighbors', n_neighbors=10),
    'SpectralClustering7': SpectralClustering(assign_labels='kmeans', random_state=42, affinity='cosine'), # bricht ab wegen inf oder nan? RuntimeWarning: invalid value encountered in sqrt w = np.where(isolated_node_mask, 1, np.sqrt(w))
    'SpectralClustering8': SpectralClustering(assign_labels='discretize', random_state=42, affinity='nearest_neighbors', n_neighbors=20),
    'SpectralClustering9': SpectralClustering(assign_labels='kmeans', random_state=42, affinity='nearest_neighbors', n_neighbors=5),
    'SpectralClustering10': SpectralClustering(assign_labels='discretize', random_state=42, affinity='cosine'), # bricht ab wegen inf oder nan? RuntimeWarning: invalid value encountered in sqrt w = np.where(isolated_node_mask, 1, np.sqrt(w))

    'DBSCAN1': DBSCAN(eps=1.3, min_samples=20),  # Mittelwert für stabilere Cluster
    'DBSCAN2': DBSCAN(eps=1.2, min_samples=18),  # Kleinere Cluster, aber robust
    'DBSCAN3': DBSCAN(eps=1.4, min_samples=22),  # Größeres eps für stabilere Cluster
    'DBSCAN4': DBSCAN(eps=1.5, min_samples=25),  # Mittelwert für größere Cluster
    'DBSCAN5': DBSCAN(eps=1.4, min_samples=15),  # Leicht reduziert für feinere Cluster
    'DBSCAN6': DBSCAN(eps=1.5, min_samples=18),
    'DBSCAN7': DBSCAN(eps=1.5, min_samples=15),  # Geringere Sample-Anzahl für kleinere Cluster
    'DBSCAN8': DBSCAN(eps=2.0, min_samples=12),  # Kleinere min_samples für mehr Cluster
    'DBSCAN9': DBSCAN(eps=1.8, min_samples=16),  # Feineres eps und reduzierte min_samples
    'DBSCAN10': DBSCAN(eps=1.6, min_samples=10),  # Sehr kleine Cluster möglich
    'DBSCAN11': DBSCAN(eps=1.8, min_samples=12),  # Kleinere Cluster für feinere Strukturen
    'DBSCAN12': DBSCAN(eps=1.7, min_samples=18),
    'DBSCAN13': DBSCAN(eps=1.8, min_samples=20, metric='euclidean'),  # Größere min_samples für robustere Cluster
    'DBSCAN14': DBSCAN(eps=1.9, min_samples=20, metric='chebyshev'),
    'DBSCAN15': DBSCAN(eps=1.8, min_samples=20, metric='minkowski', p=2),
    'DBSCAN16': DBSCAN(eps=1.8, min_samples=18, metric='manhattan'),
    'DBSCAN17': DBSCAN(eps=1.9, min_samples=25, algorithm='kd_tree'),  # Erhöhte min_samples für größere Cluster
    'DBSCAN18': DBSCAN(eps=1.8, min_samples=20, algorithm='ball_tree'),
    'DBSCAN19': DBSCAN(eps=1.2, min_samples=15),
    'DBSCAN20': DBSCAN(eps=1.1, min_samples=12),  # Feineres eps für kleinere Cluster
    'DBSCAN21': DBSCAN(eps=1.0, min_samples=10),  # Kleinere min_samples für mehr Cluster
    'DBSCAN22': DBSCAN(eps=0.9, min_samples=10),
    'DBSCAN23': DBSCAN(eps=0.8, min_samples=12),  # Geringere min_samples für mehr Cluster
    'DBSCAN24': DBSCAN(eps=0.7, min_samples=10),
    'DBSCAN25': DBSCAN(eps=0.7, min_samples=12),
    'DBSCAN26': DBSCAN(eps=0.7, min_samples=12, metric='manhattan'),
    'DBSCAN27': DBSCAN(eps=0.7, min_samples=12, metric='chebyshev'),
    'DBSCAN28': DBSCAN(eps=0.7, min_samples=12, metric='cosine'),
    'DBSCAN29': DBSCAN(eps=0.6, min_samples=10),  # Sehr fein für kleine Cluster
    'DBSCAN30': DBSCAN(eps=0.5, min_samples=12),

    'OPTICS1': OPTICS(min_samples=25),
    'OPTICS2': OPTICS(min_samples=30),#
    'OPTICS3': OPTICS(min_samples=20),#
    'OPTICS4': OPTICS(min_samples=50),
    'OPTICS5': OPTICS(min_samples=40),
    'OPTICS6': OPTICS(min_samples=35),
    'OPTICS7': OPTICS(min_samples=60),
    'OPTICS8': OPTICS(min_samples=10),#
    'OPTICS9': OPTICS(min_samples=15),#
    'OPTICS10': OPTICS(min_samples=45),
    'OPTICS11': OPTICS(min_samples=25, xi=0.05, min_cluster_size=0.1),
    'OPTICS12': OPTICS(min_samples=30, xi=0.03, min_cluster_size=0.05),
    'OPTICS13': OPTICS(min_samples=20, xi=0.04, min_cluster_size=0.2, metric='manhattan'),
    'OPTICS14': OPTICS(min_samples=50, xi=0.02, cluster_method='dbscan'),
    'OPTICS15': OPTICS(min_samples=40, xi=0.06, algorithm='ball_tree'),
    'OPTICS16': OPTICS(min_samples=35, xi=0.07, min_cluster_size=15),
    'OPTICS17': OPTICS(min_samples=60, xi=0.05, metric='chebyshev'),
    'OPTICS18': OPTICS(min_samples=10, xi=0.01, cluster_method='dbscan'),
    'OPTICS19': OPTICS(min_samples=15, xi=0.09, min_cluster_size=0.02),
    'OPTICS20': OPTICS(min_samples=45, xi=0.08, algorithm='kd_tree'),

    'HDBSCAN1': hdbscan.HDBSCAN(min_samples=5, min_cluster_size=10),
    'HDBSCAN2': hdbscan.HDBSCAN(min_samples=8, min_cluster_size=20),
    'HDBSCAN3': hdbscan.HDBSCAN(min_samples=9, min_cluster_size=50, cluster_selection_epsilon=0.1),
    'HDBSCAN4': hdbscan.HDBSCAN(min_samples=10, min_cluster_size=15), # ok
    'HDBSCAN5': hdbscan.HDBSCAN(min_samples=15, min_cluster_size=25), # ok
    'HDBSCAN6': hdbscan.HDBSCAN(min_samples=15, min_cluster_size=25, metric='euclidean'),
    'HDBSCAN7': hdbscan.HDBSCAN(min_samples=15, min_cluster_size=25, metric='manhattan'),
    'HDBSCAN8': hdbscan.HDBSCAN(min_samples=15, min_cluster_size=25, metric='chebyshev'),
    'HDBSCAN9': hdbscan.HDBSCAN(min_samples=15, min_cluster_size=40, metric='minkowski', p=3),
    'HDBSCAN10': hdbscan.HDBSCAN(min_samples=20, min_cluster_size=35),

    'WardHierarchical1': AgglomerativeClustering(linkage='ward'), ##################
    'WardHierarchical2': AgglomerativeClustering(linkage='ward', compute_full_tree=True), # dasselbe ergebnis
    'WardHierarchical3': AgglomerativeClustering(linkage='ward', compute_full_tree=False), # dasselbe ergebnis
    'WardHierarchical_none1': AgglomerativeClustering(linkage='ward', compute_full_tree=True, n_clusters=None, distance_threshold=20),
    'WardHierarchical_none2': AgglomerativeClustering(linkage='ward', compute_full_tree=True, n_clusters=None, distance_threshold=30),
    'WardHierarchical_none3': AgglomerativeClustering(linkage='ward', compute_full_tree=True, n_clusters=None, distance_threshold=40),
    'WardHierarchical_none4': AgglomerativeClustering(linkage='ward', compute_full_tree=True, n_clusters=None, distance_threshold=50),
    'WardHierarchical_none5': AgglomerativeClustering(linkage='ward', compute_full_tree=True, n_clusters=None, distance_threshold=60),
    'WardHierarchical_none6': AgglomerativeClustering(linkage='ward', compute_full_tree=True, n_clusters=None, distance_threshold=70),
    'AgglomerativeClustering2': AgglomerativeClustering(linkage='complete', metric='euclidean'), # mit k3 gute korrelation ########################
    'AgglomerativeClustering3': AgglomerativeClustering(linkage='average', metric='euclidean'), ##########################
    'AgglomerativeClustering4': AgglomerativeClustering(linkage='average', metric='manhattan'), ####################
    'AgglomerativeClustering5': AgglomerativeClustering(linkage='complete', metric='manhattan'),
    'AgglomerativeClustering6': AgglomerativeClustering(linkage='single', metric='euclidean'), ########################
    'AgglomerativeClustering7': AgglomerativeClustering(linkage='average', metric='cosine'),
    'AgglomerativeClustering8': AgglomerativeClustering(linkage='complete', metric='cosine'),
    'AgglomerativeClustering10': AgglomerativeClustering(linkage='single', metric='manhattan'),

    'Birch_none1': Birch(threshold=1.0, branching_factor=50, n_clusters=None),
    'Birch_none2': Birch(threshold=0.7, branching_factor=100, n_clusters=None), ##############
    'Birch_none3': Birch(threshold=1.5, branching_factor=150, n_clusters=None),
    'Birch_none4': Birch(threshold=0.8, branching_factor=150, n_clusters=None),
    'Birch_none5': Birch(threshold=1.2, branching_factor=100, n_clusters=None),
    'Birch6': Birch(threshold=0.3, branching_factor=50),
    'Birch7': Birch(threshold=0.5, branching_factor=50),
    'Birch8': Birch(threshold=0.5, branching_factor=100),
    'Birch9': Birch(threshold=0.2, branching_factor=50),
    'Birch10': Birch(threshold=0.8, branching_factor=50),
}

# Speichert das Programm, wie es ausgeführt wurde
script_code = inspect.getsource(inspect.stack()[0][0])
script_output_path = os.path.join(output_base, 'script_used.py')
with open(script_output_path, 'w', encoding='utf-8') as script_file:
    script_file.write(script_code)

# ACHTUNG FÜR ANALYSE NUR MIT EINEM DF
df_not_scaled = pipeline_vertical_with_duplicates_notscaled.fit_transform(dfs)
df_not_scaled = process_rf_columns(df_not_scaled)

numerical_columns = [col for col in df_not_scaled.columns if col not in merge_columns + merge_columns_2 + rf_columns]

for col in ['keystrokes_ai', 'paste_unipark_string_len_describe_50']:
    df_not_scaled[f'{col}_is_zero'] = df_not_scaled[col].apply(lambda x: -1 if x == 0 else 1)

df_not_scaled.to_csv(os.path.join(output_base, 'df_not_scaled.csv'), index=False)
print(f'Removed rows saved to "df_not_scaled.csv".')

results_all = []
counter = 0
for key, chosen_df in chosen_dfs_index.items():
    print(f'Processing: {key}')
    counter += 1
    print(f'Datenpunkte: {len(chosen_df)}')

    # Definiere die numerischen Spalten
    numerical_columns = [col for col in chosen_df.columns if col not in merge_columns + merge_columns_2 + rf_columns]

    chosen_df_numerical = chosen_df[numerical_columns].copy()
    for col in ['keystrokes_ai', 'paste_unipark_string_len_describe_50']:
        chosen_df_numerical[f'{col}_is_zero'] = chosen_df_numerical[col].apply(lambda x: -1 if x == 0 else 1)
    feature_columns = chosen_df_numerical.columns
    #chosen_df_numerical = PCA(n_components=5).fit_transform(chosen_df_numerical)
    # chosen_df_numerical = (umap.UMAP(n_components=5, random_state=42)).fit_transform(chosen_df_numerical)

    # Clusterbereich definieren
    k_values = range(2, 20)  # auf 2 bis 10 Cluster limitiert

    output_dir = os.path.join(output_base, key)
    os.makedirs(output_dir, exist_ok=True)
    chosen_df_to_save_act = chosen_df.copy()

    cluster_columns_for_heatmap = []
    results_file = []
    for algorithm_name, algorithm in algorithms.items():
        print(f'### {algorithm_name} ###')

        for idx, k in enumerate(k_values):
            # Auswahl für analyse
            if not (
                    # ("KMeans0" == algorithm_name and k == 7) or
                    # ("Birch10" == algorithm_name and k == 12) or
                    # ("SpectralClustering4" == algorithm_name and k == 11) or
                    # ("SpectralClustering4" == algorithm_name and k == 9) or
                    # ("SpectralClustering4" == algorithm_name and k == 8) or
                    # ("AffinityPropagation24" == algorithm_name) or
                    # ("SpectralClustering4" == algorithm_name and k == 10) or
                    # ("SpectralClustering4" == algorithm_name and k == 18) or
                    # ("KMeans0" == algorithm_name and k == 10) or
                    # ("KMeans0" == algorithm_name and k == 9) or
                    # ("SpectralClustering4" == algorithm_name and k == 6) or
                    # ("Birch8" == algorithm_name and k == 17) or
                    # ("SpectralClustering4" == algorithm_name and k == 12) or
                    # ("KMeans0" == algorithm_name and k == 8) or
                    # ("Birch8" == algorithm_name and k == 18) or
                    # ("SpectralClustering4" == algorithm_name and k == 5) or
                    # ("AffinityPropagation14" == algorithm_name) or
                    ("AffinityPropagation11" == algorithm_name)# or
                    # ("AffinityPropagation33" == algorithm_name)
            ):
                continue

            try:
                chosen_df_numerical_copy = chosen_df_numerical.copy()

                # Algorithmen ##############################################################################################
                # Diese Algorithmen benötigen keine Clusteranzahl
                if any(algorithm_name.startswith(prefix) for prefix in
                       ['OPTICS', 'DBSCAN', 'HDBSCAN', 'AffinityPropagation', 'WardHierarchical_none', 'Birch_none']):
                    if idx > 0:  # Verwende nur den ersten Plot für diese Algorithmen
                        continue
                    cluster_labels = algorithm.fit_predict(chosen_df_numerical_copy)

                    # Überprüfe, ob der Algorithmus mehr als 1 Cluster gefunden hat
                    if len(set(cluster_labels)) <= 1:  # Weniger als 2 Cluster
                        print(
                            f"{algorithm_name} hat nur einen oder keinen Cluster gefunden. Überspringe Silhouettenanalyse.")
                        continue

                    num_clusters = len(set(cluster_labels)) - (
                        1 if -1 in cluster_labels else 0)  # Berücksichtige Rauschen (-1)
                    k = num_clusters
                    # if algorithm_name.startswith('OPTICS'):
                    #     # Reachability Plot
                    #     reachability = algorithm.reachability_[algorithm.ordering_]
                    #     plt.figure(figsize=(10, 6))
                    #     plt.plot(reachability)
                    #     plt.title('OPTICS Reachability Plot')
                    #     plt.ylabel('Reachability Distance')
                    #     plt.xlabel('Points ordered by OPTICS')
                    #     plt.show()

                elif algorithm_name.startswith('MeanShift'):
                    if idx > 0:  # Verwende nur den ersten Plot für diese Algorithmen
                        continue
                    bandwidth = estimate_bandwidth(chosen_df_numerical_copy, quantile=0.6)
                    cluster_labels = algorithm.fit_predict(chosen_df_numerical_copy)

                    # Überprüfe, ob der Algorithmus mehr als 1 Cluster gefunden hat
                    if len(set(cluster_labels)) <= 1:  # Weniger als 2 Cluster
                        print(
                            f"{algorithm_name} hat nur einen oder keinen Cluster gefunden. Überspringe Silhouettenanalyse.")
                        continue
                    num_clusters = len(set(cluster_labels)) - (
                        1 if -1 in cluster_labels else 0)  # Berücksichtige Rauschen (-1)
                    k = num_clusters

                elif algorithm_name.startswith('GMM'):  # Gaussian Mixture benötigt n_components statt n_clusters
                    algorithm.set_params(n_components=k)
                    cluster_labels = algorithm.fit_predict(chosen_df_numerical_copy)

                else:  # Für KMeans, Agglomerative und andere, die eine Clusteranzahl erfordern
                    algorithm.set_params(n_clusters=k)
                    cluster_labels = algorithm.fit_predict(chosen_df_numerical_copy)

                # Plots ####################################################################################################
                if len(set(cluster_labels)) > 1:
                    silhouette_avg = silhouette_score(chosen_df_numerical_copy, cluster_labels)
                    sample_silhouette_values = silhouette_samples(chosen_df_numerical_copy, cluster_labels)

                    # Berechne Calinski-Harabasz-Index und Davies-Bouldin-Index
                    ch_score = calinski_harabasz_score(chosen_df_numerical_copy, cluster_labels)
                    db_score = davies_bouldin_score(chosen_df_numerical_copy, cluster_labels)

                    # Ausgabe der Ergebnisse
                    # print(f"Clusteranzahl: {k}")
                    # print(f"Durchschnittlicher Silhouettenwert: {silhouette_avg}")
                    # print(f"Calinski-Harabasz-Index: {ch_score}")
                    # print(f"Davies-Bouldin-Index: {db_score}")

                    # Speichere die Ergebnisse in der Liste
                    results_file.append({
                        'DF': key,
                        'Clusteranzahl': k,
                        'Algorithmus': algorithm_name,
                        'Silhouette Score': silhouette_avg,
                        'Calinski-Harabasz-Index': ch_score,
                        'Davies-Bouldin-Index': db_score
                    })
                else:
                    print(f"Nur ein Cluster gefunden für {algorithm_name} mit k={k}. Silhouettenanalyse übersprungen.")
                    continue

                # Reduziere die Dimensionen mit PCA
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(chosen_df_numerical_copy)

                # Speichere die Clustering-Ergebnisse als CSV
                # chosen_df_to_save_act[f'cluster_{algorithm_name}_k{k}'] = cluster_labels
                # Erstelle ein DataFrame für die neuen Cluster-Spalten
                new_clusters = pd.DataFrame({
                    f'cluster_{algorithm_name}_k{k}': cluster_labels
                })

                # Füge die neuen Cluster-Spalten gesammelt mit pd.concat hinzu
                chosen_df_to_save_act = pd.concat([chosen_df_to_save_act, new_clusters], axis=1)

                cluster_columns_for_heatmap.append(f'cluster_{algorithm_name}_k{k}')

                # Plotte und speichere die Silhouette und PCA Diagramme
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                colors = np.array(
                    list(
                        islice(
                            cycle(
                                [
                                    "#377eb8",
                                    "#ff7f00",
                                    "#4daf4a",
                                    "#f781bf",
                                    "#a65628",
                                    "#984ea3",
                                    "#999999",
                                    "#e41a1c",
                                    "#dede00",
                                    "#ffffb3",
                                    "#fb8072",
                                    "#80b1d3",
                                    "#fdb462",
                                    "#b3de69",
                                    "#fccde5",
                                    "#bc80bd",
                                    "#ccebc5",
                                    "#ffed6f",
                                    "#8dd3c7",
                                    "#bebada",
                                    "#fb9a99",
                                    "#e31a1c",
                                    "#6a3d9a",
                                ]
                            ),
                            int(max(cluster_labels) + 1),
                        )
                    )
                )
                # Füge Schwarz für das Rauschen hinzu
                colors = np.append(colors, ["#000000"])

                # Silhouette Plot
                ax1 = axes[0]
                ax1.set_xlim([-0.1, 1])
                ax1.set_ylim([0, len(chosen_df_numerical_copy) + (k + 1) * 10])

                y_lower = 10
                jump = False
                for i in range(k):
                    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    # color = cm.nipy_spectral(float(i) / k)
                    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=colors[i],
                                      edgecolor=colors[i], alpha=0.7)

                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                    y_lower = y_upper + 10

                ax1.set_title(f'Silhouette for {k} clusters')
                ax1.set_xlabel("Silhouette coefficient values")
                ax1.set_ylabel("Cluster label")
                ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
                ax1.set_yticks([])
                ax1.set_xticks(np.arange(-0.1, 1.1, 0.2))

                # PCA Plot
                ax2 = axes[1]

                # Scatterplot mit den Farben für Cluster und Schwarz für Rauschen
                plt.scatter(pca_result[:, 0], pca_result[:, 1], s=10, color=colors[cluster_labels])
                # ax2.scatter(pca_result[:, 0], pca_result[:, 1], s=30, lw=0, alpha=0.7, c=colors[cluster_labels], edgecolor='k')
                # if -1 in cluster_labels:
                #     colors = np.array([cm.nipy_spectral(float(label) / (k - 1)) if label != -1 else '#000000' for label in cluster_labels], dtype=object)
                # else:
                #     colors = cm.nipy_spectral(cluster_labels.astype(float) / k)

                # ax2.scatter(pca_result[:, 0], pca_result[:, 1], marker='o', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
                ax2.set_title(f'PCA for {k} clusters')
                ax2.set_xlabel("PCA Component 1")
                ax2.set_ylabel("PCA Component 2")

                # Speichern der Diagramme
                plt.suptitle(f'Silhouette and PCA for {algorithm_name} with k={k}', fontsize=16, fontweight='bold')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                # Zeige das Diagramm und speichere die Bilder
                silhouette_img_path = os.path.join(output_dir, f'{algorithm_name}_silhouette_pca_k{k}.png')
                fig.savefig(silhouette_img_path)

                # Zeige die Diagramme
                # plt.show()
                plt.close(fig)

                # Analyse
                df_scaled = chosen_df_numerical_copy
                df_not_scaled = df_not_scaled.copy()
                df_not_scaled['cluster'] = cluster_labels

                feature_columns = [col for col in feature_columns if col not in merge_columns + rf_columns]
                indicator_columns = [col for col in df_not_scaled.columns if col in rf_columns]

                assign_covariates_and_impute_participant_group(df_not_scaled, numerical_columns)
                c_columns = [col for col in df_not_scaled.columns if col.startswith("c_")]
                output_dir_analysen = os.path.join(output_dir, f'{algorithm_name}_{k}')
                # try:
                #     anova(df_not_scaled.copy(), feature_columns, output_dir_analysen)
                # except Exception as e:
                #     print(f"Fehler bei anova: {e}")
                #
                # try:
                #     ancova(df_not_scaled.copy(), feature_columns, output_dir_analysen)
                # except Exception as e:
                #     print(f"Fehler bei ancova: {e}")
                #
                # try:
                #     ancova_mit_transformationen(df_not_scaled.copy(), feature_columns, output_dir_analysen)
                # except Exception as e:
                #     print(f"Fehler bei ancova_mit_transformationen: {e}")
                #
                # try:
                #     rang_ancova(df_not_scaled.copy(), feature_columns, output_dir_analysen)
                # except Exception as e:
                #     print(f"Fehler bei rang_ancova: {e}")
                #
                # try:
                #     glmm_poisson(df_not_scaled.copy(), feature_columns, output_dir_analysen)
                # except Exception as e:
                #     print(f"Fehler bei glmm_poisson: {e}")
                #
                try:
                    permutation_test_without_covariate(df_not_scaled.copy(), feature_columns, output_dir_analysen)
                except Exception as e:
                    print(f"Fehler bei permutation_test_without_covariate: {e}")

                # try:
                #     permutation_test_with_covariate(df_not_scaled.copy(), feature_columns, output_dir_analysen)
                # except Exception as e:
                #     print(f"Fehler bei permutation_test_with_covariate: {e}")

                # try:
                #     permutation_test_with_and_without_covariate(df_not_scaled.copy(), feature_columns, output_dir_analysen)
                # except Exception as e:
                #     print(f"Fehler bei permutation_test_with_and_without_covariate: {e}")

                try:
                    chi_square_tests_and_plots(df_not_scaled.copy(), indicator_columns, output_dir_analysen)
                except Exception as e:
                    print(f"Fehler bei chi_square_tests_and_plots: {e}")

                try:
                    posthoc_dunn_test(df_not_scaled.copy(), feature_columns, output_dir_analysen)
                except Exception as e:
                    print(f"Fehler bei posthoc_dunn_test: {e}")

                # if k == 2:
                #     try:
                #         simple_stat_tests(df_not_scaled.copy(), feature_columns, output_dir_analysen)
                #     except Exception as e:
                #         print(f"Fehler bei simple_stat_tests: {e}")
            except Exception as e:
                print(f"Fehler bei {algorithm_name} k={k}: {e}")

    output_csv_path = os.path.join(output_dir, f'clustering_results_{key}.csv')
    chosen_df_to_save_act.to_csv(output_csv_path, index=False)
    results_all.extend(results_file)
    print(f"Ergebnisse für {algorithm_name} wurden gespeichert in: {output_dir}")
    # process_and_save_heatmaps(chosen_df_to_save_act, output_dir, cluster_columns_for_heatmap, key)

output_csv_path = os.path.join(output_base, f'results.csv')
print(f"Ergebnisse für Clusterevaluation gespeichert in: {output_base}")
pd.DataFrame(results_all).to_csv(output_csv_path, index=False)

#%%
