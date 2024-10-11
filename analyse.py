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

from process_rf_columns_and_generate_heatmaps import process_rf_columns, RF_COLUMNS, RF_COLUMNS_EINS_BIS_VIER, RF_ZEITVORGABE, RF_AUFGABENSTELLUNG, RF_NOTE
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import f_oneway, chi2_contingency, shapiro, levene, kruskal
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.stats import boxcox

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats import weightstats as stests
#from statsmodels.stats.contingency_tables import chi2_contingency
from scipy.stats import chi2_contingency
import scikit_posthocs as sp
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.model_selection import permutation_test_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


# Aktivieren der Konvertierung von pandas nach R-Datenframes
pandas2ri.activate()

# R-Bibliotheken laden
robjects.r('library(lme4)')

# Für Kmeans
os.environ["OMP_NUM_THREADS"] = "6"

# Warnungen ignorieren
warnings.filterwarnings("ignore", category=DeprecationWarning)



# Definieren einer Funktion, um die Teilnehmergruppe auf Basis der Fragebögen festzulegen
def assign_covariates_and_impute_participant_group(df_for_getting_categorical_columns, numerical_columns_for_update):
    """
    Diese Funktion weist die Dauer der AssessmentPhasen und die Teilnehmergruppen basierend auf den Fragebogendaten zu.
    Fehlende Teilnehmergruppen werden durch Abgleich mit CSV-Daten und anschließend mithilfe von KNN- oder RandomForest-Imputation ergänzt.
    """

    print('assign_covariates_and_impute_participant_group started.')

    # Mapping der AssessmentPhase zu Dauer in Minuten
    duration_mapping = {
        'Medizin Atmung': 60,
        'Medizin Auge': 60,
        'Medizin Kreislauf': 60,
        'Medizin Mittelohr': 60,
        'Nudging-Aufgabe': 60,
        'Piloten-Streik-Aufgabe': 60,
        'Start Up-Aufgabe': 60,
        'Startup-Aufgabe': 60,
        'Windpark-Aufgabe': 60,
        'Funkmast': 25,
        'Gruene Sosse': 25,
        'Hitzestift': 25,
        'Tetra Pak': 25
    }

    # Neue Spalte 'duration_c' erstellen
    df_for_getting_categorical_columns['c_Dauer'] = df_for_getting_categorical_columns['AssessmentPhase'].map(duration_mapping)

    fragebogen_mapping = {
        'Medizin Atmung': "ME_AT",
        'Medizin Auge': "ME_AU",
        'Medizin Kreislauf': "ME_KL",
        'Medizin Mittelohr': "ME_MO",
        'Nudging-Aufgabe': "EC_NU",
        'Piloten-Streik-Aufgabe': "EC_PS",
        'Start Up-Aufgabe': "EC_ST",
        'Startup-Aufgabe': "EC_ST",
        'Windpark-Aufgabe': "GEN_WP",
        'Funkmast': "GEN_FM",
        'Gruene Sosse': "GEN_GS",
        'Hitzestift': "GEN_HS",
        'Tetra Pak': "GEN_TP"
    }

    # Neue Spalte 'duration_c' erstellen
    df_for_getting_categorical_columns['c_Fragebogen'] = df_for_getting_categorical_columns['AssessmentPhase'].map(fragebogen_mapping)

    # Zuordnung der Fragebögen zu den Gruppen
    fragebogengruppen_mapping = {
        'Medizin Atmung': 'Medizin',
        'Medizin Auge': 'Medizin',
        'Medizin Kreislauf': 'Medizin',
        'Medizin Mittelohr': 'Medizin',
        'Nudging-Aufgabe': 'Medizin',
        'Piloten-Streik-Aufgabe': 'Oekonomie',
        'Start Up-Aufgabe': 'Oekonomie',
        'Startup-Aufgabe': 'Oekonomie',
        'Windpark-Aufgabe': 'Oekonomie',
        'Funkmast': 'Generisch',
        'Gruene Sosse': 'Generisch',
        'Hitzestift': 'Generisch',
        'Tetra Pak': 'Generisch',
        'Oekonomie Grundwissen': 'Oekonomie',
        'Fachtest Medizin': 'Medizin'
    }

    # 1. Erstellung der Spalte 'Fragebogengruppe'
    df_for_getting_categorical_columns['c_Fragebogengruppe'] = df_for_getting_categorical_columns['AssessmentPhase'].map(fragebogengruppen_mapping)

    # 2. Zuerst mappen wir die Fragebogengruppe auf Teilnehmergruppe
    # 'Generisch' wird zu None, während 'Medizin' und 'Oekonomie' behalten werden
    df_for_getting_categorical_columns['c_Teilnehmergruppe'] = df_for_getting_categorical_columns['c_Fragebogengruppe'].map({
        'Medizin': 'Medizin',
        'Oekonomie': 'Oekonomie',
        'Generisch': None
    })

    # 3. Erstellung eines Mappings von UserId zu Teilnehmergruppe
    for user_id in df_for_getting_categorical_columns['UserId'].unique():
        # Überprüfen, ob für die aktuelle UserId bereits eine Teilnehmergruppe zugewiesen wurde
        known_group = df_for_getting_categorical_columns.loc[df_for_getting_categorical_columns['UserId'] == user_id, 'c_Teilnehmergruppe'].dropna().unique()

        if len(known_group) > 0:
            # Wenn die Teilnehmergruppe bekannt ist, weisen wir sie für alle Zeilen mit dieser UserId zu
            df_for_getting_categorical_columns.loc[(df_for_getting_categorical_columns['UserId'] == user_id) & (df_for_getting_categorical_columns['c_Teilnehmergruppe'].isna()), 'c_Teilnehmergruppe'] = known_group[0]

    # 4. Fehlende Werte versuchen aus DB zu laden.
    attempts_df = pd.read_csv("list-attempts_filtered.csv")

    # 3. Überprüfung der None-Werte in 'Teilnehmergruppe' und Ergänzung basierend auf CSV-Daten
    for idx, row in df_for_getting_categorical_columns[df_for_getting_categorical_columns['c_Teilnehmergruppe'].isna()].iterrows():
        user_id = row['UserId']

        # Suchen der zugehörigen Phasen für die aktuelle UserId
        user_phases = attempts_df[attempts_df['session_user_userID'] == user_id]['phase_name'].unique()

        # Mapping der Phasen mit fragebogengruppen_mapping
        mapped_phases = [fragebogengruppen_mapping.get(phase_name, None) for phase_name in user_phases]

        # Überprüfen, ob einer der gemappten Werte 'Medizin' oder 'Oekonomie' ist
        if 'Medizin' in mapped_phases:
            df_for_getting_categorical_columns.at[idx, 'c_Teilnehmergruppe'] = 'Medizin'
        elif 'Oekonomie' in mapped_phases:
            df_for_getting_categorical_columns.at[idx, 'c_Teilnehmergruppe'] = 'Oekonomie'


    def knn_classifier_imputation(df_for_imputation, categorical_columns, numerical_columns_for_update, target_column_for_imputation, n_neighbors=5):
        """
        Imputation der fehlenden Werte in der Zielspalte 'target_column' mit KNeighborsClassifier.
        Wichtig: df_for_imputation mit skalierten numerischen Spalten

        Parameters:
        - df_for_imputation: pandas DataFrame (mit skalierten numerischen Spalten)
        - categorical_columns: Liste der kategorialen Spalten
        - target_column_for_imputation: Die Zielspalte, die imputiert werden soll
        - n_neighbors: Anzahl der Nachbarn für den KNN-Klassifikator (Standard: 5)

        Returns:
        - df_for_imputation: pandas DataFrame mit der imputierten Zielspalte
        """
        # a. Datensatz aufteilen: bekannte und unbekannte Zielwerte
        df_known = df_for_imputation[df_for_imputation[target_column_for_imputation].notna()]
        df_unknown = df_for_imputation[df_for_imputation[target_column_for_imputation].isna()]

        # b. One-Hot-Encoding für kategoriale Spalten
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_known_cat = encoder.fit_transform(df_known[categorical_columns])
        X_unknown_cat = encoder.transform(df_unknown[categorical_columns])

        # c. Keine zusätzliche Skalierung nötig, da die numerischen Spalten bereits skaliert sind
        X_known_num = df_known[numerical_columns_for_update].values
        X_unknown_num = df_unknown[numerical_columns_for_update].values

        # d. Kombinieren der numerischen und One-Hot-kodierten kategorialen Spalten
        X_known = np.hstack((X_known_cat, X_known_num))
        X_unknown = np.hstack((X_unknown_cat, X_unknown_num))

        y_known = df_known[target_column_for_imputation]

        # e. KNN-Klassifikator instanziieren und trainieren
        knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_classifier.fit(X_known, y_known)

        # f. Vorhersage der fehlenden Werte
        y_pred = knn_classifier.predict(X_unknown)

        # g. Vorhersagen in die ursprüngliche DataFrame schreiben
        df_for_imputation.loc[df_for_imputation[target_column_for_imputation].isna(), target_column_for_imputation] = y_pred

        return df_for_imputation[target_column_for_imputation]


    def random_forest_imputation(df_for_imputation, categorical_columns, numerical_columns_for_update, target_column_for_imputation, n_estimators=100):
        """
        Imputation der fehlenden Werte in der Zielspalte 'target_column' mit RandomForestClassifier.
        Wichtig: df_for_imputation mit skalierten numerischen Spalten

        Parameters:
        - df_for_imputation: pandas DataFrame (mit skalierten numerischen Spalten)
        - categorical_columns: Liste der kategorialen Spalten
        - target_column_for_imputation: Die Zielspalte, die imputiert werden soll
        - n_estimators: Anzahl der Bäume im Random Forest (Standard: 100)

        Returns:
        - df_for_imputation: pandas DataFrame mit der imputierten Zielspalte
        """
        # a. Datensatz aufteilen: bekannte und unbekannte Zielwerte
        df_known = df_for_imputation[df_for_imputation[target_column_for_imputation].notna()]
        df_unknown = df_for_imputation[df_for_imputation[target_column_for_imputation].isna()]

        # b. One-Hot-Encoding für kategoriale Spalten
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_known_cat = encoder.fit_transform(df_known[categorical_columns])
        X_unknown_cat = encoder.transform(df_unknown[categorical_columns])

        # c. Keine zusätzliche Skalierung nötig, da die numerischen Spalten bereits skaliert sind
        X_known_num = df_known[numerical_columns_for_update].values
        X_unknown_num = df_unknown[numerical_columns_for_update].values

        # d. Kombinieren der numerischen und One-Hot-kodierten kategorialen Spalten
        X_known = np.hstack((X_known_cat, X_known_num))
        X_unknown = np.hstack((X_unknown_cat, X_unknown_num))

        y_known = df_known[target_column_for_imputation]

        # e. RandomForestClassifier instanziieren und trainieren
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(X_known, y_known)

        # f. Vorhersage der fehlenden Werte
        y_pred = clf.predict(X_unknown)

        # g. Vorhersagen in die ursprüngliche DataFrame schreiben
        df_for_imputation.loc[df_for_imputation[target_column_for_imputation].isna(), target_column_for_imputation] = y_pred

        return df_for_imputation[target_column_for_imputation]


    # Es gibt User, die nur generische Fragebögen bearbeitet haben, nach einem Nachbessern über die DB sind noch 6 solcher User dabei. Bei 4 stimmen die beiden Methoden überein.
    categorical_columns = ['c_Fragebogengruppe', 'AssessmentPhase']
    target_column = 'c_Teilnehmergruppe'
    # 1. Imputation mit KNeighborsClassifier
    #df_for_getting_categorical_columns['Teilnehmergruppe'] = knn_classifier_imputation(df_for_getting_categorical_columns.copy(), categorical_columns, numerical_columns_for_update, target_column)
    # 2. Imputation mit RandomForestClassifier
    df_for_getting_categorical_columns['c_Teilnehmergruppe'] = random_forest_imputation(df_for_getting_categorical_columns.copy(), categorical_columns, numerical_columns_for_update, target_column)

    print('assign_covariates_and_impute_participant_group executed.')


def anova(df, feature_columns, output_path_base):
    print('anova started.')

    output_path = os.path.join(output_path_base, 'anova')
    # Prüfen, ob der Output-Ordner existiert, sonst erstellen
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Erstellen der Ausgabedatei für den Konsolenausgabe
    output_file = os.path.join(output_path, 'anova_output.txt')
    with open(output_file, 'w', encoding='utf-8') as f:

        p_values = []

        # Iteration durch alle Feature-Spalten
        for feature in feature_columns:
            f.write(f'\nFeature: {feature}\n')
            groups = [df[df['cluster'] == cluster][feature] for cluster in df['cluster'].unique()]

            # Test auf Normalverteilung (Shapiro-Wilk-Test)
            normality_p_values = []
            for i, group in enumerate(groups):
                stat, p = shapiro(group)
                normality_p_values.append(p)
                f.write(f'Cluster {i}: Shapiro-Wilk p-Wert = {p}\n')

            # Test auf Varianzhomogenität (Levene-Test)
            stat, levene_p = levene(*groups)
            f.write(f'Levene-Test p-Wert = {levene_p}\n')

            # Überprüfung der Voraussetzungen
            if all(p > 0.05 for p in normality_p_values) and levene_p > 0.05:
                # Voraussetzungen erfüllt, ANOVA durchführen
                F_statistic, p_value = f_oneway(*groups)
                f.write(f'ANOVA Ergebnis: F = {F_statistic}, p = {p_value}\n')
            else:
                # Voraussetzungen nicht erfüllt, Kruskal-Wallis-Test durchführen
                f.write('Voraussetzungen für ANOVA nicht erfüllt, Kruskal-Wallis-Test wird verwendet.\n')
                H_statistic, p_value = kruskal(*groups)
                f.write(f'Kruskal-Wallis-Test Ergebnis: H = {H_statistic}, p = {p_value}\n')

            p_values.append(p_value)

        # Anpassung der p-Werte für multiples Testen (Bonferroni-Korrektur)
        adjusted_p_values = multipletests(p_values, method='bonferroni')[1]
        for feature, adj_p in zip(feature_columns, adjusted_p_values):
            f.write(f'Angepasster p-Wert für {feature}: {adj_p}\n')

    # Visualisierung der Boxplots
    for feature in feature_columns:
        boxplot_path = os.path.join(output_path, f'boxplot_{feature}.png')
        sns.boxplot(x='cluster', y=feature, data=df)
        plt.title(f'Boxplot von {feature} nach Cluster')
        plt.savefig(boxplot_path)
        plt.close()

    print(f'Analyse abgeschlossen. Ergebnisse und Plots wurden im Ordner "{output_path}" gespeichert.')
    print('anova executed.')


def ancova(df, feature_columns, output_path_base):
    print('ancova started.')

    output_path = os.path.join(output_path_base, 'ancova')
    # Prüfen, ob der Output-Ordner existiert, sonst erstellen
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Erstellen der Ausgabedatei für den Konsolenausgabe
    output_file = os.path.join(output_path, 'ancova_output.txt')
    with open(output_file, 'w', encoding='utf-8') as f:

        # Iteration durch alle Feature-Spalten
        for feature in feature_columns:
            f.write(f'\nANCOVA für Feature: {feature}\n')

            # Modell mit 'cluster' als Faktor und den neuen Kovariaten
            model = smf.ols(f'{feature} ~ C(cluster) + c_Dauer + C(c_Teilnehmergruppe)', data=df).fit()
            table = sm.stats.anova_lm(model, typ=2)
            f.write(f'{table}\n')

            # Überprüfung der Interaktion zwischen 'cluster' und 'c_Dauer'
            model_interaction = smf.ols(f'{feature} ~ C(cluster) * c_Dauer + C(c_Teilnehmergruppe)', data=df).fit()
            interaction_table = sm.stats.anova_lm(model_interaction, typ=2)
            f.write('\nANOVA-Tabelle mit Interaktionsterm:\n')
            f.write(f'{interaction_table}\n')

            # Prüfen, ob die Interaktion signifikant ist
            if interaction_table.loc['C(cluster):c_Dauer', 'PR(>F)'] < 0.05:
                f.write('Warnung: Signifikante Interaktion zwischen Cluster und Dauer. Die Voraussetzung der Homogenität der Regressionskoeffizienten ist verletzt.\n')
            else:
                f.write('Keine signifikante Interaktion gefunden. Voraussetzung der Homogenität der Regressionskoeffizienten ist erfüllt.\n')

            # Residuen extrahieren
            residuals = model.resid

            # Normalverteilung der Residuen prüfen
            stat, p_value = shapiro(residuals)
            f.write(f'Shapiro-Wilk-Test für Residuen: p-Wert = {p_value}\n')
            if p_value < 0.05:
                f.write('Warnung: Residuen sind nicht normalverteilt.\n')
            else:
                f.write('Residuen sind normalverteilt.\n')

            # QQ-Plot der Residuen
            qq_plot_path = os.path.join(output_path, f'qq_plot_{feature}.png')
            sm.qqplot(residuals, line='45')
            plt.title(f'QQ-Plot der Residuen für {feature}')
            plt.savefig(qq_plot_path)
            plt.close()

            # Homoskedastizität prüfen (Grafisch)
            residuals_vs_fitted_path = os.path.join(output_path, f'residuen_vs_fitted_{feature}.png')
            plt.scatter(model.fittedvalues, residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Vorhergesagte Werte')
            plt.ylabel('Residuen')
            plt.title(f'Residuen vs. vorhergesagte Werte für {feature}')
            plt.savefig(residuals_vs_fitted_path)
            plt.close()

    print(f'Analyse abgeschlossen. Ergebnisse und Plots wurden im Ordner "{output_path}" gespeichert.')
    print('ancova executed.')


def ancova_mit_transformationen(df, feature_columns, output_path_base):
    print('ancova_mit_transformation started.')

    output_path = os.path.join(output_path_base, 'ancova_mit_transformationen')
    # Prüfen, ob der Output-Ordner existiert, sonst erstellen
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Kopie des DataFrames für Transformation
    df_transformed = df.copy()

    # Log-Transformation und Box-Cox-Transformation anwenden
    for feature in feature_columns:
        # Log-Transformation: Wenn Werte <= 0, wird 1 hinzugefügt
        if (df_transformed[feature] <= 0).any():
            df_transformed[f'log_{feature}'] = np.log(df_transformed[feature] + 1)
        else:
            df_transformed[f'log_{feature}'] = np.log(df_transformed[feature])

        # Box-Cox-Transformation: Nur anwendbar, wenn keine negativen Werte vorhanden sind
        df_transformed[f'boxcox_{feature}'], _ = boxcox(df_transformed[feature] + 1)  # +1 für den Fall, dass es 0-Werte gibt

    output_file_log = os.path.join(output_path, 'log_ancova_output.txt')
    with open(output_file_log, 'w', encoding='utf-8') as f:

        # Iteration durch alle Feature-Spalten
        for feature in feature_columns:
            f.write(f'\nANCOVA für Feature: {feature}\n')

            # Verwenden der log-transformierten Features für das Modell
            transformed_feature = f'log_{feature}'

            # Modell mit 'cluster' als Faktor und den neuen Kovariaten
            model = smf.ols(f'{transformed_feature} ~ C(cluster) + c_Dauer + C(c_Teilnehmergruppe)', data=df_transformed).fit()
            table = sm.stats.anova_lm(model, typ=2)
            f.write(f'{table}\n')

            # Überprüfung der Interaktion zwischen 'cluster' und 'c_Dauer'
            model_interaction = smf.ols(f'{transformed_feature} ~ C(cluster) * c_Dauer + C(c_Teilnehmergruppe)', data=df_transformed).fit()
            interaction_table = sm.stats.anova_lm(model_interaction, typ=2)
            f.write('\nANOVA-Tabelle mit Interaktionsterm:\n')
            f.write(f'{interaction_table}\n')

            # Prüfen, ob die Interaktion signifikant ist
            if interaction_table.loc['C(cluster):c_Dauer', 'PR(>F)'] < 0.05:
                f.write('Warnung: Signifikante Interaktion zwischen Cluster und Dauer. Die Voraussetzung der Homogenität der Regressionskoeffizienten ist verletzt.\n')
            else:
                f.write('Keine signifikante Interaktion gefunden. Voraussetzung der Homogenität der Regressionskoeffizienten ist erfüllt.\n')

            # Residuen extrahieren
            residuals = model.resid

            # Normalverteilung der Residuen prüfen
            stat, p_value = shapiro(residuals)
            f.write(f'Shapiro-Wilk-Test für Residuen: p-Wert = {p_value}\n')
            if p_value < 0.05:
                f.write('Warnung: Residuen sind nicht normalverteilt.\n')
            else:
                f.write('Residuen sind normalverteilt.\n')

            # QQ-Plot der Residuen
            qq_plot_path = os.path.join(output_path, f'log_qq_plot_{transformed_feature}.png')
            sm.qqplot(residuals, line='45')
            plt.title(f'QQ-Plot der Residuen für {transformed_feature}')
            plt.savefig(qq_plot_path)
            plt.close()

            # Homoskedastizität prüfen (Grafisch)
            residuals_vs_fitted_path = os.path.join(output_path, f'log_residuen_vs_fitted_{transformed_feature}.png')
            plt.scatter(model.fittedvalues, residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Vorhergesagte Werte')
            plt.ylabel('Residuen')
            plt.title(f'Residuen vs. vorhergesagte Werte für {transformed_feature}')
            plt.savefig(residuals_vs_fitted_path)
            plt.close()

    output_file_boxcox = os.path.join(output_path, 'boxcox_ancova_output.txt')
    with open(output_file_boxcox, 'w', encoding='utf-8') as f:

        # Iteration durch alle Feature-Spalten
        for feature in feature_columns:
            f.write(f'\nANCOVA für Feature: {feature}\n')

            # Verwenden der boxcox-transformierten Features für das Modell
            transformed_feature = f'boxcox_{feature}'

            # Modell mit 'cluster' als Faktor und den neuen Kovariaten
            model = smf.ols(f'{transformed_feature} ~ C(cluster) + c_Dauer + C(c_Teilnehmergruppe)', data=df_transformed).fit()
            table = sm.stats.anova_lm(model, typ=2)
            f.write(f'{table}\n')

            # Überprüfung der Interaktion zwischen 'cluster' und 'c_Dauer'
            model_interaction = smf.ols(f'{transformed_feature} ~ C(cluster) * c_Dauer + C(c_Teilnehmergruppe)', data=df_transformed).fit()
            interaction_table = sm.stats.anova_lm(model_interaction, typ=2)
            f.write('\nANOVA-Tabelle mit Interaktionsterm:\n')
            f.write(f'{interaction_table}\n')

            # Prüfen, ob die Interaktion signifikant ist
            if interaction_table.loc['C(cluster):c_Dauer', 'PR(>F)'] < 0.05:
                f.write('Warnung: Signifikante Interaktion zwischen Cluster und Dauer. Die Voraussetzung der Homogenität der Regressionskoeffizienten ist verletzt.\n')
            else:
                f.write('Keine signifikante Interaktion gefunden. Voraussetzung der Homogenität der Regressionskoeffizienten ist erfüllt.\n')

            # Residuen extrahieren
            residuals = model.resid

            # Normalverteilung der Residuen prüfen
            stat, p_value = shapiro(residuals)
            f.write(f'Shapiro-Wilk-Test für Residuen: p-Wert = {p_value}\n')
            if p_value < 0.05:
                f.write('Warnung: Residuen sind nicht normalverteilt.\n')
            else:
                f.write('Residuen sind normalverteilt.\n')

            # QQ-Plot der Residuen
            qq_plot_path = os.path.join(output_path, f'boxcox_qq_plot_{transformed_feature}.png')
            sm.qqplot(residuals, line='45')
            plt.title(f'QQ-Plot der Residuen für {transformed_feature}')
            plt.savefig(qq_plot_path)
            plt.close()

            # Homoskedastizität prüfen (Grafisch)
            residuals_vs_fitted_path = os.path.join(output_path, f'boxcox_residuen_vs_fitted_{transformed_feature}.png')
            plt.scatter(model.fittedvalues, residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Vorhergesagte Werte')
            plt.ylabel('Residuen')
            plt.title(f'Residuen vs. vorhergesagte Werte für {transformed_feature}')
            plt.savefig(residuals_vs_fitted_path)
            plt.close()

    print(f'Analyse abgeschlossen. Ergebnisse und Plots wurden im Ordner "{output_path}" gespeichert.')
    print('erweiterte_ancova_mit_transformation executed.')


def rang_ancova(df, feature_columns, output_path_base):
    print('rang_ancova started.')

    output_path = os.path.join(output_path_base, 'rang_ancova')
    # Prüfen, ob der Output-Ordner existiert, sonst erstellen
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Erstellen einer Kopie des DataFrames zur Arbeit mit den Rangdaten
    df_ranked = df.copy()

    # Erstellen der Ausgabedatei für den Konsolenausgabe
    output_file = os.path.join(output_path, 'ancova_output.txt')
    with open(output_file, 'w', encoding='utf-8') as f:

        # Iteration über alle Feature-Spalten
        for feature in feature_columns:
            f.write(f'\nRang-ANCOVA für Feature: {feature}\n')

            # Ranken der abhängigen Variable (Feature) und der Kovariate (c_Dauer)
            df_ranked[f'rank_{feature}'] = rankdata(df[feature])
            df_ranked['rank_c_Dauer'] = rankdata(df['c_Dauer'])

            # Durchführung der Rang-ANCOVA
            model_ranked = smf.ols(
                formula=f'rank_{feature} ~ C(cluster) + rank_c_Dauer + C(c_Teilnehmergruppe)',
                data=df_ranked
            ).fit()

            # Ausgabe der ANOVA-Tabelle
            table_ranked = sm.stats.anova_lm(model_ranked, typ=2)
            f.write(f'{table_ranked}\n')

            # Residuen extrahieren
            residuals = model_ranked.resid

            # QQ-Plot der Residuen
            qq_plot_path = os.path.join(output_path, f'qq_plot_{feature}_ranked.png')
            sm.qqplot(residuals, line='45')
            plt.title(f'QQ-Plot der Residuen für {feature} (Rangdaten)')
            plt.savefig(qq_plot_path)
            plt.close()

            # Homoskedastizität prüfen (Grafisch)
            residuals_vs_fitted_path = os.path.join(output_path, f'residuen_vs_fitted_{feature}_ranked.png')
            plt.scatter(model_ranked.fittedvalues, residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Vorhergesagte Werte (Rangdaten)')
            plt.ylabel('Residuen')
            plt.title(f'Residuen vs. vorhergesagte Werte für {feature} (Rangdaten)')
            plt.savefig(residuals_vs_fitted_path)
            plt.close()

    print(f'Analyse abgeschlossen. Ergebnisse und Plots wurden im Ordner "{output_path}" gespeichert.')
    print('rang_ancova executed.')


def glmm_poisson(df, feature_columns, output_path_base):
    print('GLMM Poisson started.')

    # Setze den R_HOME Pfad für deine R-Installation
    os.environ["R_HOME"] = "C:/Program Files/R/R-4.4.1"

    import rpy2.robjects.packages as rpackages
    from rpy2.robjects import pandas2ri
    from rpy2 import robjects

    pandas2ri.activate()

    # Überprüfen, ob das lme4 Paket vorhanden ist, andernfalls installieren
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)  # Wähle einen CRAN-Mirror zur Installation
    if not rpackages.isinstalled('lme4'):
        utils.install_packages('lme4')
    lme4 = rpackages.importr('lme4')

    output_path = os.path.join(output_path_base, 'glmm_poisson')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, 'glmm_poisson_summary.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        if 'c_Dauer' in df.columns:
            df['rank_c_Dauer'] = df['c_Dauer'].rank()

            # Überprüfen auf fehlende Werte
            if df.isnull().values.any():
                print("Warning: DataFrame contains NaN or None values.")

            # Konvertiere den pandas DataFrame in einen R DataFrame
            r_df = pandas2ri.py2rpy(df)

            # Schleife durch alle Features
            for feature_column in feature_columns:
                f.write(f'\nErgebnisse für Feature: {feature_column}\n')

                # Formel für das Modell (Poisson-Verteilung)
                formula = f'{feature_column} ~ C(cluster) + rank_c_Dauer + (1|cluster)'

                try:
                    # Modell in R ausführen
                    model_result = robjects.r.glmer(formula, family=robjects.r.poisson, data=r_df)
                    summary = robjects.r('summary')(model_result)
                    f.write(summary.__str__())
                except Exception as e:
                    f.write(f'Fehler bei der GLMM-Berechnung: {str(e)}\n')
                    continue

                # Pearson-Residuen und Zufallseffekte berechnen
                fitted_values = np.array(robjects.r('fitted')(model_result))
                residuals_pearson = np.array(robjects.r('residuals')(model_result, type="pearson"))

                # Pearson-Residuen plotten (Homoskedastizität prüfen)
                plt.scatter(fitted_values, residuals_pearson)
                plt.axhline(0, linestyle='--', color='r')
                plt.xlabel('Vorhergesagte Werte')
                plt.ylabel('Pearson-Residuen')
                plt.title(f'Pearson-Residuen vs. vorhergesagte Werte für {feature_column}')
                plt_file = os.path.join(output_path, f'pearson_residuals_vs_fitted_{feature_column}.png')
                plt.savefig(plt_file)
                plt.close()

                # Overdispersion prüfen (Ratio von Pearson-Chi-Square zu Freiheitsgraden)
                chi_square = sum(residuals_pearson ** 2)
                df_resid = len(fitted_values) - len(robjects.r('fixef')(model_result))
                ratio = chi_square / df_resid
                f.write(f'\nChi-Square/df ratio für {feature_column}: {ratio}\n')

                # QQ-Plot der Pearson-Residuen (Normalität der Residuen überprüfen)
                sm.qqplot(residuals_pearson, line='45')
                plt.title(f'QQ-Plot der Pearson-Residuen für {feature_column}')
                qq_plot_file = os.path.join(output_path, f'qq_plot_pearson_residuals_{feature_column}.png')
                plt.savefig(qq_plot_file)
                plt.close()

                # Zufallseffekte überprüfen
                random_effects = robjects.r('ranef')(model_result)
                random_effect_cluster = np.array(random_effects.rx2('cluster')[0])  # Korrekt auf die Effekte zugreifen
                plt.scatter(range(len(random_effect_cluster)), random_effect_cluster)
                plt.axhline(0, linestyle='--', color='r')
                plt.xlabel('Gruppen')
                plt.ylabel('Zufallseffekte')
                plt.title(f'Zufallseffekte über Gruppen hinweg für {feature_column}')
                random_effects_file = os.path.join(output_path, f'random_effects_{feature_column}.png')
                plt.savefig(random_effects_file)
                plt.close()

        else:
            print("Die Spalte 'c_Dauer' wurde nicht gefunden.")

    print('GLMM Poisson completed.')



# Funktioniert noch nicht:
# R[write to console]: Error in formula[[length(formula)]] <- value :
# long vectors not supported yet: subassign.c:1851
def glmm_negative_binomial(df, feature_columns, output_path_base):
    """
    Führt ein GLMM mit Negativ-Binomial-Verteilung für die angegebenen feature_columns durch.

    Parameters:
    df (DataFrame): Der DataFrame, der die Features und Cluster enthält.
    feature_columns (list): Eine Liste der Features, für die die Modelle durchgeführt werden.
    output_path_base (str): Der Basisordnerpfad, in dem die Ergebnisse gespeichert werden sollen.
    """
    print('GLMM Negativ-Binomial started.')

    # Setze den R_HOME Pfad für deine R-Installation
    os.environ["R_HOME"] = "C:/Program Files/R/R-4.4.1"

    import rpy2.robjects.packages as rpackages
    from rpy2.robjects import pandas2ri, Formula
    pandas2ri.activate()

    # Überprüfen, ob das lme4 Paket und MASS für Negativ-Binomial vorhanden sind, andernfalls installieren
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)  # Wähle einen CRAN-Mirror zur Installation
    if not rpackages.isinstalled('lme4'):
        utils.install_packages('lme4')
    if not rpackages.isinstalled('MASS'):
        utils.install_packages('MASS')  # Für Negativ-Binomial in R

    lme4 = rpackages.importr('lme4')
    MASS = rpackages.importr('MASS')

    output_path = os.path.join(output_path_base, 'glmm_negative_binomial')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, 'glmm_negative_binomial_summary.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        if 'c_Dauer' in df.columns:
            # Berechnung des Rangs und Umwandlung in Ganzzahlen
            df['rank_c_Dauer'] = df['c_Dauer'].rank().round().astype(int)

            # Konvertiere den pandas DataFrame in einen R DataFrame
            r_df = pandas2ri.py2rpy(df)

            # Schleife durch alle Features
            for feature_column in feature_columns:
                f.write(f'\nErgebnisse für Feature: {feature_column}\n')

                # Formel für das Modell (Negativ-Binomial-Verteilung)
                formula = Formula(f'{feature_column} ~ C(cluster) + rank_c_Dauer + (1|cluster)')

                try:
                    # Negativ-Binomial-Modell in R ausführen
                    model_result = MASS.glmer_nb(formula, data=r_df)
                    summary = robjects.r('summary')(model_result)
                    f.write(summary.__str__())

                    # Pearson-Residuen und Zufallseffekte berechnen
                    fitted_values = np.array(robjects.r('fitted')(model_result))
                    residuals_pearson = np.array(robjects.r('residuals')(model_result, type="pearson"))

                    # Pearson-Residuen plotten (Homoskedastizität prüfen)
                    plt.scatter(fitted_values, residuals_pearson)
                    plt.axhline(0, linestyle='--', color='r')
                    plt.xlabel('Vorhergesagte Werte')
                    plt.ylabel('Pearson-Residuen')
                    plt.title(f'Pearson-Residuen vs. vorhergesagte Werte für {feature_column}')
                    plt_file = os.path.join(output_path, f'pearson_residuals_vs_fitted_{feature_column}.png')
                    plt.savefig(plt_file)
                    plt.close()

                    # Overdispersion prüfen (Ratio von Pearson-Chi-Square zu Freiheitsgraden)
                    chi_square = sum(residuals_pearson ** 2)
                    df_resid = len(fitted_values) - len(robjects.r('fixef')(model_result))
                    ratio = chi_square / df_resid
                    f.write(f'\nChi-Square/df ratio für {feature_column}: {ratio}\n')

                    # QQ-Plot der Pearson-Residuen (Normalität der Residuen überprüfen)
                    sm.qqplot(residuals_pearson, line='45')
                    plt.title(f'QQ-Plot der Pearson-Residuen für {feature_column}')
                    qq_plot_file = os.path.join(output_path, f'qq_plot_pearson_residuals_{feature_column}.png')
                    plt.savefig(qq_plot_file)
                    plt.close()

                    # Zufallseffekte überprüfen
                    random_effects = robjects.r('ranef')(model_result)
                    random_effect_cluster = np.array(random_effects.rx2('cluster')[0])
                    plt.scatter(range(len(random_effect_cluster)), random_effect_cluster)
                    plt.axhline(0, linestyle='--', color='r')
                    plt.xlabel('Gruppen')
                    plt.ylabel('Zufallseffekte')
                    plt.title(f'Zufallseffekte über Gruppen hinweg für {feature_column}')
                    random_effects_file = os.path.join(output_path, f'random_effects_{feature_column}.png')
                    plt.savefig(random_effects_file)
                    plt.close()

                except Exception as e:
                    f.write(f'Fehler bei der GLMM-Berechnung für {feature_column}: {str(e)}\n')

        else:
            print("Die Spalte 'c_Dauer' wurde nicht gefunden.")

    print('GLMM Negativ-Binomial completed.')


# Funktioniert nicht gibt nullwer zurück
def zinb_model(df, feature_columns, output_path_base):
    """
    Führt ein Zero-Inflated Negativ-Binomial-Modell für jedes Feature in den angegebenen feature_columns durch.
    Wandelt nur explizit um, wo nötig, und speichert die Ergebnisse sowie Plots.

    Parameters:
    df (DataFrame): Der DataFrame, der die Features und Cluster enthält.
    feature_columns (list): Eine Liste der Features, für die die Tests durchgeführt werden.
    output_path_base (str): Der Basisordnerpfad, in dem die Ergebnisse gespeichert werden sollen.
    """
    print('Zero-Inflated Negativ-Binomial Modell gestartet.')

    import rpy2.robjects.packages as rpackages
    from rpy2.robjects import pandas2ri, Formula
    from rpy2.rinterface import NULL as rNULL
    pandas2ri.activate()

    # Pakete importieren und installieren, falls nötig
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)

    if not rpackages.isinstalled('glmmTMB'):
        utils.install_packages('glmmTMB')
    glmmTMB = rpackages.importr('glmmTMB')

    output_path = os.path.join(output_path_base, 'zinb_model')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, 'zinb_model_summary.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        # Schleife durch die numerischen Spalten und konvertiere nur, wenn es sich um tatsächliche Ganzzahlen handelt
        def safe_convert(x):
            if pd.api.types.is_numeric_dtype(x) and (x.dropna() == x.dropna().astype(int)).all():
                return x.astype('Int64')
            else:
                return x

        if 'c_Dauer' in df.columns:
            df['rank_c_Dauer'] = df['c_Dauer'].rank().round().astype(int)
        else:
            raise ValueError("'c_Dauer' column is missing from DataFrame.")

        # Wende die sichere Konvertierung auf jede Spalte an
        #df = df.apply(safe_convert)

        # Schleife durch alle Features
        for feature_column in feature_columns:
            if df[feature_column].isnull().any() or not pd.api.types.is_numeric_dtype(df[feature_column]):
                f.write(f'Überspringe Feature {feature_column}: enthält fehlende oder nicht-numerische Werte.\n')
                print(f'Überspringe Feature {feature_column}: enthält fehlende oder nicht-numerische Werte.')
                continue

            f.write(f'\nErgebnisse für Feature: {feature_column}\n')

            # Konvertiere den pandas DataFrame in einen R DataFrame
            r_df = pandas2ri.py2rpy(df)

            # Formel für das Zero-Inflated Negativ-Binomial-Modell
            #formula = Formula(f'{feature_column} ~ C(cluster) + (1|cluster) + rank_c_Dauer')
            formula = Formula(f'{feature_column} ~ C(cluster) + (1|cluster)')

            # Modell in R ausführen
            try:
                model_result = glmmTMB.glmmTMB(formula, zi=Formula('~1'), family='nbinom2', data=r_df)
                #model_result = glmmTMB.glmmTMB(formula, family='nbinom2', data=r_df)

                if model_result == rNULL:
                    raise ValueError(f"Modellergebnis für {feature_column} ist NULL.")
            except Exception as e:
                f.write(f'Fehler beim Modell für {feature_column}: {str(e)}\n')
                print(f"Fehler bei {feature_column}: {str(e)}")
                continue

            # Überprüfe, ob model_result korrekt ist, bevor du darauf zugreifst
            if model_result is not None:
                try:
                    # Zusammenfassung des Modells
                    summary = robjects.r('summary')(model_result)
                    f.write(summary.__str__())

                    # Pearson-Residuen und Zufallseffekte berechnen
                    fitted_values = np.array(robjects.r('fitted')(model_result))
                    residuals_pearson = np.array(robjects.r('residuals')(model_result, type="pearson"))

                    # Pearson-Residuen plotten
                    plt.scatter(fitted_values, residuals_pearson)
                    plt.axhline(0, linestyle='--', color='r')
                    plt.xlabel('Vorhergesagte Werte')
                    plt.ylabel('Pearson-Residuen')
                    plt.title(f'Pearson-Residuen vs. vorhergesagte Werte für {feature_column}')
                    plt_file = os.path.join(output_path, f'pearson_residuals_vs_fitted_{feature_column}.png')
                    plt.savefig(plt_file)
                    plt.close()

                    # Overdispersion prüfen (Ratio von Pearson-Chi-Square zu Freiheitsgraden)
                    chi_square = sum(residuals_pearson ** 2)
                    df_resid = len(fitted_values) - len(robjects.r('fixef')(model_result))
                    ratio = chi_square / df_resid
                    f.write(f'\nChi-Square/df ratio für {feature_column}: {ratio}\n')

                    # QQ-Plot der Pearson-Residuen
                    sm.qqplot(residuals_pearson, line='45')
                    plt.title(f'QQ-Plot der Pearson-Residuen für {feature_column}')
                    qq_plot_file = os.path.join(output_path, f'qq_plot_pearson_residuals_{feature_column}.png')
                    plt.savefig(qq_plot_file)
                    plt.close()

                    # Zufallseffekte überprüfen
                    random_effects = robjects.r('ranef')(model_result)
                    random_effect_cluster = np.array(random_effects.rx2('cluster')[0])  # Korrekt auf die Effekte zugreifen
                    plt.scatter(range(len(random_effect_cluster)), random_effect_cluster)
                    plt.axhline(0, linestyle='--', color='r')
                    plt.xlabel('Gruppen')
                    plt.ylabel('Zufallseffekte')
                    plt.title(f'Zufallseffekte über Gruppen hinweg für {feature_column}')
                    random_effects_file = os.path.join(output_path, f'random_effects_{feature_column}.png')
                    plt.savefig(random_effects_file)
                    plt.close()

                except Exception as e:
                    f.write(f'Fehler bei der Verarbeitung des Modells für {feature_column}: {str(e)}\n')
                    print(f"Fehler bei der Verarbeitung von {feature_column}: {str(e)}")
            else:
                f.write(f"Fehler: Modell für {feature_column} ist None.\n")

    print('Zero-Inflated Negativ-Binomial Modell abgeschlossen.')



def check_normality_homoscedasticity(cluster_1, cluster_2):
    # Überprüfen, ob beide Gruppen normalverteilt sind (Lilliefors-Test)
    stat1, p1 = lilliefors(cluster_1)
    stat2, p2 = lilliefors(cluster_2)
    normal1 = p1 > 0.05
    normal2 = p2 > 0.05

    # Überprüfen, ob beide Gruppen homoskedastisch sind (Levene-Test)
    stat_levene, p_levene = stats.levene(cluster_1, cluster_2)
    homoscedastic = p_levene > 0.05

    return normal1, p1, normal2, p2, homoscedastic, p_levene


def simple_stat_tests(df, feature_columns, output_path_base):
    """
    Führt verschiedene statistische Tests durch, um Unterschiede zwischen zwei Clustern (0 und 1)
    in Bezug auf die angegebenen Features zu untersuchen und erzeugt passende Visualisierungen.

    Die Tests umfassen Normalitätstests, Homoskedastizitätstests und verschiedene Signifikanztests
    wie t-Tests, Mann-Whitney-U-Tests und Chi-Quadrat-Tests. Die Funktion speichert die
    Testergebnisse und Visualisierungen in den angegebenen Pfaden.

    Parameter:
    - df: Pandas DataFrame, das die Feature-Daten und Cluster-Zuweisungen enthält.
          Es wird davon ausgegangen, dass die Spalte 'cluster' die Cluster-IDs enthält
          (mit Werten 0 und 1) und dass die Features numerische oder kategoriale Daten
          enthalten.
    - feature_columns: Liste der zu testenden Features. Jedes Feature wird auf Unterschiede
                       zwischen den beiden Clustern untersucht.
    - output_path_base: Basisverzeichnis, in dem die Ergebnisse gespeichert werden.
                        Die Funktion erstellt einen Unterordner 'simple_stat_tests'
                        und speichert die Testergebnisse sowie Visualisierungen dort.

    Durchgeführte Tests:
    1. Normalitätstest (Lilliefors-Test): Testet, ob die Daten innerhalb jedes Clusters
       normalverteilt sind.
    2. Homoskedastizitätstest (Levene-Test): Testet, ob die Varianzen in den beiden Clustern gleich sind.
    3. Unabhängiger t-Test: Vergleicht die Mittelwerte der beiden Cluster, wenn Normalität und
       Homoskedastizität gegeben sind.
    4. Gepaarter t-Test: Vergleicht die Mittelwerte der beiden Cluster, wenn die Daten gepaart sind
       (gleiche Länge und normalverteilt).
    5. Mann-Whitney-U-Test: Nicht-parametrischer Test, der verwendet wird, wenn die Daten nicht
       normalverteilt sind.
    6. Chi-Quadrat-Test: Wird durchgeführt, wenn die Daten kategorial sind oder weniger als 10
       eindeutige Werte aufweisen.

    Erzeugte Visualisierungen:
    1. Boxplot: Vergleicht die Verteilungen der Features zwischen den Clustern.
    2. Violinplot: Zeigt die Verteilungsform der Features in den Clustern.
    3. Histogramm/Dichteplot: Visualisiert die Verteilung der Features innerhalb jedes Clusters.
    4. Scatterplot: Zeigt die Beziehung zwischen zwei Features und den Clustern.
    5. QQ-Plot: Überprüft die Normalverteilung der Features.
    6. Pairplot: Vergleicht mehrere Features zwischen den Clustern.
    7. Heatmap der Korrelationen: Zeigt Korrelationen zwischen Features.

    Alle Ergebnisse und Plots werden im angegebenen Ausgabepfad gespeichert.
    """

    print('Simple Stat Tests started.')

    output_path = os.path.join(output_path_base, 'simple_stat_tests')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, 'stat_tests_summary.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        for feature_column in feature_columns:
            f.write(f'\nErgebnisse für Feature: {feature_column}\n')

            # Debug: Überprüfe Datentyp und Werte
            print(f"Überprüfe Feature: {feature_column}")
            print(f"Datentyp von {feature_column}: {df[feature_column].dtype}")
            print(f"Cluster-Werte: {df['cluster'].unique()}")

            # Daten für die beiden Cluster (0 und 1) extrahieren
            cluster_0 = df[df['cluster'] == 0][feature_column].dropna()
            cluster_1 = df[df['cluster'] == 1][feature_column].dropna()

            # Debug: Überprüfe Größe der Cluster
            print(f"Cluster 0 Größe für {feature_column}: {len(cluster_0)}")
            print(f"Cluster 1 Größe für {feature_column}: {len(cluster_1)}")

            # Überprüfen, ob die Cluster leer sind
            if len(cluster_0) == 0 or len(cluster_1) == 0:
                f.write('Einer der Cluster hat keine Daten.\n')
                continue

            # Normalitäts- und Homoskedastizitätstests
            try:
                normal1, p1, normal2, p2, homoscedastic, p_levene = check_normality_homoscedasticity(cluster_0, cluster_1)
                f.write(f'Normalität für Cluster 0: {normal1}, p-Wert: {p1}\n')
                f.write(f'Normalität für Cluster 1: {normal2}, p-Wert: {p2}\n')
                f.write(f'Homoskedastizität (gleichmäßige Varianz): {homoscedastic}, p-Wert: {p_levene}\n')
            except Exception as e:
                f.write(f'Fehler bei der Durchführung der Normalitäts- oder Homoskedastizitätstests: {str(e)}\n')
                continue

            # 1. t-Test (unabhängig)
            if normal1 and normal2 and homoscedastic:
                try:
                    t_stat, p_value = stats.ttest_ind(cluster_0, cluster_1)
                    f.write(f'Unabhängiger t-Test: t-Statistik: {t_stat}, p-Wert: {p_value}\n')
                except Exception as e:
                    f.write(f'Fehler beim t-Test: {str(e)}\n')

            # 2. Gepaarter t-Test (falls die Daten gepaart wären, z.B. Vorher-Nachher)
            elif normal1 and normal2 and len(cluster_0) == len(cluster_1):
                try:
                    t_stat, p_value = stats.ttest_rel(cluster_0, cluster_1)
                    f.write(f'Gepaarter t-Test: t-Statistik: {t_stat}, p-Wert: {p_value}\n')
                except Exception as e:
                    f.write(f'Fehler beim gepaarten t-Test: {str(e)}\n')
            else:
                f.write('Gepaarter t-Test nicht möglich, da die Cluster unterschiedlich groß sind oder nicht normalverteilt.\n')

            # 3. Mann-Whitney-U-Test (bei nicht-normalverteilten Daten)
            try:
                u_stat, p_value = stats.mannwhitneyu(cluster_0, cluster_1)
                f.write(f'Mann-Whitney-U-Test: U-Statistik: {u_stat}, p-Wert: {p_value}\n')
            except Exception as e:
                f.write(f'Fehler beim Mann-Whitney-U-Test: {str(e)}\n')

            # 4. Chi-Quadrat-Test (falls die Daten kategorial sind)
            if pd.api.types.is_categorical_dtype(df[feature_column]) or df[feature_column].nunique() < 10:
                try:
                    contingency_table = pd.crosstab(df['cluster'], df[feature_column])
                    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                    f.write(f'Chi-Quadrat-Test: Chi2-Statistik: {chi2_stat}, p-Wert: {p_value}\n')
                except Exception as e:
                    f.write(f'Fehler beim Chi-Quadrat-Test: {str(e)}\n')

            # Visualisierungen:
            # 1. Boxplot
            boxplot_path = os.path.join(output_path, f'boxplot_{feature_column}.png')
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='cluster', y=feature_column, data=df)
            plt.title(f'Boxplot für {feature_column} nach Cluster')
            plt.savefig(boxplot_path)
            plt.close()

            # 2. Violinplot
            violinplot_path = os.path.join(output_path, f'violinplot_{feature_column}.png')
            plt.figure(figsize=(10, 6))
            sns.violinplot(x='cluster', y=feature_column, data=df)
            plt.title(f'Violinplot für {feature_column} nach Cluster')
            plt.savefig(violinplot_path)
            plt.close()

            # 3. Histogramm/Dichteplot
            histplot_path = os.path.join(output_path, f'histplot_{feature_column}.png')
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=feature_column, hue='cluster', kde=True, element="step")
            plt.title(f'Histogramm von {feature_column} nach Cluster')
            plt.savefig(histplot_path)
            plt.close()

            # 4. QQ-Plot (Quantil-Quantil-Plot)
            qqplot_path = os.path.join(output_path, f'qqplot_{feature_column}.png')
            sm.qqplot(df[feature_column], line='45')
            plt.title(f'QQ-Plot von {feature_column}')
            plt.savefig(qqplot_path)
            plt.close()

    print(f'Analyse abgeschlossen. Ergebnisse und Plots wurden im Ordner "{output_path}" gespeichert.')



def chi_square_tests_and_plots(df, indicator_columns, output_path_base):
    print('Chi-Quadrat-Tests und Plots gestartet.')

    output_path = os.path.join(output_path_base, 'chi_square_tests_indicators')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Erstellen der Ausgabedatei für die Chi-Quadrat-Ergebnisse
    output_file = os.path.join(output_path, 'chi_square_results.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        # Chi-Quadrat-Test für jede Indikatorvariable
        for indicator in indicator_columns:
            contingency_table = pd.crosstab(df['cluster'], df[indicator])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            f.write(f'\nChi-Quadrat-Test Ergebnisse für {indicator}: chi2 = {chi2}, p-Wert = {p}\n')
            f.write(f'Erwartete Frequenzen:\n{expected}\n')

            # Barplot der Verteilung nach Cluster
            contingency_table_normalized = pd.crosstab(df['cluster'], df[indicator], normalize='index')
            contingency_table_normalized.plot(kind='bar', stacked=True)
            plt.title(f'Verteilung von {indicator} nach Cluster')
            plt.ylabel('Proportion')
            plt.xlabel('Cluster')
            plt_file = os.path.join(output_path, f'barplot_{indicator}.png')
            plt.savefig(plt_file)
            plt.close()

            # Heatmap der Kontingenztabelle
            sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu")
            plt.title(f'Heatmap der Kontingenztabelle für {indicator}')
            plt.ylabel('Cluster')
            plt.xlabel(indicator)
            heatmap_file = os.path.join(output_path, f'heatmap_{indicator}.png')
            plt.savefig(heatmap_file)
            plt.close()

            # Mosaic Plot erstellen
            mosaic(df, [indicator, 'cluster'])
            plt.title(f'Mosaic Plot für {indicator} nach Cluster')
            mosaic_file = os.path.join(output_path, f'mosaic_{indicator}.png')
            plt.savefig(mosaic_file)
            plt.close()

            # Pie Chart
            for cluster in df['cluster'].unique():
                cluster_data = df[df['cluster'] == cluster][indicator].value_counts()
                plt.pie(cluster_data, labels=cluster_data.index, autopct='%1.1f%%')
                plt.title(f'Tortendiagramm der Verteilung von {indicator} für Cluster {cluster}')
                pie_file = os.path.join(output_path, f'piechart_{indicator}_cluster_{cluster}.png')
                plt.savefig(pie_file)
                plt.close()

            # Stacked Area Chart
            contingency_table_normalized.T.plot(kind='area', stacked=True)
            plt.title(f'Gestapeltes Flächendiagramm für {indicator} nach Cluster')
            plt.ylabel('Proportion')
            plt.xlabel('Cluster')
            area_chart_file = os.path.join(output_path, f'stacked_area_chart_{indicator}.png')
            plt.savefig(area_chart_file)
            plt.close()

    print(f'Chi-Quadrat-Tests und Plots abgeschlossen. Ergebnisse wurden im Ordner "{output_path}" gespeichert.')


def analyze_duration_relationship(df, feature_columns):
    """
    Funktion zur Untersuchung der linearen Beziehung zwischen Dauer und Features in Clustern.

    :param df: DataFrame, der die Daten enthält
    :param feature_columns: Liste der Feature-Spalten, die untersucht werden sollen
    """
    for feature in feature_columns:
        print(f'\nRegressionskoeffizienten für Feature: {feature}')
        for cluster in df['cluster'].unique():
            subset = df[df['cluster'] == cluster]

            # Berechne die Steigung und den Achsenabschnitt für das aktuelle Cluster
            slope, intercept = np.polyfit(subset['duration_c'], subset[feature], 1)
            print(f'Cluster {cluster}: Steigung = {slope}')

        # Erstelle den lmplot für die Beziehung zwischen Dauer und Feature
        sns.lmplot(x='duration_c', y=feature, data=df, hue='cluster', markers='o')
        plt.title(f'Lineare Beziehung zwischen Dauer und {feature}')
        plt.xlabel('Dauer (z-standardisiert)')
        plt.ylabel(feature)
        plt.show()


def posthoc_dunn_test(df, feature_columns, output_path_base):
    """
    Führt Dunn-Posthoc-Tests für jedes Feature in den angegebenen feature_columns durch und speichert die Ergebnisse.

    Parameters:
    df (DataFrame): Der DataFrame, der die Features und Cluster enthält.
    feature_columns (list): Eine Liste der Features, für die die Tests durchgeführt werden.
    output_path_base (str): Der Basisordnerpfad, in dem die Ergebnisse gespeichert werden sollen.
    """
    print('Post-hoc Dunn Test started.')

    output_path = os.path.join(output_path_base, 'posthoc_dunn')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, 'posthoc_dunn_results.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        # Schleife durch alle Features
        for feature in feature_columns:
            f.write(f'Post-hoc-Ergebnisse für {feature}:\n')

            # Bereinigung der Daten für die Analyse
            data = df[[feature, 'cluster']].dropna()

            # Dunn-Test durchführen
            p_values = sp.posthoc_dunn(data, val_col=feature, group_col='cluster', p_adjust='bonferroni')

            # Ergebnisse speichern
            f.write(p_values.to_string())
            f.write('\n\n')
            print(f'Post-hoc Dunn-Test abgeschlossen für: {feature}')

            # Heatmap der p-Werte erstellen
            plt.figure(figsize=(8, 6))
            sns.heatmap(p_values, annot=True, cmap='coolwarm', cbar=True)
            plt.title(f'Heatmap der Post-hoc Dunn p-Werte für {feature}')
            heatmap_file = os.path.join(output_path, f'heatmap_{feature}.png')
            plt.savefig(heatmap_file)
            plt.close()

            # Boxplot zur Visualisierung der Verteilung pro Cluster
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='cluster', y=feature, data=df, hue='cluster', legend=False)
            plt.title(f'Boxplot der Verteilung von {feature} nach Cluster')
            boxplot_file = os.path.join(output_path, f'boxplot_{feature}.png')
            plt.savefig(boxplot_file)
            plt.close()

    print(f'Alle Post-hoc Dunn-Tests abgeschlossen. Ergebnisse und Plots wurden im Ordner "{output_path}" gespeichert.')


# def permutation_test_with_covariate(df, feature_columns, output_path_base, n_permutations=1000):
#     """
#     Führt einen Permutationstest mit der Kovariate 'c_Dauer' durch, um zu testen,
#     ob die Featurevektoren signifikante Unterschiede zwischen den Clustern aufweisen.
#
#     Parameters:
#     - df: Pandas DataFrame mit den Featurevektoren, den Clustern und der Kovariate 'c_Dauer'.
#     - feature_columns: Liste der zu testenden Features.
#     - output_path_base: Speicherort für die Testergebnisse und Visualisierungen.
#     - n_permutations: Anzahl der Permutationen.
#     """
#     print('Permutationstest mit Kovariate gestartet.')
#
#     output_path = os.path.join(output_path_base, 'permutation_test_with_covariate')
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#
#     output_file = os.path.join(output_path, 'permutation_test_with_covariate_output.txt')
#     with open(output_file, 'w', encoding='utf-8') as f:
#
#         for feature in feature_columns:
#             f.write(f'\nPermutationstest für Feature: {feature}\n')
#
#             # Modell, das die Dauer kontrolliert
#             model = smf.ols(f'{feature} ~ C(cluster) + c_Dauer', data=df).fit()
#
#             # Residuen des Modells extrahieren
#             residuals = model.resid
#
#             # Permutationstest auf den Effekt von 'cluster' durchführen
#             X = df[['cluster']].values.reshape(-1, 1)  # Cluster als Prädiktor
#             y = residuals  # Arbeite mit den Residuen, um den Effekt nach Kontrolle von 'c_Dauer' zu testen
#
#             # Verwende LinearRegression als Estimator
#             estimator = LinearRegression()
#
#             # Führe den Permutationstest durch
#             score, permutation_scores, p_value = permutation_test_score(
#                 estimator,
#                 X=X,
#                 y=y,
#                 n_permutations=n_permutations,
#                 scoring="r2"
#             )
#             f.write(f'Permutationstest: Score = {score}, p-Wert = {p_value}\n')
#
#             # Normalverteilung der Residuen prüfen
#             stat, p_value_resid = shapiro(residuals)
#             f.write(f'Shapiro-Wilk-Test für Residuen: p-Wert = {p_value_resid}\n')
#             if p_value_resid < 0.05:
#                 f.write('Warnung: Residuen sind nicht normalverteilt.\n')
#             else:
#                 f.write('Residuen sind normalverteilt.\n')
#
#             # QQ-Plot der Residuen
#             qq_plot_path = os.path.join(output_path, f'qq_plot_{feature}.png')
#             sm.qqplot(residuals, line='45')
#             plt.title(f'QQ-Plot der Residuen für {feature}')
#             plt.savefig(qq_plot_path)
#             plt.close()
#
#             # Homoskedastizität prüfen (Grafisch)
#             residuals_vs_fitted_path = os.path.join(output_path, f'residuen_vs_fitted_{feature}.png')
#             plt.scatter(model.fittedvalues, residuals)
#             plt.axhline(y=0, color='r', linestyle='--')
#             plt.xlabel('Vorhergesagte Werte')
#             plt.ylabel('Residuen')
#             plt.title(f'Residuen vs. vorhergesagte Werte für {feature}')
#             plt.savefig(residuals_vs_fitted_path)
#             plt.close()
#
#     print(f'Permutationstest abgeschlossen. Ergebnisse und Plots wurden im Ordner "{output_path}" gespeichert.')
def permutation_test_with_covariate(df, feature_columns, output_path_base, n_permutations=1000):
    print('Permutationstest mit Kovariate gestartet.')

    output_path = os.path.join(output_path_base, 'permutation_test_with_covariate')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, 'permutation_test_with_covariate_output.txt')
    with open(output_file, 'w', encoding='utf-8') as f:

        for feature in feature_columns:
            f.write(f'\nPermutationstest für Feature: {feature}\n')

            # Modell, das die Dauer kontrolliert
            model = smf.ols(f'{feature} ~ C(cluster) + c_Dauer', data=df).fit()

            # Residuen des Modells extrahieren
            residuals = model.resid

            # Permutationstest auf den Effekt von 'cluster' durchführen
            X = pd.get_dummies(df['cluster']).values  # Dummy-Kodierung des Clusters
            y = residuals  # Arbeite mit den Residuen, um den Effekt nach Kontrolle von 'c_Dauer' zu testen

            # Verwende LinearRegression als Estimator
            estimator = LinearRegression()

            # Führe den Permutationstest durch
            score, permutation_scores, p_value = permutation_test_score(
                estimator,
                X=X,
                y=y,
                n_permutations=n_permutations,
                scoring="r2"
            )
            f.write(f'Permutationstest: Score = {score}, p-Wert = {p_value}\n')

            # Normalverteilung der Residuen prüfen
            stat, p_value_resid = shapiro(residuals)
            f.write(f'Shapiro-Wilk-Test für Residuen: p-Wert = {p_value_resid}\n')
            if p_value_resid < 0.05:
                f.write('Warnung: Residuen sind nicht normalverteilt.\n')
            else:
                f.write('Residuen sind normalverteilt.\n')

            # QQ-Plot der Residuen
            qq_plot_path = os.path.join(output_path, f'qq_plot_{feature}.png')
            sm.qqplot(residuals, line='45')
            plt.title(f'QQ-Plot der Residuen für {feature}')
            plt.savefig(qq_plot_path)
            plt.close()

            # Homoskedastizität prüfen (Grafisch)
            residuals_vs_fitted_path = os.path.join(output_path, f'residuen_vs_fitted_{feature}.png')
            plt.scatter(model.fittedvalues, residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Vorhergesagte Werte')
            plt.ylabel('Residuen')
            plt.title(f'Residuen vs. vorhergesagte Werte für {feature}')
            plt.savefig(residuals_vs_fitted_path)
            plt.close()

    print(f'Permutationstest abgeschlossen. Ergebnisse und Plots wurden im Ordner "{output_path}" gespeichert.')



def permutation_test_with_and_without_covariate(df, feature_columns, output_path_base, n_permutations=1000):
    """
    Führt einen Permutationstest mit der Kovariate 'c_Dauer' durch, um zu testen,
    ob die Featurevektoren signifikante Unterschiede zwischen den Clustern aufweisen.
    Zusätzlich wird getestet, ob die Kovariate 'c_Dauer' signifikanten Einfluss auf das Modell hat.

    Parameters:
    - df: Pandas DataFrame mit den Featurevektoren, den Clustern und der Kovariate 'c_Dauer'.
    - feature_columns: Liste der zu testenden Features.
    - output_path_base: Speicherort für die Testergebnisse und Visualisierungen.
    - n_permutations: Anzahl der Permutationen.
    """
    print('permutation_test_with_and_without_covariate gestartet.')

    output_path = os.path.join(output_path_base, 'permutation_test_with_and_without_covariate')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, 'permutation_test_with_covariate_output.txt')
    with open(output_file, 'w', encoding='utf-8') as f:

        for feature in feature_columns:
            f.write(f'\nPermutationstest für Feature: {feature}\n')

            # Modell, das die Dauer kontrolliert
            model = smf.ols(f'{feature} ~ C(cluster) + c_Dauer', data=df).fit()

            # Residuen des Modells extrahieren
            residuals = model.resid

            # Boxplot der Residuen nach Cluster (nach Kontrolle von c_Dauer)
            boxplot_path = os.path.join(output_path, f'boxplot_{feature}_residuals.png')
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='cluster', y=residuals, data=df)
            plt.title(f'Boxplot der Residuen von {feature} nach Cluster (nach Kontrolle von c_Dauer)')
            plt.savefig(boxplot_path)
            plt.close()

            # Permutationstest auf den Effekt von 'cluster' durchführen
            X = df[['cluster']].values.reshape(-1, 1)  # Cluster als Prädiktor
            y = residuals  # Arbeite mit den Residuen, um den Effekt nach Kontrolle von 'c_Dauer' zu testen

            # Verwende LinearRegression als Estimator
            estimator = LinearRegression()

            # Führe den Permutationstest durch
            score, permutation_scores, p_value = permutation_test_score(
                estimator,
                X=X,
                y=y,
                n_permutations=n_permutations,
                scoring="r2"
            )
            f.write(f'Permutationstest: Score = {score}, p-Wert = {p_value}\n')

            # Normalverteilung der Residuen prüfen
            stat, p_value_resid = shapiro(residuals)
            f.write(f'Shapiro-Wilk-Test für Residuen: p-Wert = {p_value_resid}\n')
            if p_value_resid < 0.05:
                f.write('Warnung: Residuen sind nicht normalverteilt.\n')
            else:
                f.write('Residuen sind normalverteilt.\n')

            # QQ-Plot der Residuen
            qq_plot_path = os.path.join(output_path, f'qq_plot_{feature}.png')
            sm.qqplot(residuals, line='45')
            plt.title(f'QQ-Plot der Residuen für {feature}')
            plt.savefig(qq_plot_path)
            plt.close()

            # Homoskedastizität prüfen (Grafisch)
            residuals_vs_fitted_path = os.path.join(output_path, f'residuen_vs_fitted_{feature}.png')
            plt.scatter(model.fittedvalues, residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Vorhergesagte Werte')
            plt.ylabel('Residuen')
            plt.title(f'Residuen vs. vorhergesagte Werte für {feature}')
            plt.savefig(residuals_vs_fitted_path)
            plt.close()

            # Zusätzlicher Permutationstest zur Überprüfung des Effekts von 'c_Dauer'
            f.write(f'\nZusätzlicher Permutationstest für den Effekt von c_Dauer auf {feature}:\n')

            # Modelle: mit und ohne Kovariate 'c_Dauer'
            model_with_covariate = smf.ols(f'{feature} ~ C(cluster) + c_Dauer', data=df).fit()
            model_without_covariate = smf.ols(f'{feature} ~ C(cluster)', data=df).fit()

            # Teststatistik: Differenz der R²-Werte (mit und ohne Kovariate)
            r2_with_covariate = model_with_covariate.rsquared
            r2_without_covariate = model_without_covariate.rsquared
            delta_r2 = r2_with_covariate - r2_without_covariate

            # Permutationstest auf den Effekt von 'c_Dauer'
            covariate_permutation_scores = []
            for _ in range(n_permutations):
                df['c_Dauer_permuted'] = np.random.permutation(df['c_Dauer'])
                permuted_model = smf.ols(f'{feature} ~ C(cluster) + c_Dauer_permuted', data=df).fit()
                covariate_permutation_scores.append(permuted_model.rsquared - r2_without_covariate)

            covariate_p_value = (np.sum(np.abs(covariate_permutation_scores) >= np.abs(delta_r2)) + 1) / (n_permutations + 1)

            f.write(f'R² (mit c_Dauer): {r2_with_covariate}, R² (ohne c_Dauer): {r2_without_covariate}\n')
            f.write(f'Differenz der R²-Werte: {delta_r2}\n')
            f.write(f'Permutationstest für c_Dauer: p-Wert = {covariate_p_value}\n')

            if covariate_p_value < 0.05:
                f.write('Ergebnis: c_Dauer hat einen signifikanten Einfluss auf das Modell.\n')
            else:
                f.write('Ergebnis: c_Dauer hat keinen signifikanten Einfluss auf das Modell.\n')

    print(f'permutation_test_with_and_without_covariate. Ergebnisse und Plots wurden im Ordner "{output_path}" gespeichert.')


def permutation_test_without_covariate(df, feature_columns, output_path_base, n_permutations=1000):
    """
    Führt einen Permutationstest ohne Kovariaten durch, um zu testen,
    ob die Featurevektoren signifikante Unterschiede zwischen den Clustern aufweisen.

    Parameters:
    - df: Pandas DataFrame mit den Featurevektoren und den Clustern.
    - feature_columns: Liste der zu testenden Features.
    - output_path_base: Speicherort für die Testergebnisse und Visualisierungen.
    - n_permutations: Anzahl der Permutationen (Standard: 5000).
    """
    print('Permutationstest ohne Kovariate gestartet.')

    # Sicherstellen, dass der Output-Ordner existiert
    output_path = os.path.join(output_path_base, 'permutation_test_without_covariate')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, 'permutation_test_without_covariate_output.txt')
    with open(output_file, 'w', encoding='utf-8') as f:

        for feature in feature_columns:
            f.write(f'\nPermutationstest für Feature: {feature}\n')

            # X: Cluster-Zuweisungen, y: Feature-Werte
            X = df[['cluster']].values
            y = df[feature].values

            # Verwende Ridge-Regression als Estimator
            estimator = Ridge()

            # Permutationstest auf den Effekt von 'cluster' durchführen
            score, permutation_scores, p_value = permutation_test_score(
                estimator=estimator,
                X=X,
                y=y,
                n_permutations=n_permutations,
                scoring="r2"  # r^2 zur Bewertung des Effekts
            )
            f.write(f'Permutationstest: Score = {score}, p-Wert = {p_value}\n')

            # Normalverteilung der Residuen prüfen
            stat, p_value_resid = shapiro(y)
            f.write(f'Shapiro-Wilk-Test für Feature-Werte: p-Wert = {p_value_resid}\n')
            if p_value_resid < 0.05:
                f.write('Warnung: Feature-Werte sind nicht normalverteilt.\n')
            else:
                f.write('Feature-Werte sind normalverteilt.\n')

            # QQ-Plot der Feature-Werte
            qq_plot_path = os.path.join(output_path, f'qq_plot_{feature}.png')
            sm.qqplot(y, line='45')
            plt.title(f'QQ-Plot der Feature-Werte für {feature}')
            plt.savefig(qq_plot_path)
            plt.close()

            # Boxplot der Feature-Werte nach Cluster
            sns.boxplot(x='cluster', y=feature, data=df, hue='cluster', legend=False)
            plt.title(f'Boxplot der Verteilung von {feature} nach Cluster')
            boxplot_file = os.path.join(output_path, f'boxplot_{feature}.png')
            plt.savefig(boxplot_file)
            plt.close()


    print(f'Permutationstest abgeschlossen. Ergebnisse und Plots wurden im Ordner "{output_path}" gespeichert.')
