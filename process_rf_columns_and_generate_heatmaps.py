"""
Modul zur Verarbeitung von RF-Spalten und zur Erstellung von Heatmaps.

Funktionen:
- process_rf_columns(df): Bearbeitet und imputiert die RF-Spalten in einem DataFrame.
- process_and_save_heatmaps(df, output_dir, cluster_columns, key, selected_columns=RF_COLUMNS): Erstellt und speichert
  Heatmaps für die Korrelationen zwischen RF-Spalten und Cluster-Spalten.

Globale Variablen:
- RF_COLUMNS_EINS_BIS_VIER: Tupel mit RF-Spalten für die Eigenschaften der Aufgaben.
- RF_ZEITVORGABE: RF-Spalte für die Zeitvorgabe.
- RF_AUFGABENSTELLUNG: RF-Spalte für die Aufgabenstellung.
- RF_NOTE: RF-Spalte für die Noten.
- RF_COLUMNS: Kombiniertes Tupel aller relevanten RF-Spalten.

Autor: Joshua Tischlik
Datum: 03.10.2024
"""


from sklearn.experimental import enable_iterative_imputer  # Muss importiert werden, um IterativeImputer zu aktivieren
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

RF_COLUMNS_EINS_BIS_VIER = (
    "RF_aufg_motivierend",
    "RF_aufg_interessant",
    "RF_interessiert",
    "RF_aufmerksam",
    "RF_entspannt",
    "RF_gereizt",
    "RF_gestresst",
    "RF_nervos",
    "RF_informiertjetzt",
    "RF_kennefakten",
    "RF_weissviel",
    "RF_wusstevorher"
)
RF_ZEITVORGABE = "RF_zeitvorgabe"
RF_AUFGABENSTELLUNG = "RF_aufgabenstellung"
RF_NOTE = "RF_note"

RF_COLUMNS = RF_COLUMNS_EINS_BIS_VIER + (RF_ZEITVORGABE, RF_AUFGABENSTELLUNG, RF_NOTE)


def process_rf_columns(df):
    """
    Bearbeitet und imputiert die RF-Spalten in einem DataFrame.

    Parameter:
    - df: Der DataFrame, der die zu bearbeitenden RF-Spalten enthält.

    Rückgabewert:
    - df: Der bearbeitete und imputierte DataFrame.
    """

    print("process_rf_columns started.")

    # Liste der Spalten, die tatsächlich im DataFrame vorhanden sind
    existing_rf_columns = [col for col in RF_COLUMNS if col in df.columns]

    def handle_aufgabenstellung(value):
        if value == 2:  # zu einfach
            return 1
        elif value == 1:  # angemessen
            return 2
        elif 3 <= value <= 7:  # zu schwer
            return 3
        else:
            return None

    def handle_zeitvorgabe(value):
        if value == 1:
            return 1
        elif value == 2:
            return 2
        elif value == 3:
            return 3
        else:
            return None

    def handle_note(value):
        if 0.5 <= value < 1.5:
            return 1
        elif 1.5 <= value < 2.5:
            return 2
        elif 2.5 <= value < 3.5:
            return 3
        elif 3.5 <= value < 4.5:
            return 4
        elif 4.5 <= value <= 6:
            return 5
        else:
            return None

    def handle_rest(value):
        value = float(value)
        if value == 1:
            return 1
        elif value == 2:
            return 2
        elif value == 3:
            return 3
        elif value == 4:
            return 4
        else:
            return None

    # Verarbeite die spezifischen Spalten
    for rf_column in RF_COLUMNS:
        try:
            if rf_column == RF_AUFGABENSTELLUNG:
                df[rf_column] = df[rf_column].apply(handle_aufgabenstellung)
            elif rf_column == RF_ZEITVORGABE:
                df[rf_column] = df[rf_column].apply(handle_zeitvorgabe)
            elif rf_column == RF_NOTE:
                df[rf_column] = df[rf_column].apply(handle_note)
            else:
                df[rf_column] = df[rf_column].apply(handle_rest)
        except KeyError:
            print(f"Spalte '{rf_column}' nicht im DataFrame vorhanden.")

    # Wenn es vorhandene Spalten gibt, führe die Imputation durch
    if len(existing_rf_columns) > 2:
        imputer = IterativeImputer(max_iter=1000, random_state=42)
        df_imputed = imputer.fit_transform(df[existing_rf_columns])
        df_imputed = np.rint(df_imputed).astype(int)  # Runden und in Ganzzahlen konvertieren
        df[existing_rf_columns] = df_imputed[:, :len(existing_rf_columns)]
        print(f'Imputation der Spalten {str(existing_rf_columns)} durchgeführt.')
    else:
        print("Nicht mehr als zwei der spezifizierten Spalten für die Imputation vorhanden. Keine Imputation durchgeführt.")

    print("process_rf_columns executed.")
    return df


def process_and_save_heatmaps(df, output_dir, cluster_columns, key, selected_columns=RF_COLUMNS):
    """
    Erstellt und speichert Heatmaps für Korrelationen zwischen RF-Spalten und Cluster-Spalten.

    Parameter:
    - df: Der DataFrame, der die RF- und Cluster-Spalten enthält.
    - output_dir: Das Verzeichnis, in dem die Heatmaps und CSV-Dateien gespeichert werden.
    - cluster_columns: Die Cluster-Spalten im DataFrame.
    - key: Ein Schlüssel für die Benennung der Dateien.
    """

    print("process_and_save_heatmaps started.")

    correlation_methods = ['pearson', 'spearman', 'kendall']

    # Verarbeite RF-Spalten
    df = process_rf_columns(df)
    selected_columns = [col for col in selected_columns if col in df.columns]

    last_row_dfs = {method: pd.DataFrame() for method in correlation_methods}
    first = True
    for i in cluster_columns:
        for method in correlation_methods:
            # Korrelationsmatrix berechnen
            correlation_data = (df[df[i] != -1][selected_columns + [i]].copy()).corr(method=method)  # Entfernt Rauschen

            if first:
                last_row_dfs[method] = pd.concat([last_row_dfs[method], correlation_data.copy()])
            else:
                last_row = (correlation_data.iloc[-1:]).copy()
                last_row_dfs[method] = pd.concat([last_row_dfs[method], last_row])

        first = False

    def get_figsize(df, base_width=1, base_height=0.8, min_width=10, min_height=5):
        """
        Berechnet die optimale Figurgröße für die Heatmap basierend auf der Anzahl der Spalten und Zeilen.

        Parameter:
        - df: Der DataFrame, dessen Dimensionen verwendet werden.
        - base_width: Der Skalierungsfaktor für die Breite pro Spalte.
        - base_height: Der Skalierungsfaktor für die Höhe pro Zeile.
        - min_width: Die Mindestbreite der Figur.
        - min_height: Die Mindesthöhe der Figur.

        Rückgabewert:
        - Tuple: Die berechnete (width, height) Größe der Figur.
        """
        n_rows, n_cols = df.shape
        width = max(min_width, n_cols * base_width)
        height = max(min_height, n_rows * base_height)
        return (width, height)

    # Für jede Methode die Korrelationsmatrix berechnen und die Heatmap speichern
    for method in correlation_methods:
        if not last_row_dfs[method].empty:
            plt.figure(figsize=get_figsize(last_row_dfs[method][selected_columns]))  # Vergrößere die Heatmap
            ax = sns.heatmap(
                last_row_dfs[method][selected_columns], annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                annot_kws={"size": 20, "color": "black"}, fmt=".2f", cbar_kws={'label': 'Korrelation'}
            )

            plt.xticks(rotation=45, ha='right', fontsize=18, color='black')
            plt.yticks(rotation=0, fontsize=18, color='black')

            cb = ax.collections[0].colorbar
            cb.ax.tick_params(labelsize=20, color='black')
            cb.ax.yaxis.set_tick_params(color='black')
            cb.outline.set_edgecolor('black')
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='black')

            plt.suptitle(f"{method.capitalize()} Korrelationen zwischen Cluster-Spalten und RF-Spalten",
                         fontsize=22, y=1.05, color='black')

            plt.subplots_adjust(top=0.92, bottom=0.08)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            heatmap_img_path = os.path.join(output_dir, f'correlation_heatmap_{method}_{key}.png')
            plt.savefig(heatmap_img_path)
            plt.close()
            print(f"Heatmap gespeichert in: {heatmap_img_path}")

            output_csv_path = os.path.join(output_dir, f'correlation_heatmap_{method}_{key}.csv')
            pd.DataFrame(last_row_dfs[method][selected_columns]).to_csv(output_csv_path, index=True)
            print(f"Heatmap gespeichert in: {output_csv_path}")

    print("process_and_save_heatmaps executed.")
