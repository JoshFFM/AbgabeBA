"""
Survey Results Data Preparation Script

Dieses Skript konsolidiert die Schritte zur Datenbereinigung und -vorbereitung der Umfrageergebnisse.
Es generiert bereinigte CSV-Dateien und speichert alle ausgeschlossenen Zeilen (z. B. aufgrund fehlender Daten
oder Listeneinträge) in separaten Dateien. Das Skript bietet eine detaillierte Dokumentation zu ausgeschlossenen
Daten und dem Bereinigungsprozess.

Schritte:

1. Herunterladen der Umfrageergebnisse:
    - Die Umfrageergebnisse wurden manuell vom Dashboard heruntergeladen.

2. Umbenennen der Dateien:
    - Dateien wurden manuell umbenannt, um eine konsistente Namensstruktur zu gewährleisten.

3. Zusammenführen von Dateien:
    - Bestimmte Dateien, wie EC_SU.csv und EC_SU1.csv, werden zusammengeführt, um vollständige Daten sicherzustellen.

4. Neuanordnung und Umbenennung von Spalten:
    - Spalten werden nach spezifischen Anforderungen neu angeordnet und umbenannt.

5. Zusammenführen doppelter Spalten:
    - Spalten, die doppelte Daten enthalten, werden zusammengeführt, und Konflikte (z. B. unterschiedliche numerische Werte)
      werden aufgelöst.

6. Entfernen leerer Zeilen (UserId vorhanden, SessionId fehlt):
    - Zeilen, in denen eine UserId vorhanden ist, aber die SessionId und andere Felder fehlen, werden aus dem Datensatz entfernt.

7. Umgang mit Listen und Duplikaten:
    - Zeilen, die Listen in Zellen enthalten, werden aufgeteilt, sodass jede Zeile nur ein Element der Information enthält.
    - Zeilen mit doppelten UserIds werden basierend auf dem neuesten Zeitstempel verarbeitet, wobei nur die neuesten Einträge beibehalten werden.

8. Speichern der bereinigten Daten:
    - Bereinigte Daten werden im Ordner 'survey-results_with-RF_final' gespeichert.
    - Ausgeschlossene Zeilen (mit Listen oder leeren Werten) und Duplikate werden im Ordner
      'survey-results_with-RF_final_excluded-and-doku' gespeichert.

9. Ergebnisprotokollierung:
    - Ein CSV-Dokument ('results_summary.csv') fasst die Anzahl der bereinigten, geteilten und entfernten Zeilen zusammen.
    - Hinweis: Zeilen mit Listeneinträgen erscheinen an zwei Stellen in den ausgeschlossenen Dateien:
        1. In den Listendateien (als nicht aufgeteilte Listen).
        2. In den Duplikatdateien (mit in einzelne Zeilen aufgeteilten Listen).

Ausgabe:
    - Bereinigte Umfrageergebnisse zur weiteren Analyse.
    - Dokumentation der ausgeschlossenen Zeilen und Reinigungsschritte.

Author: Joshua Tischlik
Date: 05.10.2024
"""

import os
import pandas as pd
import numpy as np
import re
import ast
import shutil


def ensure_output_directory(output_dir):
    """Erstellt das Verzeichnis, wenn es nicht existiert."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def merge_survey_files(input_dir, file, df):
    """Fügt EC_SU1.csv zu EC_SU.csv zusammen."""
    if file == 'EC_SU.csv':
        ec_su1_path = os.path.join(input_dir, 'EC_SU1.csv')
        if os.path.exists(ec_su1_path):
            df_ec_su1 = pd.read_csv(ec_su1_path)
            df = pd.concat([df, df_ec_su1], ignore_index=True)
            print(f"Dateien EC_SU.csv und EC_SU1.csv erfolgreich zusammengeführt.")
    return df


def add_missing_columns(df, required_columns, file):
    """Fügt fehlende Spalten hinzu."""
    for col in required_columns:
        if col not in df.columns:
            print(f"Spalte '{col}' fehlt in der Datei: {file}")
            df[col] = pd.NA
    return df


def resolve_conflicts(df, merge_columns):
    """Führt die Spalten zusammen und löst Konflikte bei doppelten Werten."""
    for col in merge_columns:
        if isinstance(col, str) and col.startswith('^'):
            pattern = re.compile(col)
            matching_columns = [c for c in df.columns if pattern.match(c)]
        else:
            matching_columns = [c for c in df.columns if col in c]

        if len(matching_columns) == 1:
            df[col] = df[matching_columns[0]]
        elif len(matching_columns) > 1:
            df[col] = df[matching_columns].apply(resolve_conflict, axis=1)
            df.drop(columns=matching_columns[1:], inplace=True)
        else:
            df[col] = pd.NA
    return df


def resolve_conflict(row):
    """Löst Konflikte bei doppelten Werten auf."""
    val1, val2 = row[0], row[1]
    if val1 == '-' and val2 == '-':
        return np.nan
    if val1 == '-':
        return val2
    if val2 == '-':
        return val1
    if pd.to_numeric(val1, errors='coerce') is not np.nan and pd.to_numeric(val2, errors='coerce') is not np.nan:
        if val1 != val2:
            print(f"Unterschiedliche numerische Werte in Zeile {row.name}: {val1}, {val2}")
        return max(val1, val2)
    return val1 if pd.to_numeric(val1, errors='coerce') is not np.nan else val2


def clean_survey_data(input_dir, output_dir, excluded_dir):
    """Teilt Listen auf, entfernt Duplikate und bereinigt die Daten, speichert ausgeschlossene Zeilen."""
    ensure_output_directory(output_dir)
    ensure_output_directory(excluded_dir)

    results = []

    def is_list(value):
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            try:
                value_list = ast.literal_eval(value)
                if isinstance(value_list, list):
                    return value_list
            except (ValueError, SyntaxError):
                print(f'Fehler bei {value}')
                pass
        return [value]

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_dir, file_name)
            df = pd.read_csv(file_path)
            initial_row_count = df.shape[0]
            df_cleaned = df[df['date_of_last_access'] != '-']
            removed_dash_count = initial_row_count - df_cleaned.shape[0]

            # Speichere die entfernten Zeilen (mit '-')
            excluded_file_dash_path = os.path.join(excluded_dir, f'excluded_dash_{file_name}')
            df[df['date_of_last_access'] == '-'].to_csv(excluded_file_dash_path, index=False)

            expanded_rows = []
            list_split_count = 0

            for _, row in df_cleaned.iterrows():
                row_lists = [is_list(row[col]) for col in df_cleaned.columns]
                max_len = max(len(lst) for lst in row_lists)

                if max_len > 1:
                    list_split_count += 1

                for i in range(max_len):
                    new_row = {}
                    for col, values in zip(df_cleaned.columns, row_lists):
                        new_row[col] = values[i] if i < len(values) else values[-1]
                    expanded_rows.append(new_row)

            df_expanded = pd.DataFrame(expanded_rows)

            # Überprüfung der Zeilen, die Listen enthalten
            df_list_rows = df_cleaned[df_cleaned.apply(
                lambda row: any(
                    isinstance(is_list(row[col]), list) and len(is_list(row[col])) > 1 for col in df_cleaned.columns),
                axis=1
            )]

            # Speichere die Zeilen mit Listen in eine separate Datei
            if len(df_list_rows) > 0:
                excluded_file_list_path = os.path.join(excluded_dir, f'excluded_list_{file_name}')
                df_list_rows.to_csv(excluded_file_list_path, index=False)

            duplicated_userids = df_expanded[df_expanded.duplicated(subset=['UserId'], keep=False)]
            duplicate_user_ids_count = len(duplicated_userids['UserId'].unique())

            df_expanded = df_expanded.sort_values(by='datetime').drop_duplicates(subset=['UserId'], keep='first')
            removed_duplicates_count = (len(duplicated_userids) -
                                        len(df_expanded[df_expanded['UserId'].isin(duplicated_userids['UserId'])]))

            # Speichere die entfernten doppelten Zeilen
            excluded_file_duplicate_path = os.path.join(excluded_dir, f'excluded_duplicate_{file_name}')
            duplicated_userids.to_csv(excluded_file_duplicate_path, index=False)

            results.append({
                'Datei': file_name,
                'Ursprüngliche Zeilen': initial_row_count,
                'Zeilen mit - entfernt': removed_dash_count,
                'Aufgeteilte Listen': list_split_count,
                'Doppelte UserIds vor Bereinigung': duplicate_user_ids_count,
                'Durch Duplikate entfernte Zeilen': removed_duplicates_count,
                'Verbleibende Zeilen': df_expanded.shape[0]
            })

            output_file_path = os.path.join(output_dir, file_name)
            df_expanded.to_csv(output_file_path, index=False)

    results_df = pd.DataFrame(results)

    # Speichere den results_df in einer CSV-Datei
    results_df_path = os.path.join(excluded_dir, 'results_summary.csv')
    results_df.to_csv(results_df_path, index=False)

    return results_df


def process_files(input_dir, output_dir, excluded_dir):
    """Verarbeitet die Dateien im Eingabeverzeichnis und speichert die bereinigten Dateien."""
    ensure_output_directory(output_dir)
    ensure_output_directory(excluded_dir)

    required_columns = [
        'AssessmentPhase', 'userID', 'date_of_last_access', 'datetime', 'duration'
    ]
    merge_columns = [
        'RF_aufg_interessant', 'RF_aufg_motivierend', r'^.*RF_aufgabenstellung(?!_text).*$', 'RF_aufgabenstellung_text',
        'RF_aufmerksam', 'RF_entspannt', 'RF_gereizt', 'RF_gestresst', 'RF_informiertjetzt', 'RF_interessiert',
        'RF_kennefakten', 'RF_nervos', 'RF_note', 'RF_weissviel', 'RF_wusstevorher', 'RF_zeitvorgabe'
    ]
    excluded_files = ['BEFKI.csv', 'EC_FW.csv', 'ME_FW.csv', 'EC_SU1.csv']

    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(input_dir, file)
            df = pd.read_csv(file_path)

            # Überspringe die Dateien, die ausgeschlossen werden sollen
            if file in excluded_files:
                print(f"Überspringe Datei '{file}'")
                continue

            df = merge_survey_files(input_dir, file, df)
            df = add_missing_columns(df, required_columns, file)
            df = resolve_conflicts(df, merge_columns)

            remaining_columns = [col for col in df.columns if col not in required_columns + merge_columns]
            final_columns = required_columns + merge_columns + remaining_columns
            df = df[final_columns]

            # Ändere den Header von 'RF_aufgabenstellung(?!_text)' zu 'RF_aufgabenstellung'
            if '^.*RF_aufgabenstellung(?!_text).*$' in df.columns:
                df.rename(columns={'^.*RF_aufgabenstellung(?!_text).*$': 'RF_aufgabenstellung'}, inplace=True)

            output_file_path = os.path.join(output_dir, file)
            df.to_csv(output_file_path, index=False)

    # Nach der Verarbeitung der Dateien wird die clean_survey_data Funktion aufgerufen
    results_df = clean_survey_data(output_dir, output_dir, excluded_dir)
    return results_df


def rename_and_copy_files(source_dir, target_dir):
    """
    Liest alle CSV-Dateien aus dem Quellordner ein, benennt sie basierend auf einem vordefinierten Mapping um
    und kopiert die umbenannten Dateien in einen Zielordner.

    Parameter:
    source_dir (str): Der Pfad zum Quellordner, der die CSV-Dateien enthält.
    target_dir (str): Der Pfad zum Zielordner, in den die umbenannten Dateien kopiert werden.
    """
    # Mapping für die Abkürzungen
    mapping = {
        "SurveyResults_Fachtest-Medizin": "ME_FW",
        "SurveyResults_Funkmast": "GEN-COR_FM",
        "SurveyResults_Gruene-Sosse": "GEN-COR_GS",
        "SurveyResults_Hitzestift": "GEN-COR_HS",
        "SurveyResults_Medizin-Auge": "ME_AU",
        "SurveyResults_Medizin-Kreislauf": "ME_KL",
        "SurveyResults_Medizin-Mittelohr": "ME_MO",
        "SurveyResults_Medizin_Atmung": "ME_AT",
        "SurveyResults_Nudging-Aufgabe": "EC_NU",
        "SurveyResults_Oekonomie-Grundwissen": "EC_FW",
        "SurveyResults_Piloten-Streik-Aufgabe": "EC_PS",
        "SurveyResults_Schlussfolgerndes-Denken": "BEFKI",
        "SurveyResults_Start-Up-Aufgabe": "EC_SU",
        "SurveyResults_Startup-Aufgabe": "EC_SU1",
        "SurveyResults_Tetra-Pak": "GEN-COR_TP",
        "SurveyResults_Windpark-Aufgabe": "EC_WP"
    }

    # Sicherstellen, dass der Zielordner existiert
    ensure_output_directory(target_dir)

    # Dateien im Quellordner durchlaufen und umbenennen
    for filename in os.listdir(source_dir):
        # Entfernen der Dateiendung und Vergleichen mit Mapping
        name_without_extension = filename.replace('.csv', '')

        if name_without_extension in mapping:
            new_filename = mapping[name_without_extension] + '.csv'
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, new_filename)

            # Kopiere die Datei in das Zielverzeichnis mit dem neuen Namen
            shutil.copy(source_file, target_file)
            print(f"Kopiert: {filename} -> {new_filename}")
        else:
            print(f"Kein Mapping für: {filename}")


def filter_and_remove_rows(input_folder, filter_file, filter_lesetest, directory_filtered, excluded_directory):
    """
    Filter CSV files in the input folder based on UserIds from the filter file and extra file.
    Save removed UserIds to separate files (filtered_users and filtered_lesetest) in the specified folder.
    Save the updated CSV files after filtering the rows.

    Parameters:
        input_folder (str): The path to the folder containing the CSV files.
        filter_file (str): The path to the filter file containing UserIds to be filtered.
        extra_filter_file (str): The path to the extra filter file containing additional UserIds to be filtered.
        output_removed_folder (str): The path to the folder where removed UserIds will be saved.

    Returns:
        int: Total number of rows removed across all files.
    """
    # Sicherstellen, dass die Zielordner existiert
    ensure_output_directory(directory_filtered)
    ensure_output_directory(excluded_directory)

    # Read the UserIds from the filter file and store them in a set, normalized to lowercase
    with open(filter_file, 'r') as file:
        filter_ids = {line.strip().lower() for line in file}

    # Read the UserIds from the extra filter file and normalize them to lowercase
    extra_filter_df = pd.read_csv(filter_lesetest)
    if 'UserId' in extra_filter_df.columns:
        extra_filter_ids = set(extra_filter_df['UserId'].str.lower())
    else:
        print(f"'UserId' column not found in {filter_lesetest}.")
        extra_filter_ids = set()

    # Variable to track the total number of rows removed across all files
    total_rows_removed = 0

    # Lists to track removed UserIds
    removed_user_ids_normal = []
    removed_user_ids_lesetest = []

    # Iterate over all CSV files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Check if the 'UserId' column exists
            if 'UserId' in df.columns:
                # Count the number of rows before filtering
                original_row_count = len(df)

                # Normalize the 'UserId' column to lowercase for case-insensitive matching
                df['UserId'] = df['UserId'].str.lower()

                # Separate normal filter and lesetest filter
                removed_rows_normal_df = df[df['UserId'].isin(filter_ids)]
                removed_rows_lesetest_df = df[df['UserId'].isin(extra_filter_ids)]

                # Track removed UserIds separately
                removed_user_ids_normal.extend(removed_rows_normal_df['UserId'].tolist())
                removed_user_ids_lesetest.extend(removed_rows_lesetest_df['UserId'].tolist())

                # Combine the two filters and remove rows
                filtered_df = df[~df['UserId'].isin(filter_ids | extra_filter_ids)]

                # Count the number of rows after filtering
                filtered_row_count = len(filtered_df)

                # Calculate the number of rows removed
                rows_removed = original_row_count - filtered_row_count

                # Add to the total count
                total_rows_removed += rows_removed

                # Output the number of rows removed for this file
                print(f"File: {filename}. Rows removed: {rows_removed}")

                # Save the removed rows (optional, if needed for verification or future use)
                removed_file_path = os.path.join(excluded_directory, f"removed_filter_{filename}")
                pd.concat([removed_rows_normal_df, removed_rows_lesetest_df]).to_csv(removed_file_path, index=False)

                # Save the updated CSV file after filtering
                filtered_file_path = os.path.join(directory_filtered, filename)
                filtered_df.to_csv(filtered_file_path, index=False)
            else:
                print(f"'UserId' column not found in file {filename}.")

    # Save removed UserIds from filter_users.txt to a file
    filtered_users_file = os.path.join(excluded_directory, "filtered_users.txt")
    with open(filtered_users_file, 'w') as outfile:
        for user_id in removed_user_ids_normal:
            outfile.write(user_id + '\n')

    # Save removed UserIds from filter_users_SurveyResults_ELVES_Lesetest.csv to a file
    filtered_lesetest_file = os.path.join(excluded_directory, "filtered_users_lesetest.txt")
    with open(filtered_lesetest_file, 'w') as outfile:
        for user_id in removed_user_ids_lesetest:
            outfile.write(user_id + '\n')

    # Output the total number of rows removed across all files
    print(f"Total rows removed across all files: {total_rows_removed}")


def main():
    """Hauptfunktion zur Ausführung der gesamten Verarbeitung."""
    input_directory_download = 'data/survey-results_download-unbearbeitet'
    directory_renamed = 'data/survey-results_renamed'
    directory_filtered = 'data/survey-results_renamed-filtered'
    final_output_directory = 'data/survey-results_with-RF_final'
    excluded_directory = 'data/survey-results_excluded-and-doku'

    rename_and_copy_files(input_directory_download, directory_renamed)
    filter_and_remove_rows(
        directory_renamed,
        "filter_users.txt",
        "filter_users_SurveyResults_ELVES_Lesetest.csv",
        directory_filtered,
        excluded_directory
    )

    # Verarbeite die Dateien
    results_df = process_files(directory_filtered, final_output_directory, excluded_directory)
    print(results_df)


# Aufruf der Hauptfunktion
if __name__ == '__main__':
    main()
