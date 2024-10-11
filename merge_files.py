# 1. Datenzusammenführung und Auffüllen

import os
import pandas as pd

# Verzeichnisse für die verschiedenen Events
directories = {
    "survey_results": "survey-results_with-RF_final",
    "copy_events": "copy-events_with-RF_grouped",
    "paste_events": "paste-events_with-RF_grouped",
    "keystroke_events": "keystroke-events_with-RF"
}

# Identifikationsspalten für das Zusammenführen
merge_columns = ['AssessmentPhase', 'UserId', 'sessionID']

# Festlegen, welche Spalten für jeden Eventtyp übernommen werden sollen
selected_columns_per_event = {
    "survey_results": [
        "date_of_last_access", "datetime", "duration", "RF_aufg_interessant",
        "RF_aufg_motivierend", "RF_aufgabenstellung", "RF_aufgabenstellung_text",
        "RF_aufmerksam", "RF_entspannt", "RF_gereizt", "RF_gestresst",
        "RF_informiertjetzt", "RF_interessiert", "RF_kennefakten",
        "RF_nervos", "RF_note", "RF_weissviel", "RF_wusstevorher",
        "RF_zeitvorgabe"
    ],
    "copy_events": [
        "ai_tool_string_len", "ai_tool_string_len_describe_25", "ai_tool_string_len_describe_50",
        "ai_tool_string_len_describe_75", "ai_tool_string_len_describe_count",
        "ai_tool_string_len_describe_max", "ai_tool_string_len_describe_mean",
        "ai_tool_string_len_describe_min", "ai_tool_string_len_describe_std",
        "ai_tool_word_count", "ai_tool_word_count_describe_25", "ai_tool_word_count_describe_50",
        "ai_tool_word_count_describe_75", "ai_tool_word_count_describe_count",
        "ai_tool_word_count_describe_max", "ai_tool_word_count_describe_mean",
        "ai_tool_word_count_describe_min", "ai_tool_word_count_describe_std",
        "search_string_len", "search_string_len_describe_25", "search_string_len_describe_50",
        "search_string_len_describe_75", "search_string_len_describe_count",
        "search_string_len_describe_max", "search_string_len_describe_mean",
        "search_string_len_describe_min", "search_string_len_describe_std",
        "search_word_count", "search_word_count_describe_25", "search_word_count_describe_50",
        "search_word_count_describe_75", "search_word_count_describe_count",
        "search_word_count_describe_max", "search_word_count_describe_mean",
        "search_word_count_describe_min", "search_word_count_describe_std",
        "total_string_len", "total_string_len_describe_25", "total_string_len_describe_50",
        "total_string_len_describe_75", "total_string_len_describe_count",
        "total_string_len_describe_max", "total_string_len_describe_mean",
        "total_string_len_describe_min", "total_string_len_describe_std",
        "total_word_count", "total_word_count_describe_25", "total_word_count_describe_50",
        "total_word_count_describe_75", "total_word_count_describe_count",
        "total_word_count_describe_max", "total_word_count_describe_mean",
        "total_word_count_describe_min", "total_word_count_describe_std",
        "unipark_string_len", "unipark_string_len_describe_25", "unipark_string_len_describe_50",
        "unipark_string_len_describe_75", "unipark_string_len_describe_count",
        "unipark_string_len_describe_max", "unipark_string_len_describe_mean",
        "unipark_string_len_describe_min", "unipark_string_len_describe_std",
        "unipark_word_count", "unipark_word_count_describe_25", "unipark_word_count_describe_50",
        "unipark_word_count_describe_75", "unipark_word_count_describe_count",
        "unipark_word_count_describe_max", "unipark_word_count_describe_mean",
        "unipark_word_count_describe_min", "unipark_word_count_describe_std"
    ],
    "paste_events": [
        "ai_tool_string_len", "ai_tool_string_len_describe_25", "ai_tool_string_len_describe_50",
        "ai_tool_string_len_describe_75", "ai_tool_string_len_describe_count",
        "ai_tool_string_len_describe_max", "ai_tool_string_len_describe_mean",
        "ai_tool_string_len_describe_min", "ai_tool_string_len_describe_std",
        "ai_tool_word_count", "ai_tool_word_count_describe_25", "ai_tool_word_count_describe_50",
        "ai_tool_word_count_describe_75", "ai_tool_word_count_describe_count",
        "ai_tool_word_count_describe_max", "ai_tool_word_count_describe_mean",
        "ai_tool_word_count_describe_min", "ai_tool_word_count_describe_std",
        "search_string_len", "search_string_len_describe_25", "search_string_len_describe_50",
        "search_string_len_describe_75", "search_string_len_describe_count",
        "search_string_len_describe_max", "search_string_len_describe_mean",
        "search_string_len_describe_min", "search_string_len_describe_std",
        "search_word_count", "search_word_count_describe_25", "search_word_count_describe_50",
        "search_word_count_describe_75", "search_word_count_describe_count",
        "search_word_count_describe_max", "search_word_count_describe_mean",
        "search_word_count_describe_min", "search_word_count_describe_std",
        "total_string_len", "total_string_len_describe_25", "total_string_len_describe_50",
        "total_string_len_describe_75", "total_string_len_describe_count",
        "total_string_len_describe_max", "total_string_len_describe_mean",
        "total_string_len_describe_min", "total_string_len_describe_std",
        "total_word_count", "total_word_count_describe_25", "total_word_count_describe_50",
        "total_word_count_describe_75", "total_word_count_describe_count",
        "total_word_count_describe_max", "total_word_count_describe_mean",
        "total_word_count_describe_min", "total_word_count_describe_std",
        "unipark_string_len", "unipark_string_len_describe_25", "unipark_string_len_describe_50",
        "unipark_string_len_describe_75", "unipark_string_len_describe_count",
        "unipark_string_len_describe_max", "unipark_string_len_describe_mean",
        "unipark_string_len_describe_min", "unipark_string_len_describe_std",
        "unipark_word_count", "unipark_word_count_describe_25", "unipark_word_count_describe_50",
        "unipark_word_count_describe_75", "unipark_word_count_describe_count",
        "unipark_word_count_describe_max", "unipark_word_count_describe_mean",
        "unipark_word_count_describe_min", "unipark_word_count_describe_std"
    ],
    "keystroke_events": [
        'keystrokes_total',
        'text_keystrokes_total',
        'keystrokes_unipark',
        'text_keystrokes_unipark',
        'keystrokes_search',
        'text_keystrokes_search',
        'keystrokes_ai',
        'text_keystrokes_ai'
    ]
}

# Funktion zum Laden und Filtern der Spalten
def load_and_filter_columns(file_path, selected_columns):
    df = pd.read_csv(file_path)

    # Sicherstellen, dass die gewünschten Spalten in der Datei vorhanden sind
    available_columns = [col for col in selected_columns if col in df.columns]

    # Falls einige Spalten fehlen, den Benutzer informieren
    missing_columns = set(selected_columns) - set(available_columns)
    if missing_columns:
        print(f"Warnung: Die folgenden Spalten fehlen in der Datei {file_path}: {missing_columns}")

    # Zurückgeben nur der ausgewählten Spalten + merge_columns
    return df[available_columns + merge_columns]

# Funktion zum Zusammenführen von zusammengehörenden Dateien
def merge_files(survey_file_base):
    merged_df = None

    for event_type, directory in directories.items():
        # Passe den Dateinamen an, um den korrekten Dateinamen ohne doppelten Präfix zu erhalten
        if event_type == 'survey_results':
            # Hier wird der Präfix für Survey-Dateien entfernt
            file_name = f"{directory}/{survey_file_base}.csv"
        elif event_type == 'keystroke_events':
            file_name = f"{directory}/keystroke_{survey_file_base}.csv"
        else:
            # Für copy paste
            file_name = f"{directory}/{event_type}_{survey_file_base}.csv"

        if os.path.exists(file_name):
            selected_columns = selected_columns_per_event.get(event_type, [])
            df = load_and_filter_columns(file_name, selected_columns)

            if event_type in ['copy_events', 'paste_events']:
                # Wähle alle Spalten außer den merge_columns und füge nur für diese das Präfix hinzu
                columns_to_prefix = [col for col in df.columns if col not in merge_columns]
                if event_type == 'copy_events':
                    df.rename(columns={col: f"copy_{col}" for col in columns_to_prefix}, inplace=True)
                elif event_type == 'paste_events':
                    df.rename(columns={col: f"paste_{col}" for col in columns_to_prefix}, inplace=True)

            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on=merge_columns, how='outer')

    # Nachdem alle Daten zusammengeführt wurden, füllen wir leere Zellen auf
    if merged_df is not None:
        # Fülle leere Zellen mit 0, außer bei Spalten, die 'timestamp' im Namen enthalten
        for col in merged_df.columns:
            if 'timestamp' not in col.lower():
                merged_df[col] = merged_df[col].fillna(0)
    return merged_df

# Liste der zu verarbeitenden Basisdateinamen (aus deinem ersten Screenshot)
survey_file_bases = [
    "EC_NU", "EC_PS", "EC_SU", "EC_WP", "GEN-COR_FM", "GEN-COR_GS",
    "GEN-COR_HS", "GEN-COR_TP", "ME_AT", "ME_AU", "ME_KL", "ME_MO"
]

# Verzeichnis für das Speichern der zusammengeführten Dateien
output_directory = "merged_files"
os.makedirs(output_directory, exist_ok=True)

# Zusammenführung der Dateien für jede Basisdatei
for base in survey_file_bases:
    merged_df = merge_files(base)
    if merged_df is not None:
        output_file = f"{output_directory}/{base}.csv"
        merged_df.to_csv(output_file, index=False)
        print(f"Zusammengeführte Datei gespeichert: {output_file}")

#%%
