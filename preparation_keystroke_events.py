"""
This script processes keystroke data from JSON files and survey data from CSV files.
It counts the total and text-specific keystrokes for users during different phases of assessments.
Keystroke data is categorized by URL types (e.g., Unipark, search engines, AI tools).
The results, including keystroke timestamps, are saved in new CSV files.

Functions:
    - load_keystroke_data: Loads keystroke data from a JSON file.
    - count_keystrokes: Counts total keystrokes and categorizes them based on URL types.
    - process_survey_data_with_keystrokes: Processes survey data and adds keystroke statistics.
    - main: Main function that handles the processing of data and saves the results.

Author: Joshua Tischlik
Date: 23.09.2024
""""""
This script processes keystroke data from JSON files and survey data from CSV files.
It counts the total and text-specific keystrokes for users during different phases of assessments.
Keystroke data is categorized by URL types (e.g., Unipark, search engines, AI tools).
The results, including keystroke timestamps, are saved in new CSV files.

Functions:
    - load_keystroke_data: Loads keystroke data from a JSON file.
    - count_keystrokes: Counts total keystrokes and categorizes them based on URL types.
    - process_survey_data_with_keystrokes: Processes survey data and adds keystroke statistics.
    - main: Main function that handles the processing of data and saves the results.

Author: Joshua Tischlik
Date: 23.09.2024
"""

import os
import json
import pandas as pd
from preparation_text_analysis_utils import (
    is_search_engine_url,
    is_ai_tool_url
)
import ast
import csv
from datetime import datetime
import sys
import re


def load_keystroke_data(json_file):
    """Lade Keystroke-Daten aus einer JSON-Datei."""
    with open(json_file, 'r', encoding='utf-8') as f:
        keystroke_data = json.load(f)
    return keystroke_data


def count_keystrokes(keystroke_data, user_id, assessment_phase, session_id):
    """
    Zähle Keystrokes und Text-Keystrokes für einen bestimmten Benutzer.
    Berücksichtige Unipark-, Search- und AI-Seiten und speichere die entsprechenden Timestamps.
    """
    # Definiere die Liste der Text-Keystrokes
    text_keystrokes_list = [
        'Backquote', 'Backslash', 'BracketLeft', 'BracketRight', 'Comma', 'Digit0', 'Digit1',
        'Digit2', 'Digit3', 'Digit4', 'Digit5', 'Digit6', 'Digit7', 'Digit8', 'Digit9', 'Equal',
        'IntlBackslash', 'KeyA', 'KeyB', 'KeyC', 'KeyD', 'KeyE', 'KeyF', 'KeyG', 'KeyH', 'KeyI',
        'KeyJ', 'KeyK', 'KeyL', 'KeyM', 'KeyN', 'KeyO', 'KeyP', 'KeyQ', 'KeyR', 'KeyS', 'KeyT',
        'KeyU', 'KeyV', 'KeyW', 'KeyX', 'KeyY', 'KeyZ', 'Minus', 'Numpad0', 'Numpad1', 'Numpad2',
        'Numpad3', 'Numpad4', 'Numpad5', 'Numpad6', 'Numpad7', 'Numpad8', 'Numpad9', 'NumpadAdd',
        'NumpadDecimal', 'NumpadDivide', 'NumpadMultiply', 'NumpadSubtract', 'Period', 'Quote',
        'Semicolon', 'Slash', 'Space'
    ]

    # Variablen für die Anzahl der Keystrokes und Timestamps
    total_keystrokes = 0
    text_keystrokes = 0
    unipark_keystrokes = 0
    search_keystrokes = 0
    ai_keystrokes = 0
    unipark_text_keystrokes = 0
    search_text_keystrokes = 0
    ai_text_keystrokes = 0

    # Timestamplisten
    total_keystrokes_timestamps = []
    text_keystrokes_timestamps = []
    unipark_text_keystrokes_timestamps = []
    search_text_keystrokes_timestamps = []
    ai_text_keystrokes_timestamps = []

    for entry in keystroke_data:
        keystroke_user_id = entry['phase']['session']['user']['userID']
        keystroke_phase_name = entry['phase']['phase']['name']
        keystroke_session_id = entry['phase']['session']['webExtensionKey']
        if (
                keystroke_user_id == user_id
                and keystroke_phase_name == assessment_phase
                and keystroke_session_id == session_id
        ):
            key_code = entry['keyCode']
            timestamp = pd.to_datetime(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f%z')

            total_keystrokes += 1
            total_keystrokes_timestamps.append(timestamp)  # Timestamp hinzufügen

            if key_code in text_keystrokes_list:
                text_keystrokes += 1
                text_keystrokes_timestamps.append(timestamp)  # Timestamp hinzufügen

            if entry.get('page') and entry['page'].get('url'):
                entry_url = entry['page']['url']

                # Unipark-Seiten
                if entry_url.startswith('https://ww3.unipark.de/uc/core/'):
                    unipark_keystrokes += 1
                    if key_code in text_keystrokes_list:
                        unipark_text_keystrokes += 1
                        unipark_text_keystrokes_timestamps.append(timestamp)  # Timestamp hinzufügen

                # Search-Seiten
                if is_search_engine_url(entry_url):
                    search_keystrokes += 1
                    if key_code in text_keystrokes_list:
                        search_text_keystrokes += 1
                        search_text_keystrokes_timestamps.append(timestamp)  # Timestamp hinzufügen

                # AI-Tools-Seiten
                if is_ai_tool_url(entry_url):
                    ai_keystrokes += 1
                    if key_code in text_keystrokes_list:
                        ai_text_keystrokes += 1
                        ai_text_keystrokes_timestamps.append(timestamp)  # Timestamp hinzufügen

    return {
        'keystrokes_total': total_keystrokes,
        'text_keystrokes_total': text_keystrokes,
        'keystrokes_unipark': unipark_keystrokes,
        'text_keystrokes_unipark': unipark_text_keystrokes,
        'keystrokes_search': search_keystrokes,
        'text_keystrokes_search': search_text_keystrokes,
        'keystrokes_ai': ai_keystrokes,
        'text_keystrokes_ai': ai_text_keystrokes,
        'keystrokes_total_timestamps': total_keystrokes_timestamps,
        'text_keystrokes_total_timestamps': text_keystrokes_timestamps,
        'text_keystrokes_unipark_timestamps': unipark_text_keystrokes_timestamps,
        'text_keystrokes_search_timestamps': search_text_keystrokes_timestamps,
        'text_keystrokes_ai_timestamps': ai_text_keystrokes_timestamps
    }


def process_survey_data_with_keystrokes(json_folder, input_folder, output_folder):
    """
    Verarbeite Keystroke-Daten und speichere die Ergebnisse in einer CSV-Datei.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Bearbeite jede CSV-Datei im input_folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            input_file_path = os.path.join(input_folder, file_name)
            df = pd.read_csv(input_file_path)
            df_cleaned = df[['AssessmentPhase', 'UserId', 'sessionID']]  # Kopiere nur relevante Spalten

            keystroke_data = load_keystroke_data(f'{json_folder}/keystroke_451_{file_name[:-4]}.json')

            # Neue Spalten für die Keystroke-Daten
            keystroke_results = []

            for _, row in df_cleaned.iterrows():
                user_id = row['UserId']
                assessment_phase = row['AssessmentPhase']
                session_id = row['sessionID']

                user_keystrokes = count_keystrokes(keystroke_data, user_id, assessment_phase, session_id)

                keystroke_results.append(user_keystrokes)

            keystroke_df = pd.DataFrame(keystroke_results)

            # Kombiniere die Daten
            output_df = pd.concat([df_cleaned, keystroke_df], axis=1)

            # Speichere die aktualisierte CSV-Datei mit dem Präfix 'keystroke_'
            output_file_path = os.path.join(output_folder, f'keystroke_{file_name}')
            output_df.to_csv(output_file_path, index=False)
            print(f'Keystroke-Daten für {file_name} gespeichert.')


def main():
    """
    Hauptfunktion zum Verarbeiten der Keystroke-Daten.
    """

    json_folder = 'data/keystroke_451_renamed-merged'
    input_folder = 'data/survey-results_with-RF_final'
    output_folder = 'data/keystroke-events_with-RF'

    process_survey_data_with_keystrokes(json_folder, input_folder, output_folder)


if __name__ == '__main__':
    main()

#%%
