# -*- coding: utf-8 -*-
"""
Survey and Copy Event Data Processing Script

This script processes survey data and clipboard copy events from JSON and CSV files.
It performs the following tasks:
1. Loads copy event data from a JSON file.
2. Matches copy events to user entries in the survey data based on UserID, AssessmentPhase, and sessionID.
3. Filters out irrelevant or sensitive copy events (e.g., emails, file paths, non-language content).
4. Identifies copy events containing only links or combinations of links and text.
5. Detects whether URLs belong to search engines or AI tools.
6. Processes filtered copy events and saves the results in separate CSV files for:
   - Allowed copy entries.
   - Excluded copy entries (e.g., links, sensitive data).
7. Aggregates the total, Unipark, search, and AI tool-related copy event statistics per user and session.
8. Provides a summary of allowed and excluded copy events for each processed survey file.

The script generates the following output:
- CSV files containing matched copy entries for each survey file.
- CSV files containing excluded copy entries and their corresponding details.
- Aggregated CSV files for grouped copy events, including statistical summaries.
- A summary CSV file listing the number of allowed, excluded, and link-based copy entries per survey file.

Usage:
- Define the input folder for survey CSV files, the output folder for processed copy events, and the JSON file with copy
  event data.
- Execute the script to process, clean, and group the data.

Main Functions:
- load_copy_data(): Loads copy event data from the specified JSON file.
- match_copy_entries(): Matches copy events to survey data entries based on UserID, AssessmentPhase, and sessionID.
- is_link_only(), is_link_and_text(): Helper functions to detect link-only or mixed link-text content.
- is_search_engine_url(), is_ai_tool_url(): Detects URLs from search engines or AI tools.
- process_survey_data_with_copy(): Matches copy data with survey entries and saves results to CSV.
- clean_survey_data_with_copy(): Filters out sensitive or irrelevant copy events and saves them separately.
- group_survey_data_with_copypaste(): Groups copy events per user and session, and calculates descriptive statistics.
- main(): Orchestrates data loading, processing, cleaning, and grouping.

Author: Joshua Tischlik
Date: 23.09.2024
"""

import os
import json
import pandas as pd
import re
import urllib.parse
from langdetect import detect, LangDetectException
from preparation_text_analysis_utils import (
    is_link_only,
    is_link_and_text,
    is_search_engine_url,
    is_ai_tool_url,
    is_email_with_padding,
    is_not_language,
    is_file_path,
    is_probably_password
)


def load_copy_data(json_file):
    """Load copy data from a JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        copy_data = json.load(f)
    return copy_data


def match_copy_entries(user_id, assessment_phase, session_id, copy_data):
    """Match copy entries based on UserId, AssessmentPhase, and SessionId."""
    matched_copies = []
    for entry in copy_data:
        copy_user_id = entry['phase']['session']['user']['userID']
        copy_phase_name = entry['phase']['phase']['name']
        copy_session_id = entry['phase']['session']['webExtensionKey']
        if (
                copy_user_id == user_id
                and copy_phase_name == assessment_phase
                and copy_session_id == session_id
        ):
            matched_copies.append({
                'AssessmentPhase': assessment_phase,
                'UserId': user_id,
                'copy_id': entry['id'],
                'clipboard_content': entry['clipboardContent'],
                'timestamp': pd.to_datetime(entry['timestamp']),
                'page_url': entry['page']['url'] if entry.get('page') and entry['page'].get('url') else None,
                'page_is_unipark': entry['page']['url'].startswith('https://ww3.unipark.de/uc/core/') if entry.get('page') and entry['page'].get('url') else None,
                'page_is_search': is_search_engine_url(entry['page']['url']) if entry.get('page') and entry['page'].get('url') else None,
                'page_is_ai_tool': is_ai_tool_url(entry['page']['url']) if entry.get('page') and entry['page'].get('url') else None,
                'sessionID': session_id,
                'session_id': copy_session_id,
                'full_copy_entry': json.dumps(entry)
            })

    return matched_copies


def process_survey_data_with_copy(json_file, input_folder, output_folder):
    """
    Matches copy entries to users in survey results. Each copy entry gets its own row.
    """
    copy_data = load_copy_data(json_file)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder, file_name)
            df = pd.read_csv(file_path)
            df_cleaned = df[['AssessmentPhase', 'UserId', 'sessionID']]

            file_copy_entries = []

            for _, row in df_cleaned.iterrows():
                user_id = row['UserId']
                assessment_phase = row['AssessmentPhase']
                session_id = row['sessionID']
                copy_entries = match_copy_entries(
                    user_id, assessment_phase, session_id, copy_data
                )
                file_copy_entries.extend(copy_entries)

            copy_df = pd.DataFrame(file_copy_entries)
            output_file_path = os.path.join(output_folder, f'copy_events_{file_name}')
            copy_df.to_csv(output_file_path, index=False)
            print(f'copy entries for {file_name} saved.')


def clean_survey_data_with_copy(input_folder, output_folder_cleaned, excluded_folder):
    """
    Cleans copy entries in the already created files. Filters out certain entries
    (e.g., emails, non-language, file paths) and saves them separately.
    """
    if not os.path.exists(output_folder_cleaned):
        os.makedirs(output_folder_cleaned)
    if not os.path.exists(excluded_folder):
        os.makedirs(excluded_folder)

    results = []

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder, file_name)
            df = pd.read_csv(file_path)

            allowed_copies = []
            excluded_copies = []
            excluded_copies_links = []

            for _, row in df.iterrows():
                if pd.isna(row['clipboard_content']):
                    excluded_copies.append(row)
                    continue

                clipboard_content = str(row['clipboard_content'])

                if is_file_path(clipboard_content):
                    excluded_copies.append(row)
                    continue
                elif is_link_only(clipboard_content):
                    excluded_copies_links.append(row)
                    continue
                elif is_link_and_text(clipboard_content):
                    excluded_copies_links.append(row)
                    continue
                elif is_email_with_padding(clipboard_content):
                    excluded_copies.append(row)
                    continue
                elif is_not_language(clipboard_content):
                    excluded_copies.append(row)
                    continue
                elif is_probably_password(clipboard_content):
                    excluded_copies.append(row)
                    continue
                elif len(str(clipboard_content).split()) == 1:
                    excluded_copies.append(row)
                    continue
                else:
                    allowed_copies.append(row)

            if allowed_copies:
                allowed_df = pd.DataFrame(allowed_copies)
                allowed_output_file = os.path.join(output_folder_cleaned, file_name)
                allowed_df.to_csv(allowed_output_file, index=False)
                print(f'copy entries for {file_name} saved.')

            if excluded_copies:
                excluded_df = pd.DataFrame(excluded_copies)
                excluded_output_file = os.path.join(
                    excluded_folder, file_name.replace('.csv', '_excluded.csv')
                )
                excluded_df.to_csv(excluded_output_file, index=False)
                print(f'Excluded copy entries for {excluded_output_file} saved.')

            if excluded_copies_links:
                excluded_links_df = pd.DataFrame(excluded_copies_links)
                excluded_links_output_file = os.path.join(
                    excluded_folder, file_name.replace('.csv', '_excluded_links.csv')
                )
                excluded_links_df.to_csv(excluded_links_output_file, index=False)
                print(f'Excluded link copy entries for {excluded_links_output_file} saved.')

            results.append({
                'File': file_name,
                'Allowed Entries': len(allowed_copies),
                'Excluded Entries': len(excluded_copies),
                'Excluded Links': len(excluded_links_df)
            })

    results_df = pd.DataFrame(results)
    results_summary_file = os.path.join(excluded_folder, 'copy_events_summary.csv')
    results_df.to_csv(results_summary_file, index=False)
    print(f'Summary of copy entries saved to {results_summary_file}')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_colwidth', None)
    print(results_df)


# Schreibt describe als dic
'''
def group_survey_data_with_copypaste(input_folder, output_folder_grouped):
    """
    Gruppiert die Copy-Einträge eines Nutzers und aggregiert die Variablen für Total,
    Unipark, Search und AI Tool für jede der angegebenen Variablen.
    Ergänzt die Statistiken für String Length und Word Count mit den .describe()-Methoden.
    """
    if not os.path.exists(output_folder_grouped):
        os.makedirs(output_folder_grouped)

    results = []

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder, file_name)
            df = pd.read_csv(file_path)

            # Variablen initialisieren
            grouped_data = {}

            # Gruppiere nach UserId, AssessmentPhase und sessionID
            grouped = df.groupby(['UserID', 'AssessmentPhase', 'sessionID'])

            for (user_id, assessment_phase, session_id), group in grouped:
                # Initialisiere das Dictionary für den Nutzer, falls noch nicht vorhanden
                if (user_id, assessment_phase, session_id) not in grouped_data:
                    grouped_data[(user_id, assessment_phase, session_id)] = {
                        'UserID': user_id,
                        'AssessmentPhase': assessment_phase,
                        'sessionID': session_id,
                        'total_string_len': [],
                        'total_string_len_describe': {},
                        'total_word_count': [],
                        'total_word_count_describe': {},
                        'unipark_string_len': [],
                        'unipark_string_len_describe': {},
                        'unipark_word_count': [],
                        'unipark_word_count_describe': {},
                        'search_string_len': [],
                        'search_string_len_describe': {},
                        'search_word_count': [],
                        'search_word_count_describe': {},
                        'ai_tool_string_len': [],
                        'ai_tool_string_len_describe': {},
                        'ai_tool_word_count': [],
                        'ai_tool_word_count_describe': {},
                        'timestamps_total': [],
                        'timestamps_unipark': [],
                        'timestamps_search': [],
                        'timestamps_ai_tool': []
                    }

                # Iteriere durch die Copy-Einträge in der Gruppe
                for _, row in group.iterrows():
                    clipboard_content = row['clipboard_content']
                    content_length = len(str(clipboard_content)) if pd.notna(clipboard_content) else 0
                    word_count = len(str(clipboard_content).split()) if pd.notna(clipboard_content) else 0
                    timestamp = row['timestamp']

                    # Gesamtwerte (Total)
                    grouped_data[(user_id, assessment_phase, session_id)]['total_string_len'].append(content_length)
                    grouped_data[(user_id, assessment_phase, session_id)]['total_word_count'].append(word_count)
                    grouped_data[(user_id, assessment_phase, session_id)]['timestamps_total'].append(timestamp)

                    # Unipark
                    if row['page_is_unipark']:
                        grouped_data[(user_id, assessment_phase, session_id)]['unipark_string_len'].append(content_length)
                        grouped_data[(user_id, assessment_phase, session_id)]['unipark_word_count'].append(word_count)
                        grouped_data[(user_id, assessment_phase, session_id)]['timestamps_unipark'].append(timestamp)

                    # Search
                    if row['page_is_search']:
                        grouped_data[(user_id, assessment_phase, session_id)]['search_string_len'].append(content_length)
                        grouped_data[(user_id, assessment_phase, session_id)]['search_word_count'].append(word_count)
                        grouped_data[(user_id, assessment_phase, session_id)]['timestamps_search'].append(timestamp)

                    # AI Tool
                    if row['page_is_ai_tool']:
                        grouped_data[(user_id, assessment_phase, session_id)]['ai_tool_string_len'].append(content_length)
                        grouped_data[(user_id, assessment_phase, session_id)]['ai_tool_word_count'].append(word_count)
                        grouped_data[(user_id, assessment_phase, session_id)]['timestamps_ai_tool'].append(timestamp)

            # Füge describe()-Statistiken hinzu
            def describe_statistics(data):
                if data:
                    return pd.Series(data).describe().to_dict()
                else:
                    return pd.Series([0]).describe().to_dict()

            for key, aggregated in grouped_data.items():
                # Berechnung der describe-Statistiken für total, unipark, search und ai_tool
                aggregated['total_string_len_describe'] = describe_statistics(aggregated['total_string_len'])
                aggregated['total_word_count_describe'] = describe_statistics(aggregated['total_word_count'])
                aggregated['unipark_string_len_describe'] = describe_statistics(aggregated['unipark_string_len'])
                aggregated['unipark_word_count_describe'] = describe_statistics(aggregated['unipark_word_count'])
                aggregated['search_string_len_describe'] = describe_statistics(aggregated['search_string_len'])
                aggregated['search_word_count_describe'] = describe_statistics(aggregated['search_word_count'])
                aggregated['ai_tool_string_len_describe'] = describe_statistics(aggregated['ai_tool_string_len'])
                aggregated['ai_tool_word_count_describe'] = describe_statistics(aggregated['ai_tool_word_count'])

                # Berechne und überschreibe die Listen mit Summen:
                aggregated['total_string_len'] = sum(aggregated['total_string_len'])
                aggregated['total_word_count'] = sum(aggregated['total_word_count'])
                aggregated['unipark_string_len'] = sum(aggregated['unipark_string_len'])
                aggregated['unipark_word_count'] = sum(aggregated['unipark_word_count'])
                aggregated['search_string_len'] = sum(aggregated['search_string_len'])
                aggregated['search_word_count'] = sum(aggregated['search_word_count'])
                aggregated['ai_tool_string_len'] = sum(aggregated['ai_tool_string_len'])
                aggregated['ai_tool_word_count'] = sum(aggregated['ai_tool_word_count'])

                # Füge die aggregierten Ergebnisse zur Result-Liste hinzu
                results.append(aggregated)

            # Erstelle ein DataFrame mit den aggregierten Ergebnissen und speichere es
            grouped_df = pd.DataFrame(results)
            output_file_path = os.path.join(output_folder_grouped, f'{file_name}')
            grouped_df.to_csv(output_file_path, index=False)
            print(f'Grouped copy entries for {file_name} saved to {output_file_path}')
'''


def group_survey_data_with_copypaste(input_folder, output_folder_grouped):
    """
    Gruppiert die Copy-Einträge eines Nutzers und aggregiert die Variablen für Total,
    Unipark, Search und AI Tool für jede der angegebenen Variablen.
    Ergänzt die Statistiken für String Length und Word Count mit den .describe()-Methoden.
    """
    if not os.path.exists(output_folder_grouped):
        os.makedirs(output_folder_grouped)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder, file_name)
            df = pd.read_csv(file_path)

            # Variablen initialisieren
            grouped_data = {}
            results = []

            # Gruppiere nach UserId, AssessmentPhase und sessionID
            grouped = df.groupby(['UserId', 'AssessmentPhase', 'sessionID'])

            for (user_id, assessment_phase, session_id), group in grouped:
                # Initialisiere das Dictionary für den Nutzer, falls noch nicht vorhanden
                if (user_id, assessment_phase, session_id) not in grouped_data:
                    grouped_data[(user_id, assessment_phase, session_id)] = {
                        'UserId': user_id,
                        'AssessmentPhase': assessment_phase,
                        'sessionID': session_id,
                        'total_string_len': [],
                        'total_word_count': [],
                        'unipark_string_len': [],
                        'unipark_word_count': [],
                        'search_string_len': [],
                        'search_word_count': [],
                        'ai_tool_string_len': [],
                        'ai_tool_word_count': [],
                        'timestamps_total': [],
                        'timestamps_unipark': [],
                        'timestamps_search': [],
                        'timestamps_ai_tool': []
                    }

                # Iteriere durch die Copy-Einträge in der Gruppe
                for _, row in group.iterrows():
                    clipboard_content = row['clipboard_content']
                    content_length = len(str(clipboard_content))
                    word_count = len(str(clipboard_content).split())
                    timestamp = row['timestamp']

                    # Gesamtwerte (Total)
                    grouped_data[(user_id, assessment_phase, session_id)]['total_string_len'].append(content_length)
                    grouped_data[(user_id, assessment_phase, session_id)]['total_word_count'].append(word_count)
                    grouped_data[(user_id, assessment_phase, session_id)]['timestamps_total'].append(timestamp)

                    # Unipark
                    if row['page_is_unipark']:
                        grouped_data[(user_id, assessment_phase, session_id)]['unipark_string_len'].append(content_length)
                        grouped_data[(user_id, assessment_phase, session_id)]['unipark_word_count'].append(word_count)
                        grouped_data[(user_id, assessment_phase, session_id)]['timestamps_unipark'].append(timestamp)

                    # Search
                    if row['page_is_search']:
                        grouped_data[(user_id, assessment_phase, session_id)]['search_string_len'].append(content_length)
                        grouped_data[(user_id, assessment_phase, session_id)]['search_word_count'].append(word_count)
                        grouped_data[(user_id, assessment_phase, session_id)]['timestamps_search'].append(timestamp)

                    # AI Tool
                    if row['page_is_ai_tool']:
                        grouped_data[(user_id, assessment_phase, session_id)]['ai_tool_string_len'].append(content_length)
                        grouped_data[(user_id, assessment_phase, session_id)]['ai_tool_word_count'].append(word_count)
                        grouped_data[(user_id, assessment_phase, session_id)]['timestamps_ai_tool'].append(timestamp)

            # Füge describe()-Statistiken hinzu und konvertiere in Spalten
            def describe_statistics(data, prefix):
                stats = pd.Series(data).describe() #if data else pd.Series([0]).describe()
                if not data:  # Leere Datenliste
                    return {
                        f'{prefix}_count': 0,
                        f'{prefix}_mean': 0,
                        f'{prefix}_std': 0,
                        f'{prefix}_min': 0,
                        f'{prefix}_25': 0,
                        f'{prefix}_50': 0,
                        f'{prefix}_75': 0,
                        f'{prefix}_max': 0
                    }
                if pd.isna(stats['std']):
                    stats['std'] = 0
                return {
                    f'{prefix}_count': stats['count'],
                    f'{prefix}_mean': stats['mean'],
                    f'{prefix}_std': stats['std'],
                    f'{prefix}_min': stats['min'],
                    f'{prefix}_25': stats['25%'],
                    f'{prefix}_50': stats['50%'],
                    f'{prefix}_75': stats['75%'],
                    f'{prefix}_max': stats['max']
                }

            for key, aggregated in grouped_data.items():
                # Berechnung und Umwandlung der describe-Statistiken für total, unipark, search und ai_tool
                aggregated.update(describe_statistics(aggregated['total_string_len'], 'total_string_len_describe'))
                aggregated.update(describe_statistics(aggregated['total_word_count'], 'total_word_count_describe'))
                aggregated.update(describe_statistics(aggregated['unipark_string_len'], 'unipark_string_len_describe'))
                aggregated.update(describe_statistics(aggregated['unipark_word_count'], 'unipark_word_count_describe'))
                aggregated.update(describe_statistics(aggregated['search_string_len'], 'search_string_len_describe'))
                aggregated.update(describe_statistics(aggregated['search_word_count'], 'search_word_count_describe'))
                aggregated.update(describe_statistics(aggregated['ai_tool_string_len'], 'ai_tool_string_len_describe'))
                aggregated.update(describe_statistics(aggregated['ai_tool_word_count'], 'ai_tool_word_count_describe'))

                # Berechne und überschreibe die Listen mit Summen:
                aggregated['total_string_len'] = sum(aggregated['total_string_len'])
                aggregated['total_word_count'] = sum(aggregated['total_word_count'])
                aggregated['unipark_string_len'] = sum(aggregated['unipark_string_len'])
                aggregated['unipark_word_count'] = sum(aggregated['unipark_word_count'])
                aggregated['search_string_len'] = sum(aggregated['search_string_len'])
                aggregated['search_word_count'] = sum(aggregated['search_word_count'])
                aggregated['ai_tool_string_len'] = sum(aggregated['ai_tool_string_len'])
                aggregated['ai_tool_word_count'] = sum(aggregated['ai_tool_word_count'])

                # Füge die aggregierten Ergebnisse zur Result-Liste hinzu
                results.append(aggregated)

            # Erstelle ein DataFrame mit den aggregierten Ergebnissen und speichere es
            grouped_df = pd.DataFrame(results)
            output_file_path = os.path.join(output_folder_grouped, f'{file_name}')
            grouped_df.to_csv(output_file_path, index=False)
            print(f'Grouped copy entries for {file_name} saved to {output_file_path}')


def main():
    """
    Main function to define directories, process, and clean survey data with
    copy entries, then save the results.
    """
    json_file = 'data/copy_451.json'
    input_folder = 'data/survey-results_with-RF_final'
    output_folder = 'data/copy-events_with-RF'
    output_folder_cleaned = 'data/copy-events_with-RF_cleaned'
    excluded_folder = 'data/copy-events_with-RF_excluded'
    output_folder_grouped = 'data/copy-events_with-RF_grouped'

    #process_survey_data_with_copy(json_file, input_folder, output_folder)
    #clean_survey_data_with_copy(output_folder, output_folder_cleaned, excluded_folder)
    group_survey_data_with_copypaste(output_folder_cleaned, output_folder_grouped)


if __name__ == '__main__':
    main()

#%%
