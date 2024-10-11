# -*- coding: utf-8 -*-
"""
Survey and Paste Event Data Processing Script

This script processes survey data and clipboard paste events from JSON and CSV files.
It performs the following tasks:
1. Loads paste event data from a JSON file.
2. Matches paste events to user entries in the survey data based on UserID, AssessmentPhase, and sessionID.
3. Filters out irrelevant or sensitive paste events (e.g., emails, file paths, non-language content).
4. Identifies paste events containing only links or combinations of links and text.
5. Detects whether URLs belong to search engines or AI tools.
6. Processes filtered paste events and saves the results in separate CSV files for:
   - Allowed paste entries.
   - Excluded paste entries (e.g., links, sensitive data).
7. Aggregates the total, Unipark, search, and AI tool-related paste event statistics per user and session.
8. Provides a summary of allowed and excluded paste events for each processed survey file.

The script generates the following output:
- CSV files containing matched paste entries for each survey file.
- CSV files containing excluded paste entries and their corresponding details.
- Aggregated CSV files for grouped paste events, including statistical summaries.
- A summary CSV file listing the number of allowed, excluded, and link-based paste entries per survey file.

Usage:
- Define the input folder for survey CSV files, the output folder for processed paste events, and the JSON file with paste event data.
- Execute the script to process, clean, and group the data.

Main Functions:
- load_paste_data(): Loads paste event data from the specified JSON file.
- match_paste_entries(): Matches paste events to survey data entries based on UserID, AssessmentPhase, and sessionID.
- is_link_only(), is_link_and_text(): Helper functions to detect link-only or mixed link-text content.
- is_search_engine_url(), is_ai_tool_url(): Detects URLs from search engines or AI tools.
- process_survey_data_with_paste(): Matches paste data with survey entries and saves results to CSV.
- clean_survey_data_with_paste(): Filters out sensitive or irrelevant paste events and saves them separately.
- group_survey_data_with_copypaste(): Groups paste events per user and session, and calculates descriptive statistics.
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
from preparation_copy_events import group_survey_data_with_copypaste


def load_paste_data(json_file):
    """Load paste data from a JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        paste_data = json.load(f)
    return paste_data


def match_paste_entries(user_id, assessment_phase, session_id, paste_data):
    """Match paste entries based on UserId, AssessmentPhase, and SessionId."""
    matched_pastes = []
    for entry in paste_data:
        paste_user_id = entry['page']['phase']['session']['user']['userID']
        paste_phase_name = entry['page']['phase']['phase']['name']
        paste_session_id = entry['page']['phase']['session']['webExtensionKey']
        if (
                paste_user_id == user_id
                and paste_phase_name == assessment_phase
                and paste_session_id == session_id
        ):
            matched_pastes.append({
                'AssessmentPhase': assessment_phase,
                'UserId': user_id,
                'paste_id': entry['id'],
                'clipboard_content': entry['clipboardContent'],
                'timestamp': pd.to_datetime(entry['timestamp']),
                'page_url': entry['page']['url'],
                'page_is_unipark': entry['page']['url'].startswith('https://ww3.unipark.de/uc/core/'),
                'page_is_search': is_search_engine_url(entry['page']['url']),
                'page_is_ai_tool': is_ai_tool_url(entry['page']['url']),
                'sessionID': session_id,
                'session_id': paste_session_id,
                'full_paste_entry': json.dumps(entry)
            })
    return matched_pastes


def process_survey_data_with_paste(json_file, input_folder, output_folder):
    """
    Matches paste entries to users in survey results. Each paste entry gets its own row.
    """
    paste_data = load_paste_data(json_file)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder, file_name)
            df = pd.read_csv(file_path)
            df_cleaned = df[['AssessmentPhase', 'UserId', 'sessionID']]

            file_paste_entries = []

            for _, row in df_cleaned.iterrows():
                user_id = row['UserId']
                assessment_phase = row['AssessmentPhase']
                session_id = row['sessionID']
                paste_entries = match_paste_entries(
                    user_id, assessment_phase, session_id, paste_data
                )
                file_paste_entries.extend(paste_entries)

            paste_df = pd.DataFrame(file_paste_entries)
            output_file_path = os.path.join(output_folder, f'paste_events_{file_name}')
            paste_df.to_csv(output_file_path, index=False)
            print(f'Paste entries for {file_name} saved.')


def clean_survey_data_with_paste(input_folder, output_folder_cleaned, excluded_folder):
    """
    Cleans paste entries in the already created files. Filters out certain entries
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

            allowed_pastes = []
            excluded_pastes = []
            excluded_pastes_links = []

            for _, row in df.iterrows():
                if pd.isna(row['clipboard_content']):
                    excluded_pastes.append(row)
                    continue

                clipboard_content = str(row['clipboard_content'])

                if is_file_path(clipboard_content):
                    excluded_pastes.append(row)
                    continue
                elif is_link_only(clipboard_content):
                    excluded_pastes_links.append(row)
                    continue
                elif is_link_and_text(clipboard_content):
                    excluded_pastes_links.append(row)
                    continue
                elif is_email_with_padding(clipboard_content):
                    excluded_pastes.append(row)
                    continue
                elif is_not_language(clipboard_content):
                    excluded_pastes.append(row)
                    continue
                elif is_probably_password(clipboard_content):
                    excluded_pastes.append(row)
                    continue
                elif len(str(clipboard_content).split()) == 1:
                    excluded_pastes.append(row)
                    continue
                else:
                    allowed_pastes.append(row)

            if allowed_pastes:
                allowed_df = pd.DataFrame(allowed_pastes)
                allowed_output_file = os.path.join(output_folder_cleaned, file_name)
                allowed_df.to_csv(allowed_output_file, index=False)
                print(f'Paste entries for {file_name} saved.')

            if excluded_pastes:
                excluded_df = pd.DataFrame(excluded_pastes)
                excluded_output_file = os.path.join(
                    excluded_folder, file_name.replace('.csv', '_excluded.csv')
                )
                excluded_df.to_csv(excluded_output_file, index=False)
                print(f'Excluded paste entries for {excluded_output_file} saved.')

            if excluded_pastes_links:
                excluded_links_df = pd.DataFrame(excluded_pastes_links)
                excluded_links_output_file = os.path.join(
                    excluded_folder, file_name.replace('.csv', '_excluded_links.csv')
                )
                excluded_links_df.to_csv(excluded_links_output_file, index=False)
                print(f'Excluded link paste entries for {excluded_links_output_file} saved.')

            results.append({
                'File': file_name,
                'Allowed Entries': len(allowed_pastes),
                'Excluded Entries': len(excluded_pastes),
                'Excluded Links': len(excluded_links_df)
            })

    results_df = pd.DataFrame(results)
    results_summary_file = os.path.join(excluded_folder, 'paste_events_summary.csv')
    results_df.to_csv(results_summary_file, index=False)
    print(f'Summary of paste entries saved to {results_summary_file}')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_colwidth', None)
    print(results_df)


def main():
    """
    Main function to define directories, process, and clean survey data with
    paste entries, then save the results.
    """
    json_file = 'data/paste_451.json'
    input_folder = 'data/survey-results_with-RF_final'
    output_folder = 'data/paste-events_with-RF'
    output_folder_cleaned = 'data/paste-events_with-RF_cleaned'
    excluded_folder = 'data/paste-events_with-RF_excluded'
    output_folder_grouped = 'data/paste-events_with-RF_grouped'

    #process_survey_data_with_paste(json_file, input_folder, output_folder)
    #clean_survey_data_with_paste(output_folder, output_folder_cleaned, excluded_folder)
    group_survey_data_with_copypaste(output_folder_cleaned, output_folder_grouped)


if __name__ == '__main__':
    main()

#%%
