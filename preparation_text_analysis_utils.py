"""
preparation_text_analysis_utils.py

This module provides various utility functions for analyzing text,
including detecting links, URLs, emails, file paths, and passwords.
It also includes functions to detect AI tool URLs, search engine URLs,
and non-recognized language text.

Functions:
- is_link_only(text): Checks if the given text consists only of links.
- is_link_and_text(text): Checks if the text contains both links and regular text.
- is_search_engine_url(url): Checks if a URL belongs to a search engine.
- is_ai_tool_url(url): Checks if a URL is related to AI tools.
- is_email_with_padding(text, padding): Detects email addresses with padding.
- is_not_language(clipboard_content): Checks if the text is not a recognized language.
- is_file_path(string): Checks if the text is a file path.
- is_probably_password(clipboard_content): Checks if the content is likely a password.

Author: Joshua Tischlik
Date: 20.09.2024
"""

import os
import json
import pandas as pd
import re
import urllib.parse
from langdetect import detect, LangDetectException


def is_link_only(text):
    """
    Check if the given text consists only of links separated by spaces
    or line breaks, including incomplete links like 'ttps'.
    """
    url_pattern = re.compile(
        r'((h?t?t?ps?|ftp|www):\/\/[^\s/$.?#].[^\s]*)|(www\.[^\s/$.?#].[^\s]*)',
        re.IGNORECASE
    )
    text = text.strip()
    links = re.split(r'\s+', text)
    return all(re.search(url_pattern, link) for link in links)


def is_link_and_text(text):
    """
    Check if the given text contains both links and regular text,
    with a maximum of 20 words.
    """
    url_pattern = re.compile(
        r'((h?t?t?ps?|ftp|www):\/\/[^\s/$.?#].[^\s]*)|(www\.[^\s/$.?#].[^\s]*)',
        re.IGNORECASE
    )
    text = text.strip()
    words = re.split(r'\s+', text)
    if len(words) > 20:
        return False
    contains_link = any(re.search(url_pattern, word) for word in words)
    return contains_link


def is_search_engine_url(url):
    """Check if a URL belongs to a search engine."""
    search_urls = [
        'https://www.google.com/search',
        'https://www.bing.com/search',
        'https://search.yahoo.com/search',
        'https://www.baidu.com/s',
        'https://yandex.ru/search',
        'https://duckduckgo.com/',
        'https://www.ecosia.org/search',
        'https://search.naver.com/search.naver',
        'https://www.seznam.cz/hledat',
        'https://www.qwant.com/?q=',
        'https://www.sogou.com/web',
        'https://www.startpage.com/sp/search',
        'https://swisscows.com/web',
        'https://www.ask.com/web',
        'https://www.mojeek.com/search',
        'https://www.dogpile.com/search/web'
    ]
    return any(url.startswith(search_url) for search_url in search_urls)


def is_ai_tool_url(url):
    """Check if a URL is related to AI tools."""
    decoded_url = urllib.parse.unquote(url)
    ai_tools_url_patterns = [
        r'https?://chat\.openai\.com.*',
        r'https?://platform\.openai\.com.*',
        r'https?://(www\.)?openai\.com/chatgpt.*',
        r'https?://.*openai.*',
        r'https?://bard\.google\.com.*',
        r'https?://ai\.google\.com/bard.*',
        r'https?://(www\.)?bing\.com/chat.*',
        r'https?://(www\.)?you\.com/chat.*',
        r'https?://(www\.)?writesonic\.com.*',
        r'https?://(www\.)?jasper\.ai.*',
        r'https?://(www\.)?forefront\.ai.*',
        r'https?://(www\.)?character\.ai.*',
        r'https?://(www\.)?claude\.anthropic\.com.*',
        r'https?://(www\.)?poe\.com.*',
        r'https?://(www\.)?inflection\.ai.*',
        r'https?://(www\.)?pi\.ai.*',
        r'https?://(www\.)?playground\.openai\.com.*',
        r'https?://(www\.)?gpt-4\.com.*',
        r'https?://chatgpt\.com.*',
        r'https?://(www\.)?huggingface\.co.*',
        r'https?://(www\.)?perplexity\.ai.*',
        r'https?://(www\.)?replika\.ai.*',
        r'https?://(www\.)?midjourney\.com.*',
        r'https?://(www\.)?stability\.ai.*',
        r'https?://(www\.)?deepmind\.com.*',
        r'https?://(www\.)?github\.com/copilot.*',
        r'https?://(www\.)?notion\.so/product/ai.*',
        r'https?://(www\.)?runwayml\.com.*',
        r'https?://(www\.)?synthesia\.io.*',
        r'https?://labs\.openai\.com.*',
        r'https?://(console\.)?cloud\.google\.com/ai.*',
        r'https?://(portal\.)?azure\.com.*',
        r'https?://(www\.)?ibm\.com/watson.*'
    ]
    for pattern in ai_tools_url_patterns:
        if re.match(pattern, decoded_url):
            return True
    return False


def is_email_with_padding(text, padding=10):
    """Detects email addresses surrounded by up to 'padding' characters."""
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    match = re.search(email_pattern, text)
    if match:
        start, end = match.span()
        before_email = text[:start]
        after_email = text[end:]
        if len(before_email) <= padding and len(after_email) <= padding:
            return True
    return False


def is_not_language(clipboard_content):
    """Checks if the text is not a recognized language."""
    try:
        detect(clipboard_content)
        return False
    except LangDetectException:
        return True


def is_file_path(string):
    """Checks if the text is a file path."""
    return string.strip().lower().startswith("file:")


def is_probably_password(clipboard_content):
    """Checks if the clipboard content is likely a password."""
    if 6 < len(clipboard_content) < 20 or is_link_only(clipboard_content):
        return False

    has_uppercase = re.search(r'[A-Z]', clipboard_content)
    has_lowercase = re.search(r'[a-z]', clipboard_content)
    has_digit = re.search(r'[0-9]', clipboard_content)
    has_special_char = re.search(r'[!@#$%^&*(),.?":{}|<>]', clipboard_content)
    has_spaces = re.search(r'\s', clipboard_content)

    if has_spaces:
        return False

    return all([has_uppercase, has_lowercase, has_digit, has_special_char])
