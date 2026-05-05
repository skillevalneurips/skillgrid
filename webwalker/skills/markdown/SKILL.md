---
name: markdown
description: Use this skill whenever the agent needs to process markdown content converted from web pages. This includes parsing markdown output from crawl4ai, extracting headers and sections, finding links within markdown, extracting tables from markdown, cleaning noisy markdown content, splitting markdown into logical sections, and converting markdown to plain text. If the task involves working with markdown-formatted web page content, use this skill.
license: Proprietary. LICENSE.txt has complete terms
---

# Markdown Processing Guide

## Overview

This guide covers processing markdown content that has been converted from web pages, primarily via crawl4ai. In the WebWalker benchmark, web pages are crawled and their content is returned as markdown. Understanding how to parse, clean, and extract information from this markdown is essential for answering queries.

## Quick Start

```python
import re

# Clean markdown from crawl4ai
def clean_markdown(content):
    # Remove markdown links
    result = re.sub(r'\[.*?\]\(.*?\)', '', content)
    # Remove standalone URLs
    result = re.sub(r'http[s]?://\S+', '', result)
    # Collapse whitespace
    result = re.sub(r'\n\n+', '\n', result)
    return result.strip()
```

## Cleaning Crawled Markdown

### Remove Links and URLs
```python
import re

def clean_markdown(content):
    """Clean markdown content from crawl4ai output."""
    # Remove markdown links [text](url)
    result = re.sub(r'\[.*?\]\(.*?\)', '', content)
    # Remove standalone URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    result = re.sub(url_pattern, '', result)
    # Remove empty list items
    result = result.replace("* \n", "")
    # Collapse multiple newlines
    result = re.sub(r"\n\n+", "\n", result)
    return result
```

### Preserve Link Text Only
```python
import re

def strip_links_keep_text(markdown):
    """Remove link URLs but keep the link text."""
    # [text](url) -> text
    result = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', markdown)
    # Remove standalone URLs
    result = re.sub(r'http[s]?://\S+', '', result)
    return result
```

### Remove Images
```python
import re

def remove_images(markdown):
    """Remove image markdown tags."""
    # ![alt](url)
    result = re.sub(r'!\[.*?\]\(.*?\)', '', markdown)
    return result
```

### Full Cleanup Pipeline
```python
import re

def full_cleanup(markdown):
    """Complete cleanup pipeline for crawled markdown."""
    result = markdown
    # Remove images
    result = re.sub(r'!\[.*?\]\(.*?\)', '', result)
    # Remove links but keep text
    result = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', result)
    # Remove standalone URLs
    result = re.sub(r'http[s]?://\S+', '', result)
    # Remove HTML tags that might remain
    result = re.sub(r'<[^>]+>', '', result)
    # Remove empty list items
    result = result.replace("* \n", "")
    # Remove excessive whitespace
    result = re.sub(r'[ \t]+', ' ', result)
    # Collapse blank lines
    result = re.sub(r'\n\s*\n', '\n\n', result)
    return result.strip()
```

## Extracting Structure

### Extract Headers
```python
import re

def extract_headers(markdown):
    """Extract all markdown headers with their levels."""
    headers = []
    for match in re.finditer(r'^(#{1,6})\s+(.+)$', markdown, re.MULTILINE):
        level = len(match.group(1))
        text = match.group(2).strip()
        headers.append({"level": level, "text": text, "position": match.start()})
    return headers
```

### Extract Sections by Header
```python
import re

def extract_section(markdown, section_name):
    """Extract the content under a specific header."""
    pattern = rf'(?:^|\n)(#{1,6})\s+{re.escape(section_name)}\s*\n(.*?)(?=\n#{{{1},{6}}}\s|\Z)'
    match = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return None

def extract_all_sections(markdown):
    """Split markdown into sections by headers."""
    sections = {}
    parts = re.split(r'(?:^|\n)(#{1,6})\s+(.+)\n', markdown)
    # parts: [pre-header text, #level, header1, content1, #level, header2, content2, ...]
    for i in range(1, len(parts) - 2, 3):
        level = len(parts[i])
        header = parts[i + 1].strip()
        content = parts[i + 2].strip()
        sections[header] = {"level": level, "content": content}
    return sections
```

### Build Table of Contents
```python
def build_toc(markdown):
    """Build a table of contents from markdown headers."""
    headers = extract_headers(markdown)
    toc_lines = []
    for h in headers:
        indent = "  " * (h["level"] - 1)
        toc_lines.append(f"{indent}- {h['text']}")
    return "\n".join(toc_lines)
```

## Extracting Content

### Extract Tables from Markdown
```python
import re

def extract_markdown_tables(markdown):
    """Extract tables from markdown content."""
    tables = []
    # Match markdown table blocks
    table_pattern = r'(\|[^\n]+\|\n\|[-: |]+\|\n(?:\|[^\n]+\|\n)*)'
    for match in re.finditer(table_pattern, markdown):
        table_text = match.group(1)
        rows = []
        for line in table_text.strip().split("\n"):
            if re.match(r'\|[-: |]+\|', line):  # Skip separator row
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            rows.append(cells)
        if rows:
            tables.append(rows)
    return tables
```

### Extract Lists
```python
import re

def extract_lists(markdown):
    """Extract list items from markdown."""
    items = []
    for match in re.finditer(r'^[\s]*[-*+]\s+(.+)$', markdown, re.MULTILINE):
        items.append(match.group(1).strip())
    return items

def extract_numbered_lists(markdown):
    """Extract numbered list items."""
    items = []
    for match in re.finditer(r'^[\s]*\d+[.)]\s+(.+)$', markdown, re.MULTILINE):
        items.append(match.group(1).strip())
    return items
```

### Extract Bold/Emphasized Text
```python
import re

def extract_bold(markdown):
    """Extract bold text (**text** or __text__)."""
    return re.findall(r'\*\*(.+?)\*\*|__(.+?)__', markdown)

def extract_italic(markdown):
    """Extract italic text (*text* or _text_)."""
    return re.findall(r'(?<!\*)\*([^*]+)\*(?!\*)|(?<!_)_([^_]+)_(?!_)', markdown)
```

### Extract Links from Markdown
```python
import re

def extract_markdown_links(markdown):
    """Extract all links from markdown content."""
    links = []
    for match in re.finditer(r'\[([^\]]+)\]\(([^\)]+)\)', markdown):
        links.append({"text": match.group(1), "url": match.group(2)})
    return links
```

## Text Extraction

### Convert Markdown to Plain Text
```python
import re

def markdown_to_text(markdown):
    """Convert markdown to plain text."""
    text = markdown
    # Remove headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.+?)_{1,3}', r'\1', text)
    # Remove links, keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove images
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove blockquotes
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    # Clean whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()
```

## Common Tasks

### Find Relevant Content by Query
```python
def find_relevant_content(markdown, query_keywords, context_lines=2):
    """Find sections of markdown relevant to query keywords."""
    lines = markdown.split("\n")
    relevant = []
    keywords_lower = [kw.lower() for kw in query_keywords]

    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in keywords_lower):
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            context = "\n".join(lines[start:end])
            relevant.append(context)

    return relevant
```

### Summarize Page Content
```python
def summarize_page(markdown, max_length=2000):
    """Create a concise summary of page content."""
    # Get headers for structure
    headers = extract_headers(markdown)
    header_text = " | ".join(h["text"] for h in headers)

    # Get first N chars of cleaned content
    clean = full_cleanup(markdown)
    content_preview = clean[:max_length]

    return f"Page sections: {header_text}\n\nContent preview:\n{content_preview}"
```

### Chunk Long Markdown
```python
def chunk_markdown(markdown, max_chunk_size=5000):
    """Split long markdown into manageable chunks by sections."""
    sections = re.split(r'\n(?=#{1,3}\s)', markdown)
    chunks = []
    current_chunk = ""

    for section in sections:
        if len(current_chunk) + len(section) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = section
        else:
            current_chunk += "\n" + section

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks
```

## Quick Reference

| Task | Method | Example |
|------|--------|---------|
| Clean markdown | Regex removal | `re.sub(r'\[.*?\]\(.*?\)', '', md)` |
| Extract headers | Regex match | `re.finditer(r'^#+\s+(.+)', md)` |
| Extract section | Regex search | `re.search(header_pattern, md)` |
| Extract tables | Regex match | Match `\|...\|` patterns |
| Extract lists | Regex match | `re.finditer(r'^[-*+]\s+(.+)', md)` |
| Extract links | Regex match | `re.finditer(r'\[text\]\(url\)', md)` |
| To plain text | Regex strip | Remove all markdown formatting |
| Find by keyword | Line search | Check each line for keywords |
