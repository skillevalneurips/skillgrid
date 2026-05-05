---
name: html_parsing
description: Use this skill whenever the agent needs to parse HTML content from web pages. This includes extracting links and buttons from HTML, finding clickable elements (anchor tags, onclick handlers, data-url attributes), extracting text content from specific HTML elements, parsing tables from HTML, handling navigation menus, and filtering links by domain or pattern. If the task involves analyzing HTML structure or extracting elements from a web page, use this skill.
license: Proprietary. LICENSE.txt has complete terms
---

# HTML Parsing Guide

## Overview

This guide covers parsing HTML content from web pages using BeautifulSoup. In the WebWalker benchmark, HTML parsing is critical for extracting clickable links/buttons for navigation and identifying relevant content areas on a page.

## Quick Start

```python
from bs4 import BeautifulSoup

html = "<html><body><a href='/page1'>Link 1</a><a href='/page2'>Link 2</a></body></html>"
soup = BeautifulSoup(html, "html.parser")

for link in soup.find_all("a", href=True):
    print(f"{link.get_text(strip=True)}: {link['href']}")
```

## Dependencies

```bash
pip install beautifulsoup4 lxml
```

## Parsing HTML

### Initialize Parser
```python
from bs4 import BeautifulSoup

# From string
soup = BeautifulSoup(html_string, "html.parser")

# With lxml (faster, more lenient)
soup = BeautifulSoup(html_string, "lxml")

# From file
with open("page.html", "r") as f:
    soup = BeautifulSoup(f, "html.parser")
```

### Extract Page Title
```python
title = soup.title.string if soup.title else "No title"
```

### Extract All Text
```python
# Simple text extraction
text = soup.get_text(separator="\n", strip=True)

# Remove script/style elements first
for tag in soup(["script", "style"]):
    tag.decompose()
text = soup.get_text(separator="\n", strip=True)
```

## Extracting Links and Buttons

### Extract Standard Links
```python
from bs4 import BeautifulSoup

def extract_links(html):
    """Extract all anchor tag links with text."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a_tag in soup.find_all("a", href=True):
        url = a_tag["href"]
        text = "".join(a_tag.stripped_strings)
        if text and "javascript" not in url:
            links.append({"url": url, "text": text})
    return links
```

### Extract Links with onclick Handlers
```python
import re
from bs4 import BeautifulSoup

def extract_onclick_links(html):
    """Extract links from onclick JavaScript handlers."""
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a_tag in soup.find_all("a", onclick=True):
        onclick_text = a_tag["onclick"]
        text = "".join(a_tag.stripped_strings)
        match = re.search(r"window\.location\.href='([^']*)'", onclick_text)
        if match:
            url = match.group(1)
            if url and text:
                links.append({"url": url, "text": text})
    return links
```

### Extract Links from data-url Attributes
```python
def extract_data_url_links(html):
    """Extract links from data-url attributes."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a_tag in soup.find_all("a", attrs={"data-url": True}):
        url = a_tag["data-url"]
        text = "".join(a_tag.stripped_strings)
        if url and text:
            links.append({"url": url, "text": text})
    return links
```

### Extract Button Links
```python
import re
from bs4 import BeautifulSoup

def extract_button_links(html):
    """Extract links from button elements with onclick handlers."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for button in soup.find_all("button", onclick=True):
        onclick_text = button["onclick"]
        text = (button.get("title")
                or button.get("aria-label")
                or "".join(button.stripped_strings))
        match = re.search(r"window\.location\.href='([^']*)'", onclick_text)
        if match:
            url = match.group(1)
            if url and text:
                links.append({"url": url, "text": text})
    return links
```

### Comprehensive Link Extraction (All Types)
```python
import re
import urllib.parse
from bs4 import BeautifulSoup

def extract_all_links(html, root_url):
    """Extract all clickable links from a page, resolving relative URLs."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    skip_extensions = (".jpg", ".png", ".gif", ".jpeg", ".pdf")

    # Standard <a href="..."> links
    for a_tag in soup.find_all("a", href=True):
        url = a_tag["href"]
        text = "".join(a_tag.stripped_strings)
        if text and "javascript" not in url and not url.endswith(skip_extensions):
            full_url = urllib.parse.urljoin(root_url, url)
            if full_url.startswith(root_url):
                links.append({"url": full_url, "text": text})

    # <a onclick="..."> links
    for a_tag in soup.find_all("a", onclick=True):
        onclick_text = a_tag["onclick"]
        text = "".join(a_tag.stripped_strings)
        match = re.search(r"window\.location\.href='([^']*)'", onclick_text)
        if match:
            url = match.group(1)
            if url and text and not url.endswith(skip_extensions):
                full_url = urllib.parse.urljoin(root_url, url)
                if full_url.startswith(root_url):
                    links.append({"url": full_url, "text": text})

    # <a data-url="..."> links
    for a_tag in soup.find_all("a", attrs={"data-url": True}):
        url = a_tag["data-url"]
        text = "".join(a_tag.stripped_strings)
        if url and text and not url.endswith(skip_extensions):
            full_url = urllib.parse.urljoin(root_url, url)
            if full_url.startswith(root_url):
                links.append({"url": full_url, "text": text})

    # <a class="herf-mask"> links (with title fallback)
    for a_tag in soup.find_all("a", class_="herf-mask"):
        url = a_tag.get("href")
        text = a_tag.get("title") or "".join(a_tag.stripped_strings)
        if url and text and not url.endswith(skip_extensions):
            full_url = urllib.parse.urljoin(root_url, url)
            if full_url.startswith(root_url):
                links.append({"url": full_url, "text": text})

    # <button onclick="..."> links
    for button in soup.find_all("button", onclick=True):
        onclick_text = button["onclick"]
        text = (button.get("title")
                or button.get("aria-label")
                or "".join(button.stripped_strings))
        match = re.search(r"window\.location\.href='([^']*)'", onclick_text)
        if match:
            url = match.group(1)
            if url and text:
                full_url = urllib.parse.urljoin(root_url, url)
                if full_url.startswith(root_url):
                    links.append({"url": full_url, "text": text})

    # Deduplicate
    unique = {f"{item['url']}_{item['text']}": item for item in links}
    return list(unique.values())
```

### Format Links as Clickable Buttons
```python
def format_as_buttons(links):
    """Format extracted links as button text for the agent."""
    info = ""
    for link in links:
        info += "<button>" + link["text"] + "<button>\n"
    return info
```

### Build Button-URL Mapping
```python
import json

def build_button_map(links, filepath="button_url_map.json"):
    """Create and save a mapping from button text to URL."""
    button_map = {}
    for link in links:
        button_map[link["text"]] = link["url"]
    with open(filepath, "w") as f:
        json.dump(button_map, f)
    return button_map
```

## Extracting Tables

### Extract HTML Tables
```python
from bs4 import BeautifulSoup

def extract_tables(html):
    """Extract all tables from HTML as lists of rows."""
    soup = BeautifulSoup(html, "html.parser")
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables
```

### Extract Tables to DataFrame
```python
import pandas as pd
from bs4 import BeautifulSoup

def tables_to_dataframes(html):
    """Extract tables and convert to pandas DataFrames."""
    dfs = pd.read_html(html)
    return dfs
```

## Extracting Specific Content

### Extract by CSS Selector
```python
soup = BeautifulSoup(html, "html.parser")

# By ID
element = soup.select_one("#main-content")

# By class
elements = soup.select(".article-body p")

# By tag and attribute
elements = soup.select("div[data-section='content']")
```

### Extract Navigation Menu
```python
def extract_nav(html):
    """Extract navigation menu items."""
    soup = BeautifulSoup(html, "html.parser")
    nav_items = []
    for nav in soup.find_all("nav"):
        for a in nav.find_all("a", href=True):
            text = a.get_text(strip=True)
            if text:
                nav_items.append({"text": text, "url": a["href"]})
    return nav_items
```

### Extract Meta Information
```python
def extract_meta(html):
    """Extract page metadata."""
    soup = BeautifulSoup(html, "html.parser")
    meta = {}
    meta["title"] = soup.title.string if soup.title else ""
    for tag in soup.find_all("meta"):
        name = tag.get("name") or tag.get("property", "")
        content = tag.get("content", "")
        if name and content:
            meta[name] = content
    return meta
```

## Common Tasks

### Filter Links by Domain
```python
import urllib.parse

def filter_same_domain(links, root_url):
    """Keep only links within the same domain."""
    root_domain = urllib.parse.urlparse(root_url).netloc
    return [
        link for link in links
        if urllib.parse.urlparse(link["url"]).netloc == root_domain
    ]
```

### Find Links Matching Keywords
```python
def find_relevant_links(links, keywords):
    """Find links whose text matches any of the given keywords."""
    keywords_lower = [k.lower() for k in keywords]
    return [
        link for link in links
        if any(kw in link["text"].lower() for kw in keywords_lower)
    ]
```

### Remove Duplicate Links
```python
def deduplicate_links(links):
    """Remove duplicate links by URL."""
    seen = set()
    unique = []
    for link in links:
        if link["url"] not in seen:
            seen.add(link["url"])
            unique.append(link)
    return unique
```

## Quick Reference

| Task | Method | Example |
|------|--------|---------|
| Parse HTML | BeautifulSoup | `BeautifulSoup(html, "html.parser")` |
| Find links | `find_all("a")` | `soup.find_all("a", href=True)` |
| Get text | `get_text()` | `tag.get_text(strip=True)` |
| CSS selector | `select()` | `soup.select(".class tag")` |
| Extract tables | `find_all("table")` | Iterate `tr` and `td` |
| Onclick links | regex on onclick | `re.search(r"href='([^']*)'", ...)` |
| Button text | stripped_strings | `"".join(tag.stripped_strings)` |
| Meta info | `find_all("meta")` | `tag.get("content")` |
