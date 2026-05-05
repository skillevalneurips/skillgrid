---
name: web_crawling
description: Use this skill whenever the agent needs to fetch, crawl, or retrieve web page content. This includes loading a URL and getting its HTML, converting web pages to markdown for processing, handling asynchronous crawling, managing browser rendering for JavaScript-heavy pages, taking screenshots during crawling, and dealing with crawl errors or timeouts. If the task involves visiting a URL or fetching page content, use this skill. Always validate and sanitize candidate URLs before fetching, resolve relative links correctly, and use SSL-safe request settings with retries.
license: Proprietary. LICENSE.txt has complete terms
---

# Web Crawling Guide

## Overview

This guide covers fetching and crawling web pages using `crawl4ai` (primary async crawler) and `requests`/`aiohttp` as alternatives. The WebWalker benchmark requires visiting web pages, extracting their content as markdown, and optionally capturing screenshots for visual analysis.

## Critical Fixes For Unstable Runs

Use these rules to avoid common WebWalker rollout failures:

- Only fetch URLs with `http` or `https` scheme.
- Never pass raw Google internal links such as `/httpservice/retry/enablejs?...` to `requests.get`.
- Resolve relative links with `urllib.parse.urljoin(base_url, href)` before fetching.
- Skip placeholder hosts like `example.com` unless the task explicitly targets them.
- Use retries + explicit CA bundle (`certifi`) for SSL robustness.
- On SSL or transient network failure, retry and then continue with alternative links instead of crashing.

## Quick Start

```python
import certifi
import requests
from urllib.parse import urljoin, urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def sanitize_url(base_url, candidate):
    """Resolve and validate candidate URL before crawling."""
    if not candidate:
        return None
    resolved = urljoin(base_url, candidate)
    parsed = urlparse(resolved)
    if parsed.scheme not in {"http", "https"}:
        return None
    if "/httpservice/retry/enablejs" in parsed.path:
        return None
    if parsed.netloc == "example.com":
        return None
    return resolved

def build_session():
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session

session = build_session()
url = sanitize_url("https://duckduckgo.com/", "https://duckduckgo.com/?q=SUPPLY+project+EHA")
if url:
    r = session.get(url, timeout=20, verify=certifi.where())
    r.raise_for_status()
    print(r.text[:500])
```

## Dependencies

```bash
pip install crawl4ai requests aiohttp beautifulsoup4 certifi
```

## Crawling Methods

### Crawl4AI (Primary) - Async Browser-Based Crawling

Crawl4AI renders JavaScript, handles dynamic content, and converts pages to clean markdown automatically.

#### Basic Page Fetch
```python
from crawl4ai import AsyncWebCrawler
import asyncio

async def crawl_page(url):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url)
        return {
            "html": result.html,
            "markdown": result.markdown,
            "success": result.success,
            "status_code": result.status_code
        }

page = asyncio.run(crawl_page("https://example.com"))
print(page["markdown"][:1000])
```

#### Fetch with Screenshot
```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
import base64
import asyncio

async def crawl_with_screenshot(url):
    config = CrawlerRunConfig(
        screenshot=True,
        screenshot_wait_for=1.0,  # Wait 1s for page to render before capture
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url, config=config)
        return result.html, result.markdown, result.screenshot

html, markdown, screenshot_b64 = asyncio.run(crawl_with_screenshot("https://example.com"))

# Save screenshot to file
if screenshot_b64:
    with open("page_screenshot.png", "wb") as f:
        f.write(base64.b64decode(screenshot_b64))
```

#### Crawl Multiple Pages
```python
from crawl4ai import AsyncWebCrawler
import asyncio

async def crawl_multiple(urls):
    results = {}
    async with AsyncWebCrawler() as crawler:
        for url in urls:
            try:
                result = await crawler.arun(url)
                results[url] = {
                    "markdown": result.markdown,
                    "success": result.success
                }
            except Exception as e:
                results[url] = {"error": str(e), "success": False}
    return results

urls = ["https://example.com/page1", "https://example.com/page2"]
pages = asyncio.run(crawl_multiple(urls))
```

### Requests (Fallback) - Simple HTTP Fetching

Use when crawl4ai is unavailable or for simple static pages.

```python
import certifi
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def fetch_page_simple(url, timeout=30):
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    response = session.get(url, headers=headers, timeout=timeout, verify=certifi.where())
    response.raise_for_status()
    return response.text

html = fetch_page_simple("https://duckduckgo.com/?q=SUPPLY+project+EHA")
soup = BeautifulSoup(html, "html.parser")
text = soup.get_text(separator="\n", strip=True)
```

### Aiohttp - Async HTTP Without Browser

```python
import aiohttp
import asyncio

async def fetch_async(url, timeout=30):
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            return await response.text()

html = asyncio.run(fetch_async("https://example.com"))
```

## Cleaning Crawled Content

### Clean Markdown Output
```python
import re

def clean_markdown(content):
    """Remove URLs and clean markdown content from crawl4ai output."""
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

### Extract Text Only
```python
from bs4 import BeautifulSoup

def html_to_text(html):
    """Extract plain text from HTML, removing all tags."""
    soup = BeautifulSoup(html, "html.parser")
    # Remove script and style elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)
```

## Common Tasks

### Crawl a Page and Get Both Content and Links
```python
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import asyncio
from urllib.parse import urljoin, urlparse

async def crawl_full(url):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url)
        soup = BeautifulSoup(result.html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            full_url = urljoin(url, a["href"])
            parsed = urlparse(full_url)
            if not text:
                continue
            if parsed.scheme not in {"http", "https"}:
                continue
            if "/httpservice/retry/enablejs" in parsed.path:
                continue
            if parsed.netloc == "example.com":
                continue
            links.append({"text": text, "url": full_url})
        return {
            "markdown": result.markdown,
            "links": links,
            "title": soup.title.string if soup.title else ""
        }

page = asyncio.run(crawl_full("https://duckduckgo.com/?q=SUPPLY+project+EHA"))
```

### Handle Crawl Failures with Retry
```python
import asyncio
import time
from crawl4ai import AsyncWebCrawler

async def crawl_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url)
                if result.success:
                    return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (2 ** attempt))
    return None
```

### Rate-Limited Crawling
```python
import asyncio
from crawl4ai import AsyncWebCrawler

async def crawl_with_delay(urls, delay=1.0):
    """Crawl multiple pages with a delay between requests."""
    results = {}
    async with AsyncWebCrawler() as crawler:
        for url in urls:
            try:
                result = await crawler.arun(url)
                results[url] = result.markdown
            except Exception as e:
                results[url] = f"Error: {e}"
            await asyncio.sleep(delay)
    return results
```

## Quick Reference

| Task | Method | Example |
|------|--------|---------|
| Basic crawl | crawl4ai | `crawler.arun(url)` |
| With screenshot | CrawlerRunConfig | `screenshot=True, screenshot_wait_for=1.0` |
| Get markdown | crawl4ai result | `result.markdown` |
| Get HTML | crawl4ai result | `result.html` |
| Simple fetch | requests | `requests.get(url)` |
| Async fetch | aiohttp | `session.get(url)` |
| Clean markdown | regex | `re.sub(pattern, '', content)` |
| HTML to text | BeautifulSoup | `soup.get_text()` |

## Error Handling

```python
from crawl4ai import AsyncWebCrawler
import asyncio

async def safe_crawl(url):
    """Crawl with comprehensive error handling."""
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url)
            if not result.success:
                return {"error": f"Crawl failed with status {result.status_code}", "content": None}
            return {"error": None, "content": result.markdown}
    except asyncio.TimeoutError:
        return {"error": "Timeout while crawling", "content": None}
    except Exception as e:
        return {"error": str(e), "content": None}
```
