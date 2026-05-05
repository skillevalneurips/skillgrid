---
name: url
description: Use this skill whenever the agent needs to process, resolve, validate, or manipulate URLs. This includes resolving relative URLs against a base URL, joining URL paths, parsing URL components (scheme, host, path, query, fragment), normalizing URLs, comparing URLs, building query strings, and validating URL formats. If the task involves constructing or interpreting web addresses, use this skill.
license: Proprietary. LICENSE.txt has complete terms
---

# URL Processing Guide

## Overview

This guide covers URL manipulation using Python's built-in `urllib.parse` module. In the WebWalker benchmark, proper URL processing is essential for resolving relative links found on web pages to absolute URLs, staying within the target domain, and constructing valid navigation paths.

## Quick Start

```python
import urllib.parse

# Resolve a relative URL against a base
base_url = "https://2025.aclweb.org/"
relative_url = "calls/main_conference_papers/"
absolute_url = urllib.parse.urljoin(base_url, relative_url)
print(absolute_url)  # https://2025.aclweb.org/calls/main_conference_papers/
```

## URL Resolution

### Resolve Relative URLs
```python
import urllib.parse

def resolve_url(base_url, relative_url):
    """Resolve a relative URL against a base URL."""
    return urllib.parse.urljoin(base_url, relative_url)

# Examples
base = "https://2025.aclweb.org/calls/"

# Relative path
resolve_url(base, "industry_track")
# -> https://2025.aclweb.org/calls/industry_track

# Parent directory
resolve_url(base, "../venue")
# -> https://2025.aclweb.org/venue

# Absolute path
resolve_url(base, "/about")
# -> https://2025.aclweb.org/about

# Full URL (returned as-is)
resolve_url(base, "https://other.site.com/page")
# -> https://other.site.com/page

# Protocol-relative
resolve_url(base, "//cdn.example.com/script.js")
# -> https://cdn.example.com/script.js
```

### Batch Resolve URLs
```python
import urllib.parse

def resolve_urls(base_url, relative_urls):
    """Resolve a list of relative URLs against a base."""
    return [urllib.parse.urljoin(base_url, url) for url in relative_urls]
```

## Parsing URLs

### Parse URL Components
```python
import urllib.parse

url = "https://2025.aclweb.org/calls/papers/?type=main#deadlines"
parsed = urllib.parse.urlparse(url)

print(parsed.scheme)    # https
print(parsed.netloc)    # 2025.aclweb.org
print(parsed.path)      # /calls/papers/
print(parsed.query)     # type=main
print(parsed.fragment)  # deadlines
print(parsed.hostname)  # 2025.aclweb.org
print(parsed.port)      # None
```

### Parse Query Parameters
```python
import urllib.parse

url = "https://example.com/search?q=python&page=2&sort=date"
parsed = urllib.parse.urlparse(url)
params = urllib.parse.parse_qs(parsed.query)
# {'q': ['python'], 'page': ['2'], 'sort': ['date']}

# Single values
params_flat = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
```

### Build Query Strings
```python
import urllib.parse

params = {"q": "web traversal", "page": 1, "lang": "en"}
query_string = urllib.parse.urlencode(params)
# q=web+traversal&page=1&lang=en

url = f"https://example.com/search?{query_string}"
```

## URL Validation and Comparison

### Check if URL is Valid
```python
import urllib.parse

def is_valid_url(url):
    """Check if a string is a valid URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        return all([parsed.scheme, parsed.netloc])
    except Exception:
        return False

is_valid_url("https://example.com")  # True
is_valid_url("not-a-url")            # False
is_valid_url("/relative/path")       # False
```

### Check Same Domain
```python
import urllib.parse

def is_same_domain(url, root_url):
    """Check if a URL belongs to the same domain as root."""
    url_domain = urllib.parse.urlparse(url).netloc
    root_domain = urllib.parse.urlparse(root_url).netloc
    return url_domain == root_domain

def is_within_root(url, root_url):
    """Check if a URL starts with the root URL (subdirectory check)."""
    return url.startswith(root_url)
```

### Normalize URL
```python
import urllib.parse

def normalize_url(url):
    """Normalize a URL for consistent comparison."""
    parsed = urllib.parse.urlparse(url)
    # Lowercase scheme and host
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    # Remove default ports
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]
    # Remove trailing slash from path (unless root)
    path = parsed.path.rstrip("/") or "/"
    # Remove fragment
    return urllib.parse.urlunparse((scheme, netloc, path, "", parsed.query, ""))
```

## URL Filtering

### Filter by Extension
```python
def filter_skip_extensions(urls, skip_exts=(".jpg", ".png", ".gif", ".jpeg", ".pdf", ".zip")):
    """Remove URLs pointing to files with unwanted extensions."""
    return [url for url in urls if not any(url.lower().endswith(ext) for ext in skip_exts)]
```

### Filter Internal Links Only
```python
import urllib.parse

def filter_internal(urls, root_url):
    """Keep only URLs within the root URL scope."""
    resolved = [urllib.parse.urljoin(root_url, url) for url in urls]
    return [url for url in resolved if url.startswith(root_url)]
```

### Extract Path Components
```python
import urllib.parse
from pathlib import PurePosixPath

url = "https://example.com/calls/industry_track/deadlines/"
path = urllib.parse.urlparse(url).path
parts = PurePosixPath(path).parts
# ('/', 'calls', 'industry_track', 'deadlines')
```

## Common Tasks

### Get Root URL from Any URL
```python
import urllib.parse

def get_root_url(url):
    """Extract the root URL (scheme + domain)."""
    parsed = urllib.parse.urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"

get_root_url("https://2025.aclweb.org/calls/papers/")
# -> https://2025.aclweb.org
```

### Build URL from Components
```python
import urllib.parse

url = urllib.parse.urlunparse((
    "https",           # scheme
    "example.com",     # netloc
    "/api/search",     # path
    "",                # params
    "q=test&page=1",   # query
    ""                 # fragment
))
# https://example.com/api/search?q=test&page=1
```

### URL Encode/Decode
```python
import urllib.parse

# Encode special characters
encoded = urllib.parse.quote("hello world/path", safe="/")
# hello%20world/path

# Decode
decoded = urllib.parse.unquote("hello%20world%2Fpath")
# hello world/path

# Encode query value
encoded_val = urllib.parse.quote_plus("search query with spaces")
# search+query+with+spaces
```

### Compute URL Depth
```python
import urllib.parse

def url_depth(url):
    """Compute the depth of a URL path (number of segments)."""
    path = urllib.parse.urlparse(url).path.strip("/")
    if not path:
        return 0
    return len(path.split("/"))

url_depth("https://example.com/")           # 0
url_depth("https://example.com/calls/")     # 1
url_depth("https://example.com/calls/cfp/") # 2
```

## Quick Reference

| Task | Method | Example |
|------|--------|---------|
| Resolve relative URL | `urljoin()` | `urllib.parse.urljoin(base, rel)` |
| Parse URL | `urlparse()` | `urllib.parse.urlparse(url)` |
| Get domain | `urlparse().netloc` | `parsed.netloc` |
| Parse query params | `parse_qs()` | `urllib.parse.parse_qs(query)` |
| Build query string | `urlencode()` | `urllib.parse.urlencode(params)` |
| Build URL | `urlunparse()` | `urllib.parse.urlunparse(parts)` |
| Encode URL | `quote()` | `urllib.parse.quote(text)` |
| Decode URL | `unquote()` | `urllib.parse.unquote(text)` |
| Validate URL | `urlparse()` check | Check scheme and netloc present |
