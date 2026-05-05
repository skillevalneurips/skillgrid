---
name: screenshot
description: Use this skill whenever the agent needs to capture, process, or analyze screenshots of web pages. This includes taking screenshots during crawling, saving screenshots as image files, decoding base64 screenshot data, comparing visual states of pages, creating visual navigation logs, and using screenshots for visual verification of page content. If the task involves visual capture or analysis of web pages, use this skill.
license: Proprietary. LICENSE.txt has complete terms
---

# Screenshot Capture and Processing Guide

## Overview

This guide covers capturing and processing screenshots of web pages during traversal in the WebWalker benchmark. Screenshots provide visual context for understanding page layout, verifying content, and creating navigation logs. Crawl4ai captures screenshots as base64-encoded PNG data.

## Quick Start

```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
import base64
import asyncio

async def take_screenshot(url):
    config = CrawlerRunConfig(
        screenshot=True,
        screenshot_wait_for=1.0,
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url, config=config)
        if result.screenshot:
            with open("screenshot.png", "wb") as f:
                f.write(base64.b64decode(result.screenshot))
        return result.screenshot

screenshot = asyncio.run(take_screenshot("https://example.com"))
```

## Dependencies

```bash
pip install crawl4ai pillow
```

## Capturing Screenshots

### Basic Screenshot with Crawl4AI
```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
import base64
import asyncio

async def capture_screenshot(url, wait_time=1.0):
    """Capture a screenshot of a web page."""
    config = CrawlerRunConfig(
        screenshot=True,
        screenshot_wait_for=wait_time,
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url, config=config)
        return result.screenshot  # base64-encoded PNG

screenshot_b64 = asyncio.run(capture_screenshot("https://example.com"))
```

### Save Screenshot to File
```python
import base64
import os

def save_screenshot(screenshot_b64, filepath="screenshot.png"):
    """Save a base64-encoded screenshot to a file."""
    if screenshot_b64:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(screenshot_b64))
        return True
    return False
```

### Capture with Page Content
```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
import asyncio

async def crawl_with_visual(url):
    """Get both page content and screenshot."""
    config = CrawlerRunConfig(
        screenshot=True,
        screenshot_wait_for=1.0,
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url, config=config)
        return {
            "html": result.html,
            "markdown": result.markdown,
            "screenshot": result.screenshot,
            "success": result.success
        }

page = asyncio.run(crawl_with_visual("https://example.com"))
```

### Capture Without Screenshot (Faster)
```python
from crawl4ai import AsyncWebCrawler
import asyncio

async def crawl_fast(url):
    """Crawl without screenshot for faster execution."""
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url, screenshot=False)
        return result.html, result.markdown
```

## Processing Screenshots

### Load and Resize Screenshot
```python
from PIL import Image
import base64
from io import BytesIO

def load_screenshot(screenshot_b64):
    """Load a base64 screenshot as a PIL Image."""
    image_data = base64.b64decode(screenshot_b64)
    return Image.open(BytesIO(image_data))

def resize_screenshot(image, max_width=800):
    """Resize screenshot while maintaining aspect ratio."""
    if image.width > max_width:
        ratio = max_width / image.width
        new_height = int(image.height * ratio)
        return image.resize((max_width, new_height), Image.LANCZOS)
    return image
```

### Get Screenshot Dimensions
```python
from PIL import Image
import base64
from io import BytesIO

def get_screenshot_info(screenshot_b64):
    """Get basic information about a screenshot."""
    image_data = base64.b64decode(screenshot_b64)
    image = Image.open(BytesIO(image_data))
    return {
        "width": image.width,
        "height": image.height,
        "format": image.format,
        "size_bytes": len(image_data)
    }
```

### Crop Screenshot Region
```python
from PIL import Image

def crop_screenshot(image, left, top, right, bottom):
    """Crop a region from a screenshot."""
    return image.crop((left, top, right, bottom))

def crop_top_portion(image, fraction=0.3):
    """Crop the top portion of a screenshot (e.g., header area)."""
    height = int(image.height * fraction)
    return image.crop((0, 0, image.width, height))
```

## Navigation Logging

### Create Visual Navigation Log
```python
import os
import base64

class ScreenshotLogger:
    def __init__(self, output_dir="screenshots"):
        self.output_dir = output_dir
        self.step_count = 0
        os.makedirs(output_dir, exist_ok=True)

    def log_step(self, screenshot_b64, url, action=""):
        """Save a screenshot with step metadata."""
        if not screenshot_b64:
            return None

        filepath = os.path.join(self.output_dir, f"step_{self.step_count:03d}.png")
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(screenshot_b64))

        # Save metadata
        meta_path = os.path.join(self.output_dir, f"step_{self.step_count:03d}.txt")
        with open(meta_path, "w") as f:
            f.write(f"Step: {self.step_count}\n")
            f.write(f"URL: {url}\n")
            f.write(f"Action: {action}\n")

        self.step_count += 1
        return filepath

    def get_all_steps(self):
        """List all logged screenshots."""
        files = sorted(f for f in os.listdir(self.output_dir) if f.endswith(".png"))
        return [os.path.join(self.output_dir, f) for f in files]
```

### Create Screenshot Comparison
```python
from PIL import Image

def side_by_side(image1, image2, gap=10):
    """Create a side-by-side comparison of two screenshots."""
    max_height = max(image1.height, image2.height)
    total_width = image1.width + image2.width + gap

    result = Image.new("RGB", (total_width, max_height), "white")
    result.paste(image1, (0, 0))
    result.paste(image2, (image1.width + gap, 0))
    return result
```

## Streamlit Integration

### Display Screenshots in Streamlit UI
```python
import streamlit as st
from PIL import Image
import base64

def display_screenshot(screenshot_b64, caption="Page Screenshot", width=400):
    """Display a screenshot in Streamlit."""
    if screenshot_b64:
        # Save temporarily
        filepath = "/tmp/screenshot_temp.png"
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(screenshot_b64))
        image = Image.open(filepath)
        st.image(image, caption=caption, width=width)
```

### Display Navigation Steps
```python
import streamlit as st
from PIL import Image
import os

def display_navigation_log(screenshot_dir):
    """Display all navigation screenshots in order."""
    files = sorted(f for f in os.listdir(screenshot_dir) if f.endswith(".png"))
    for i, filename in enumerate(files):
        filepath = os.path.join(screenshot_dir, filename)
        image = Image.open(filepath)
        st.image(image, caption=f"Step {i}", width=400)
```

## Common Tasks

### Full Crawl-and-Screenshot Pipeline
```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
import base64
import os
import asyncio

async def crawl_and_screenshot(url, output_dir="images", step_index=0):
    """Crawl a page and save its screenshot."""
    config = CrawlerRunConfig(
        screenshot=True,
        screenshot_wait_for=1.0,
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url, config=config)

        # Save screenshot
        if result.screenshot:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{step_index}.png")
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(result.screenshot))

        return {
            "html": result.html,
            "markdown": result.markdown,
            "screenshot_saved": bool(result.screenshot),
        }
```

### Convert Screenshot to Base64 String
```python
import base64

def image_to_base64(filepath):
    """Convert an image file to base64 string."""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
```

### Batch Screenshot Capture
```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
import base64
import os
import asyncio

async def batch_screenshots(urls, output_dir="screenshots"):
    """Capture screenshots of multiple pages."""
    config = CrawlerRunConfig(screenshot=True, screenshot_wait_for=1.0)
    os.makedirs(output_dir, exist_ok=True)

    async with AsyncWebCrawler() as crawler:
        for i, url in enumerate(urls):
            try:
                result = await crawler.arun(url, config=config)
                if result.screenshot:
                    filepath = os.path.join(output_dir, f"page_{i:03d}.png")
                    with open(filepath, "wb") as f:
                        f.write(base64.b64decode(result.screenshot))
            except Exception as e:
                print(f"Failed to screenshot {url}: {e}")
```

## Quick Reference

| Task | Method | Example |
|------|--------|---------|
| Capture screenshot | CrawlerRunConfig | `screenshot=True, screenshot_wait_for=1.0` |
| Save to file | base64 decode | `base64.b64decode(screenshot)` |
| Load as PIL Image | PIL + BytesIO | `Image.open(BytesIO(data))` |
| Resize | PIL resize | `image.resize((w, h))` |
| Crop region | PIL crop | `image.crop((l, t, r, b))` |
| Display in Streamlit | st.image | `st.image(image, width=400)` |
| Log navigation | ScreenshotLogger | `logger.log_step(screenshot, url)` |
| Fast crawl (no screenshot) | skip screenshot | `screenshot=False` |
