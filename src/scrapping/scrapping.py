#!/usr/bin/env python3
from __future__ import annotations

import logging
from collections.abc import Hashable
from pathlib import Path
from random import uniform
from time import sleep
from urllib.parse import urljoin

import pandas as pd
import requests
import undetected_chromedriver as uc  # type: ignore
import yaml
from bs4 import BeautifulSoup

from src.config import ROOT_DIR

GENERIC_TITLES = {
    "404",
    "page not found",
    "home",
    "homepage",
    "index",
    "untitled",
}
SKIP_KEYWORDS = {
    "privacy",
    "terms",
    "login",
    "subscribe",
    "contact",
    "about",
    "settings",
    "account",
    "register",
    "cookies",
    "help",
    "sitemap",
    "advertise",
    "donate",
    "events",
    "newsletter",
    "careers",
    "faq",
    "feedback",
    "preferences",
    "signup",
    "auth",
    "admin",
    "navigation",
    "guidelines",
    "accessibility",
    "schedules",
    "live",
    "bracket-odds",
    ".com/users/",
    "/video/watch",
    "/give/",
    "/assessment/",
}
NON_ARTICLE_DOMAINS = {
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "youtube.com",
    "linkedin.com",
    "tiktok.com",
}
ARTICLE_KEYWORDS = {
    # 'article', 'story', 'feature', 'report', 'review', 'explainer', 'investigation', 'interview', 'news'
    "article"
}
NON_ARTICLE_EXTS = {
    "jpg",
    "jpeg",
    "png",
    "gif",
    "svg",
    "pdf",
    "mp4",
    "mp3",
    "webp",
    "ico",
    "css",
    "js",
    "#comments",
}


SOURCES_FILE = ROOT_DIR / "src" / "scrapping" / "sources.yaml"
CHECKPOINT_CSV = ROOT_DIR / ".data" / "scraped_articles.csv"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_sources(path: Path):
    """
    Load scraping target entries from a YAML file.
    """
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.safe_load(f)
    return pd.DataFrame(entries)


def is_valid_article(link_tag, root_url: str, penalty: int = 500):
    """
    Score a BeautifulSoup link tag to decide if it points to an article.
    """
    title = link_tag.get_text(strip=True)
    href = link_tag.get("href", "")
    full_url = urljoin(root_url, href)

    score = len(full_url)
    if not href.startswith(("http://", "https://", "/")):
        score -= penalty
    if not title or title.lower() in GENERIC_TITLES:
        score -= penalty
    if any(full_url.endswith(ext) for ext in NON_ARTICLE_EXTS):
        score -= penalty
    if any(skip in full_url for skip in SKIP_KEYWORDS):
        score -= penalty
    if any(domain in full_url for domain in NON_ARTICLE_DOMAINS):
        score -= penalty
    if any(kw in full_url for kw in ARTICLE_KEYWORDS):
        score += penalty

    return score


def get_article_urls(root_url: str, max_articles: int):
    """
    Fetch and score links on a page, returning top N article URLs.
    """
    resp = requests.get(root_url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    tags = {tag["href"]: tag for tag in soup.find_all("a", href=True)}.values()

    scored = [(tag, is_valid_article(tag, root_url)) for tag in tags]
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for tag, _ in scored[:max_articles]:
        href = tag["href"]
        url = urljoin(root_url, href)
        results.append({"title": tag.get_text(strip=True), "link": url})
    return results


def fetch_article_content(url: str):
    """
    Render a page via undetected_chromedriver and return its HTML source.
    """
    try:
        options = uc.ChromeOptions()
        options.binary_location = "/usr/bin/google-chrome"
        driver = uc.Chrome(options=options, headless=True)
        driver.get(url)
        html = driver.page_source
        driver.quit()
        return html
    except Exception as e:
        logger.warning(f"fetch_article_content failed: {e}")
        return ""


def save_df(data: list[dict[Hashable, str]], path: Path):
    """
    Save list of dicts to CSV and return DataFrame.
    """
    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Saved %d records to %s", len(df), path)
    return df


def get_article_df(
    sources: pd.DataFrame,
    max_articles: int = 15,
    checkpoint: pd.DataFrame | None = None,
    backoff: tuple[float, float] = (1.0, 5.0),
):
    """
    Iterate sources, scrape up to max_articles per site, respecting checkpoints.
    """
    data = checkpoint.to_dict(orient="records") if checkpoint is not None else []
    for _count_source, row in sources.iterrows():
        website_name, url, domain, slop = (
            row[i] for i in ["website_name", "url", "domain", "slop"]
        )
        print(data)
        if website_name in [d.get("website", "") for d in data]:
            logger.info(f"Skipping {website_name}")
            continue
        logger.info(f"Fetching articles from {website_name}...")
        try:
            articles = get_article_urls(f"https://{url}", max_articles)
        except Exception as e:
            logger.info(f"Error fetching articles from {website_name}: {e}")
            continue
        logger.info(f"Retrieved articles: {articles}")

        for _count_article, article in enumerate(articles):
            content = fetch_article_content(article["link"])
            logger.info(f"Scrapped article {article['link']}")
            data.append(
                {
                    "website": website_name,
                    "title": article["title"],
                    "url": article["link"],
                    "content": content,
                    "domain": domain,
                    "slop": slop,
                }
            )
            sleep(uniform(*backoff))
        save_df(data, CHECKPOINT_CSV)
    save_df(data, CHECKPOINT_CSV)


if __name__ == "__main__":
    try:
        cp = pd.read_csv(CHECKPOINT_CSV)
    except FileNotFoundError:
        cp = pd.DataFrame()

    src_df = load_sources(SOURCES_FILE)
    final_df = get_article_df(src_df, max_articles=1, checkpoint=cp)
    logger.info("Scraping complete: %d total articles", len(final_df))
