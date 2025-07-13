#!/usr/bin/env python3
# date_filtered_rss_aggregator.py  – 快速补丁版
import feedparser
import hashlib
import datetime
from xml.etree import ElementTree as ET

# ---------- 1. 配置 ----------
SOURCES = [
    "https://www.ifanr.com/feed",
    "https://sspai.com/feed",
    "https://www.geekpark.net/rss"
]
OUTPUT_XML  = "final_rss.xml"
OUTPUT_HTML = "final_daily_news.html"
TIMEZONE    = datetime.timezone(datetime.timedelta(hours=8))   # 东八区
TODAY       = datetime.datetime.now(TIMEZONE).date()
TOMORROW    = TODAY + datetime.timedelta(days=1)

# ---------- 2. 工具函数 ----------
def plain_summary(text: str, limit: int = 200) -> str:
    """去掉 HTML 标签，只保留前 limit 个字符的纯文本"""
    import re
    txt = re.sub(r"<[^>]+>", "", text or "")
    return txt[:limit] + ("…" if len(txt) > limit else "")

def build_guid(link: str, pub: str) -> str:
    """稳定 GUID：仅由 link + 原始发布时间决定"""
    payload = f"{link}-{pub}"
    return hashlib.md5(payload.encode()).hexdigest()

def is_today(entry) -> bool:
    """判断文章发布日期是否为今天（东八区）"""
    dt = datetime.datetime(*entry.published_parsed[:6], tzinfo=datetime.timezone.utc)
    return TODAY <= dt.astimezone(TIMEZONE).date() < TOMORROW

# ---------- 3. 抓取并过滤 ----------
all_entries = []
for url in SOURCES:
    feed = feedparser.parse(url)
    for e in feed.entries:
        if not is_today(e):
            continue
        all_entries.append({
            "title": e.title,
            "link": e.link,
            "published": e.published,
            "summary": plain_summary(e.get("summary", "")),
            "guid": build_guid(e.link, e.published)
        })

# ---------- 4. 生成极简 RSS ----------
rss = ET.Element("rss", version="2.0")
chan = ET.SubElement(rss, "channel")
ET.SubElement(chan, "title").text = "每日新闻聚合"
ET.SubElement(chan, "link").text = "https://hybridrbt.github.io/rss-merger"
ET.SubElement(chan, "description").text = "多源中文科技新闻，每日去重精选"
ET.SubElement(chan, "language").text = "zh-cn"
ET.SubElement(chan, "lastBuildDate").text = datetime.datetime.now(TIMEZONE).strftime("%a, %d %b %Y %H:%M:%S %z")
ET.SubElement(chan, "ttl").text = "60"

for art in all_entries:
    item = ET.SubElement(chan, "item")
    ET.SubElement(item, "title").text = art["title"]
    ET.SubElement(item, "link").text  = art["link"]
    ET.SubElement(item, "description").text = f"<![CDATA[{art['summary']}]]>"
    ET.SubElement(item, "pubDate").text = art["published"]
    ET.SubElement(item, "guid", isPermaLink="false").text = art["guid"]

tree = ET.ElementTree(rss)
tree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)

# ---------- 5.（可选）生成 HTML 完整页 ----------
# 若仍需 final_daily_news.html，可沿用旧逻辑生成，但与 RSS 解耦