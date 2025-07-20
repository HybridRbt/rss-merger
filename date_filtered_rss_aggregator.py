#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import feedparser
import requests
from datetime import datetime, timezone, timedelta
import xml.etree.ElementTree as ET
import hashlib
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import html

class DateFilteredRSSAggregator:
    def __init__(self, output_file='final_rss.xml', html_file='final_daily_news.html'):
        # RSS源列表（已移除少数派）
        self.rss_feeds = [
            'https://www.ifanr.com/feed',  # 爱范儿
            'https://www.geekpark.net/rss'  # 极客公园
        ]
        
        self.output_file = output_file
        self.html_file = html_file
        
        # 中国标准时间 (CST, UTC+8)
        self.cst_timezone = timezone(timedelta(hours=8))
        
        # 获取当前CST时间
        self.today_cst = datetime.now(self.cst_timezone)
        self.today_date_str = self.today_cst.strftime('%Y-%m-%d')
        
        print(f"当前CST时间: {self.today_cst}")
        print(f"筛选日期: {self.today_date_str}")

    def _is_today_cst(self, pub_date):
        """检查文章发布日期是否为今天（CST时间）"""
        try:
            if pub_date.tzinfo is None:
                # 如果没有时区信息，假设为CST
                pub_date = pub_date.replace(tzinfo=self.cst_timezone)
            else:
                # 转换为CST时间
                pub_date = pub_date.astimezone(self.cst_timezone)
            
            pub_date_str = pub_date.strftime('%Y-%m-%d')
            return pub_date_str == self.today_date_str
        except Exception as e:
            print(f"日期解析错误: {e}")
            return False

    def _extract_text_features(self, title, description):
        """提取文本特征用于聚类"""
        # 清理HTML标签
        clean_desc = re.sub(r'<[^>]+>', '', description) if description else ''
        
        # 合并标题和描述
        text = f"{title} {clean_desc}"
        
        # 提取关键词（简单的中文分词）
        keywords = re.findall(r'[\u4e00-\u9fa5]+', text)
        features = ' '.join(keywords)
        
        return {
            'text': features,
            'title': title,
            'description': clean_desc
        }

    def _cluster_and_filter_topics(self, articles):
        """聚类相似话题并去重"""
        if len(articles) <= 1:
            return articles, []

        # 提取文本特征
        features_list = []
        for article in articles:
            features = self._extract_text_features(article['title'], article.get('description', ''))
            features_list.append(features)
            article['features'] = features

        # 使用TF-IDF进行文本向量化
        texts = [f['text'] for f in features_list]
        if not any(texts):  # 如果所有文本都为空
            return articles, []

        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # 计算余弦相似度
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # 聚类相似文章（相似度阈值0.3）
            clusters = []
            used = set()
            
            for i in range(len(articles)):
                if i in used:
                    continue
                    
                cluster = [i]
                used.add(i)
                
                for j in range(i + 1, len(articles)):
                    if j not in used and similarity_matrix[i][j] > 0.3:
                        cluster.append(j)
                        used.add(j)
                
                clusters.append(cluster)

            # 处理每个聚类
            merged_topics = []
            unique_topics = []

            for cluster in clusters:
                cluster_articles = [articles[i] for i in cluster]
                
                if len(cluster_articles) == 1:
                    # 单独的文章，直接添加
                    unique_topics.append(cluster_articles[0])
                else:
                    # 多篇相似文章，进行合并
                    merged_topic = self._merge_topic_cluster(cluster_articles)
                    if merged_topic:
                        merged_topics.append(merged_topic)

            print(f"话题聚类与过滤后，保留 {len(merged_topics)} 个主要话题和 {len(unique_topics)} 个独特话题")
            return merged_topics, unique_topics

        except Exception as e:
            print(f"聚类过程出错: {e}")
            return articles, []

    def _merge_topic_cluster(self, cluster_articles):
        """合并同一话题下的多篇文章"""
        if not cluster_articles:
            return None

        # 选择最长的文章作为代表，并标记为有待处理的少数派文章
        representative = max(cluster_articles, key=lambda t: len(t.get('features', {}).get('text', '')))
        
        # 合并所有文章的信息
        all_sources = []
        all_links = []
        
        for article in cluster_articles:
            source = article.get('source', '未知来源')
            link = article.get('link', '')
            
            if source not in all_sources:
                all_sources.append(source)
            if link and link not in all_links:
                all_links.append(link)

        # 创建合并后的文章
        merged_article = {
            'title': representative['title'],
            'description': representative.get('description', ''),
            'link': representative.get('link', ''),
            'pub_date': representative.get('pub_date'),
            'source': ', '.join(all_sources),
            'guid': representative.get('guid', ''),
            'merged_sources': len(cluster_articles),
            'all_links': all_links
        }

        return merged_article

    def fetch_and_filter_articles(self):
        """获取并筛选今日文章"""
        all_articles = []
        
        for feed_url in self.rss_feeds:
            try:
                print(f"正在获取RSS源: {feed_url}")
                response = requests.get(feed_url, timeout=30)
                response.raise_for_status()
                
                feed = feedparser.parse(response.content)
                
                if feed.bozo:
                    print(f"警告: RSS源可能有格式问题: {feed_url}")
                
                source_name = feed.feed.get('title', feed_url)
                print(f"RSS源标题: {source_name}")
                
                today_articles = []
                for entry in feed.entries:
                    try:
                        # 解析发布日期
                        pub_date = None
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            pub_date = datetime(*entry.updated_parsed[:6])
                        
                        if pub_date and self._is_today_cst(pub_date):
                            article = {
                                'title': entry.get('title', '无标题'),
                                'description': entry.get('description', ''),
                                'link': entry.get('link', ''),
                                'pub_date': pub_date,
                                'source': source_name,
                                'guid': entry.get('id', entry.get('link', ''))
                            }
                            today_articles.append(article)
                    
                    except Exception as e:
                        print(f"处理文章时出错: {e}")
                        continue
                
                print(f"从 {source_name} 获取到 {len(today_articles)} 篇今日文章")
                all_articles.extend(today_articles)
                
            except Exception as e:
                print(f"获取RSS源失败 {feed_url}: {e}")
                continue
        
        print(f"总共获取到 {len(all_articles)} 篇今日文章")
        return all_articles

    def generate_rss(self, merged_topics, unique_topics):
        """生成RSS文件"""
        # 创建RSS根元素
        rss = ET.Element('rss')
        rss.set('version', '2.0')
        rss.set('xmlns:atom', 'http://www.w3.org/2005/Atom')
        
        channel = ET.SubElement(rss, 'channel')
        
        # RSS频道信息
        title = ET.SubElement(channel, 'title')
        title.text = '每日科技新闻聚合'
        
        description = ET.SubElement(channel, 'description')
        description.text = f'聚合爱范儿、极客公园的每日科技新闻 - {self.today_date_str}'
        
        link = ET.SubElement(channel, 'link')
        github_pages_url = 'https://hybridrbt.github.io/rss-merger'
        link.text = github_pages_url
        
        language = ET.SubElement(channel, 'language')
        language.text = 'zh-CN'
        
        # 添加当前CST时间作为lastBuildDate
        last_build_date = ET.SubElement(channel, 'lastBuildDate')
        last_build_date.text = self.today_cst.strftime('%a, %d %b %Y %H:%M:%S %z')
        
        # 添加pubDate
        pub_date_elem = ET.SubElement(channel, 'pubDate')
        pub_date_elem.text = self.today_cst.strftime('%a, %d %b %Y %H:%M:%S %z')
        
        # 添加atom:link自引用
        atom_link = ET.SubElement(channel, 'atom:link')
        atom_link.set('href', f'{github_pages_url}/{self.output_file}')
        atom_link.set('rel', 'self')
        atom_link.set('type', 'application/rss+xml')
        
        # 添加generator信息
        generator = ET.SubElement(channel, 'generator')
        generator.text = '新闻聚合器'
        
        # 添加source信息
        source = ET.SubElement(channel, 'source')
        source.set('url', github_pages_url)
        source.text = '新闻聚合器'
        
        # 生成聚合文章的唯一GUID（基于CST日期）
        date_hash = hashlib.md5(self.today_date_str.encode('utf-8')).hexdigest()[:8]
        aggregated_guid = f"aggregated-news-{self.today_date_str}-{date_hash}"
        
        # 添加聚合文章
        if merged_topics or unique_topics:
            item = ET.SubElement(channel, 'item')
            
            item_title = ET.SubElement(item, 'title')
            item_title.text = f'每日科技新闻聚合 - {self.today_date_str}'
            
            item_description = ET.SubElement(item, 'description')
            
            # 生成HTML内容
            html_content = self._generate_html_content(merged_topics, unique_topics)
            item_description.text = f'<![CDATA[{html_content}]]>'
            
            item_link = ET.SubElement(item, 'link')
            item_link.text = f'{github_pages_url}/{self.html_file}'
            
            item_guid = ET.SubElement(item, 'guid')
            item_guid.set('isPermaLink', 'false')
            item_guid.text = aggregated_guid
            
            item_pub_date = ET.SubElement(item, 'pubDate')
            item_pub_date.text = self.today_cst.strftime('%a, %d %b %Y %H:%M:%S %z')
            
            item_source = ET.SubElement(item, 'source')
            item_source.set('url', github_pages_url)
            item_source.text = '新闻聚合器'
        
        # 写入RSS文件
        tree = ET.ElementTree(rss)
        ET.indent(tree, space="  ", level=0)
        tree.write(self.output_file, encoding='utf-8', xml_declaration=True)
        
        print(f"RSS文件已生成: {self.output_file}")

    def _generate_html_content(self, merged_topics, unique_topics):
        """生成HTML内容"""
        html_parts = []
        html_parts.append('<!DOCTYPE html>')
        html_parts.append('<html lang="zh-CN">')
        html_parts.append('<head>')
        html_parts.append('<meta charset="UTF-8">')
        html_parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
        html_parts.append(f'<title>每日科技新闻聚合 - {self.today_date_str}</title>')
        html_parts.append('<style>')
        html_parts.append('body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }')
        html_parts.append('h1 { color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }')
        html_parts.append('h2 { color: #007acc; margin-top: 30px; }')
        html_parts.append('.topic { margin-bottom: 25px; padding: 15px; border-left: 4px solid #007acc; background-color: #f8f9fa; }')
        html_parts.append('.topic-title { font-weight: bold; font-size: 1.1em; margin-bottom: 8px; }')
        html_parts.append('.topic-content { margin-bottom: 10px; }')
        html_parts.append('.topic-meta { font-size: 0.9em; color: #666; }')
        html_parts.append('.topic-links { margin-top: 8px; }')
        html_parts.append('.topic-links a { margin-right: 15px; color: #007acc; text-decoration: none; }')
        html_parts.append('.topic-links a:hover { text-decoration: underline; }')
        html_parts.append('.toc { background-color: #f0f0f0; padding: 15px; margin-bottom: 30px; border-radius: 5px; }')
        html_parts.append('.toc h3 { margin-top: 0; }')
        html_parts.append('.toc ul { margin: 0; padding-left: 20px; }')
        html_parts.append('.toc a { color: #007acc; text-decoration: none; }')
        html_parts.append('.toc a:hover { text-decoration: underline; }')
        html_parts.append('</style>')
        html_parts.append('</head>')
        html_parts.append('<body>')
        
        html_parts.append(f'<h1>每日科技新闻聚合 - {self.today_date_str}</h1>')
        html_parts.append(f'<p><strong>更新时间:</strong> {self.today_cst.strftime("%Y年%m月%d日 %H:%M:%S CST")}</p>')
        
        # 生成目录
        if merged_topics or unique_topics:
            html_parts.append('<div class="toc">')
            html_parts.append('<h3>📋 今日话题目录</h3>')
            html_parts.append('<ul>')
            
            topic_index = 1
            for topic in merged_topics:
                anchor = f"topic-{topic_index}"
                title = html.escape(topic['title'][:50] + ('...' if len(topic['title']) > 50 else ''))
                sources_info = f" ({topic['merged_sources']}篇文章)" if topic.get('merged_sources', 0) > 1 else ""
                html_parts.append(f'<li><a href="#{anchor}">{title}{sources_info}</a></li>')
                topic_index += 1
            
            for topic in unique_topics:
                anchor = f"topic-{topic_index}"
                title = html.escape(topic['title'][:50] + ('...' if len(topic['title']) > 50 else ''))
                html_parts.append(f'<li><a href="#{anchor}">{title}</a></li>')
                topic_index += 1
            
            html_parts.append('</ul>')
            html_parts.append('</div>')
        
        # 主要话题（合并后的）
        if merged_topics:
            html_parts.append('<h2>🔥 主要话题</h2>')
            topic_index = 1
            for topic in merged_topics:
                anchor = f"topic-{topic_index}"
                html_parts.append(f'<div class="topic" id="{anchor}">')
                html_parts.append(f'<div class="topic-title">{html.escape(topic["title"])}</div>')
                
                if topic.get('description'):
                    clean_desc = re.sub(r'<[^>]+>', '', topic['description'])
                    if len(clean_desc) > 200:
                        clean_desc = clean_desc[:200] + '...'
                    html_parts.append(f'<div class="topic-content">{html.escape(clean_desc)}</div>')
                
                sources_info = f"来源: {html.escape(topic.get('source', '未知'))}"
                if topic.get('merged_sources', 0) > 1:
                    sources_info += f" (合并了{topic['merged_sources']}篇相关文章)"
                
                html_parts.append(f'<div class="topic-meta">{sources_info}</div>')
                
                if topic.get('all_links'):
                    html_parts.append('<div class="topic-links">')
                    for i, link in enumerate(topic['all_links'][:3]):  # 最多显示3个链接
                        html_parts.append(f'<a href="{html.escape(link)}" target="_blank">阅读原文 {i+1}</a>')
                    html_parts.append('</div>')
                elif topic.get('link'):
                    html_parts.append('<div class="topic-links">')
                    html_parts.append(f'<a href="{html.escape(topic["link"])}" target="_blank">阅读原文</a>')
                    html_parts.append('</div>')
                
                html_parts.append('</div>')
                topic_index += 1
        
        # 独特话题
        if unique_topics:
            html_parts.append('<h2>📰 其他新闻</h2>')
            for topic in unique_topics:
                anchor = f"topic-{topic_index}"
                html_parts.append(f'<div class="topic" id="{anchor}">')
                html_parts.append(f'<div class="topic-title">{html.escape(topic["title"])}</div>')
                
                if topic.get('description'):
                    clean_desc = re.sub(r'<[^>]+>', '', topic['description'])
                    if len(clean_desc) > 200:
                        clean_desc = clean_desc[:200] + '...'
                    html_parts.append(f'<div class="topic-content">{html.escape(clean_desc)}</div>')
                
                html_parts.append(f'<div class="topic-meta">来源: {html.escape(topic.get("source", "未知"))}</div>')
                
                if topic.get('link'):
                    html_parts.append('<div class="topic-links">')
                    html_parts.append(f'<a href="{html.escape(topic["link"])}" target="_blank">阅读原文</a>')
                    html_parts.append('</div>')
                
                html_parts.append('</div>')
                topic_index += 1
        
        if not merged_topics and not unique_topics:
            html_parts.append('<p>今日暂无新闻内容。</p>')
        
        html_parts.append('</body>')
        html_parts.append('</html>')
        
        return '\n'.join(html_parts)

    def generate_html_file(self, merged_topics, unique_topics):
        """生成独立的HTML文件"""
        html_content = self._generate_html_content(merged_topics, unique_topics)
        
        with open(self.html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML文件已生成: {self.html_file}")

def main():
    aggregator = DateFilteredRSSAggregator()
    
    # 获取今日文章
    articles = aggregator.fetch_and_filter_articles()
    
    if not articles:
        print("今日没有找到任何文章，生成空的RSS文件")
        aggregator.generate_rss([], [])
        aggregator.generate_html_file([], [])
    else:
        # 聚类和去重
        merged_topics, unique_topics = aggregator._cluster_and_filter_topics(articles)
        
        # 生成RSS和HTML文件
        aggregator.generate_rss(merged_topics, unique_topics)
        aggregator.generate_html_file(merged_topics, unique_topics)
    
    print(f"RSS聚合去重完成，结果保存在: {aggregator.output_file} 和 {aggregator.html_file}")

if __name__ == "__main__":
    main()