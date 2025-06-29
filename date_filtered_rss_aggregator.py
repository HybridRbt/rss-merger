#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最终版话题级别RSS聚合器
- 仅保留爱范儿、少数派和极客公园三个源
- 对少数派内容进行特殊处理（去重或附加到末尾）
- 生成平铺式话题列表输出
- 修复RSS链接指向GitHub Pages实际地址
- 添加日期过滤功能，只保留当天发布的内容
"""

import feedparser
import jieba
import json
import os
import time
import re
import hashlib
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from bs4 import BeautifulSoup

class FinalRSSAggregator:
    def __init__(self, rss_feeds, output_file="enhanced_rss.xml", html_output_file="daily_news.html", github_username=None, github_repo=None):
        self.rss_feeds = rss_feeds
        self.output_file = output_file
        self.html_output_file = html_output_file
        self.github_username = github_username
        self.github_repo = github_repo
        self.vectorizer = TfidfVectorizer(tokenizer=self._tokenize, min_df=1, max_df=0.95)
        self.topic_similarity_threshold = 0.55
        self.time_window = 3
        self.stopwords = self._load_stopwords()
        os.makedirs(os.path.dirname(os.path.abspath(output_file)) if os.path.dirname(output_file) else ".", exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(html_output_file)) if os.path.dirname(html_output_file) else ".", exist_ok=True)
        
        # 设置当天日期范围，用于过滤内容
        self.today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.tomorrow = self.today + timedelta(days=1)

    def _load_stopwords(self):
        # 常见中文停用词
        stopwords = set([
            '的', '了', '和', '是', '就', '都', '而', '及', '与', '这', '那', '有', '在',
            '中', '为', '对', '到', '以', '等', '上', '下', '由', '于', '或', '如', '我',
            '你', '他', '她', '它', '们', '个', '之', '也', '但', '还', '只', '又', '并',
            '很', '将', '没', '说', '被', '着', '给', '让', '从', '向', '地', '得', '更',
            '日前', '据悉', '据介绍', '据报道', '目前', '近日', '昨日', '今日', '此前',
            '另外', '同时', '此外', '值得一提', '具体来看', '据了解', '据称', '据悉',
            '表示', '认为', '指出', '称', '透露', '介绍', '不过', '其中', '这些', '那些',
            '一些', '一个', '这个', '那个', '可以', '不能', '这样', '那样', '如此', '所以',
            '因此', '因为', '如果', '虽然', '即使', '尽管', '无论', '除了', '只有', '几乎',
            '正在', '已经', '曾经', '将要', '可能', '应该', '必须', '需要', '能够', '不会',
            '一直', '一定', '一般', '通常', '经常', '总是', '从来', '不断', '继续', '开始',
            '结束', '完成', '进行', '发生', '出现', '成为', '变成', '看到', '听到', '知道',
            '觉得', '认为', '希望', '想要', '喜欢', '讨厌', '害怕', '担心', '相信', '怀疑'
        ])
        return stopwords

    def fetch_feeds(self):
        all_entries = []
        for feed_info in self.rss_feeds:
            try:
                print(f"正在获取 {feed_info['name']} 的RSS内容...")
                feed = feedparser.parse(feed_info['url'])
                for entry in feed.entries:
                    standardized_entry = self._standardize_entry(entry, feed_info['name'])
                    
                    # 只保留当天发布的内容
                    if self._is_published_today(standardized_entry):
                        all_entries.append(standardized_entry)
                        
                print(f"成功获取 {feed_info['name']} 的当天内容")
            except Exception as e:
                print(f"获取 {feed_info['name']} 内容时出错: {str(e)}")
        
        print(f"总共获取了 {len(all_entries)} 条当天发布的内容")
        return all_entries

    def _is_published_today(self, entry):
        """检查条目是否在当天发布"""
        if not entry.get('published_parsed'):
            # 如果没有日期信息，默认保留
            return True
            
        pub_time = datetime.fromtimestamp(time.mktime(entry['published_parsed']))
        
        # 检查是否在今天的日期范围内
        return self.today <= pub_time < self.tomorrow

    def _standardize_entry(self, entry, source):
        published = entry.get('published', '')
        published_parsed = entry.get('published_parsed')
        content = ''
        if 'content' in entry:
            for content_item in entry.content:
                if 'value' in content_item:
                    content = content_item.value
                    break
        if not content and 'summary' in entry:
            content = entry.summary
        standardized = {
            'title': entry.get('title', ''),
            'link': entry.get('link', ''),
            'summary': content,
            'published': published,
            'published_parsed': published_parsed,
            'source': source,
            'id': hashlib.md5(entry.get('link', '').encode()).hexdigest()
        }
        return standardized

    def _tokenize(self, text):
        words = jieba.cut(text)
        return " ".join([w for w in words if w not in self.stopwords and len(w.strip()) > 1])

    def _clean_html(self, html_content, source=None):
        if not html_content:
            return ""
        soup = BeautifulSoup(html_content, 'html.parser')
        if source == "少数派":
            for element in soup.find_all(['style', 'script', 'iframe', 'noscript']):
                element.decompose()
            main_content = soup.find('div', class_='content')
            if main_content:
                soup = main_content
            for blockquote in soup.find_all('blockquote'):
                blockquote.insert_before(soup.new_tag('br'))
                blockquote.insert_after(soup.new_tag('br'))
            for p in soup.find_all('p'):
                if p.get_text().strip():
                    p.insert_after(soup.new_tag('br'))
        text = soup.get_text(separator='\n', strip=True)
        text = re.sub(r'\n+', '\n', text).strip()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _split_into_topics(self, entry):
        content = entry.get('summary', '')
        source = entry.get('source', '')
        if not content:
            return [{
                'text': entry.get('title', ''), 'source': source, 'link': entry.get('link', ''),
                'published': entry.get('published', ''), 'published_parsed': entry.get('published_parsed'),
                'entry_id': entry.get('id', ''), 'title': entry.get('title', ''), 'is_sspai': source == "少数派"
            }]
        clean_content = self._clean_html(content, source)
        topics = []
        if source == "少数派":
            topics = self._split_sspai_content(content, clean_content)
        if not topics or len(topics) <= 1:
            topics = self._split_by_html_structure(content)
        if not topics or len(topics) <= 1:
            topics = self._split_by_text_features(clean_content)
        if not topics:
            topics = [{'text': clean_content, 'topic_title': None}]
        
        final_topics = []
        for topic in topics:
            topic['source'] = source
            topic['link'] = entry.get('link', '')
            topic['published'] = entry.get('published', '')
            topic['published_parsed'] = entry.get('published_parsed')
            topic['entry_id'] = entry.get('id', '')
            topic['title'] = entry.get('title', '') # 原始条目标题
            topic['is_sspai'] = (source == "少数派") # 标记是否来自少数派
            if len(topic.get('text', '').strip()) < 30:
                 topic['text'] = entry.get('title', '') + ' ' + topic.get('text', '')
            if not topic.get('topic_title'):
                topic['topic_title'] = self._generate_topic_title(topic['text'])
            final_topics.append(topic)
            
        merged_topics = self._merge_short_topics(final_topics)
        return merged_topics

    def _split_sspai_content(self, html_content, clean_content):
        if not html_content:
            return []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            for element in soup.find_all(['style', 'script', 'iframe', 'noscript']):
                element.decompose()
            topics = []
            headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'b'])
            if headers and len(headers) > 1:
                current_topic = {"text": "", "topic_title": None}
                current_header = None
                for element in soup.descendants:
                    if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'b']:
                        if current_topic["text"].strip():
                            topics.append(current_topic)
                        current_header = element.get_text(strip=True)
                        current_topic = {"text": current_header + "\n", "topic_title": current_header}
                    elif current_header and element.string and element.string.strip():
                        current_topic["text"] += element.string.strip() + " "
                if current_topic["text"].strip():
                    topics.append(current_topic)
                return topics
            paragraphs = soup.find_all('p')
            if paragraphs and len(paragraphs) > 3:
                current_topic = {"text": "", "topic_title": None}
                para_count = 0
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if not text:
                        continue
                    if para_count == 0:
                        current_topic["topic_title"] = text
                    current_topic["text"] += text + "\n"
                    para_count += 1
                    if para_count >= 3:
                        topics.append(current_topic)
                        current_topic = {"text": "", "topic_title": None}
                        para_count = 0
                if current_topic["text"].strip():
                    topics.append(current_topic)
                return topics
            if not topics:
                return self._split_by_text_features(clean_content)
            return topics
        except Exception as e:
            print(f"少数派内容分割出错: {str(e)}")
            return []

    def _generate_topic_title(self, text):
        sentences = re.split(r'[。！？.!?]+', text)
        first_sentence = sentences[0].strip() if sentences else text
        if len(first_sentence) > 30:
            title = first_sentence[:30] + "..."
        else:
            title = first_sentence
        return title

    def _merge_short_topics(self, topics):
        if len(topics) <= 1:
            return topics
        result = []
        current_topic = None
        for topic in topics:
            if not current_topic:
                current_topic = topic
                continue
            if len(current_topic.get('text', '').strip()) < 100 or len(topic.get('text', '').strip()) < 100:
                current_topic['text'] = current_topic.get('text', '') + ' ' + topic.get('text', '')
            else:
                result.append(current_topic)
                current_topic = topic
        if current_topic:
            result.append(current_topic)
        return result

    def _split_by_html_structure(self, html_content):
        if not html_content:
            return []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            topics = []
            hr_tags = soup.find_all('hr')
            if hr_tags and len(hr_tags) > 1:
                segments = []
                current_segment = []
                for element in soup.body.children if soup.body else soup.children:
                    if element.name == 'hr':
                        if current_segment:
                            segments.append(current_segment)
                            current_segment = []
                    else:
                        current_segment.append(element)
                if current_segment:
                    segments.append(current_segment)
                for segment in segments:
                    segment_text = ' '.join([elem.get_text(strip=True) for elem in segment if elem.get_text(strip=True)])
                    if segment_text.strip():
                        topic_title = None
                        for elem in segment:
                            if elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'b']:
                                topic_title = elem.get_text(strip=True)
                                break
                        topics.append({'text': segment_text, 'topic_title': topic_title})
                return topics
            headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if headers:
                for header in headers:
                    title = header.get_text(strip=True)
                    content = []
                    for sibling in header.next_siblings:
                        if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            break
                        if sibling.string and sibling.string.strip():
                            content.append(sibling.get_text(strip=True))
                    topic_text = " ".join(content)
                    if len(topic_text.strip()) > 10:
                        topics.append({'text': topic_text, 'topic_title': title})
            if not topics:
                divs = soup.find_all('div', recursive=False)
                if len(divs) > 1:
                    for div in divs:
                        text = div.get_text(strip=True)
                        if len(text) > 30:
                            topic_title = None
                            header = div.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'b'])
                            if header:
                                topic_title = header.get_text(strip=True)
                            topics.append({'text': text, 'topic_title': topic_title})
            return topics
        except Exception as e:
            print(f"HTML结构分割出错: {str(e)}")
            return []

    def _split_by_text_features(self, text):
        if not text or len(text) < 100:
            return [{'text': text, 'topic_title': None}] if text else []
        topics = []
        separators = [r'\n\s*\n', r'\*{3,}', r'-{3,}', r'={3,}', r'【.*?】', r'■.*?■', r'▌.*?▌', r'\d+[\.\s、]+', r'•\s+']
        pattern = '|'.join(separators)
        segments = re.split(pattern, text)
        if len(segments) > 1:
            for segment in segments:
                segment = segment.strip()
                if len(segment) > 50:
                    lines = segment.split('\n')
                    topic_title = None
                    if len(lines) > 1 and len(lines[0]) < 50:
                        topic_title = lines[0].strip()
                    topics.append({'text': segment, 'topic_title': topic_title})
            return topics
        topic_shift_markers = ['另外', '此外', '同时', '值得一提的是', '据悉', '据报道', '据了解', '另据', '与此同时', '除此之外', '不仅如此', '更重要的是', '与此相关', '相关消息', '最新消息', '最新进展', '最新动态']
        for marker in topic_shift_markers:
            if marker in text:
                segments = text.split(marker)
                if len(segments) > 1:
                    topics.append({'text': segments[0].strip(), 'topic_title': None})
                    for i in range(1, len(segments)):
                        if segments[i].strip():
                            topics.append({'text': marker + ' ' + segments[i].strip(), 'topic_title': None})
                    return topics
        sentences = re.split(r'[。！？.!?]+', text)
        if len(sentences) > 5:
            current_topic = ""
            sentence_count = 0
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                current_topic += sentence + "。"
                sentence_count += 1
                if sentence_count >= 3 and len(current_topic) > 100:
                    topics.append({'text': current_topic, 'topic_title': None})
                    current_topic = ""
                    sentence_count = 0
            if current_topic:
                topics.append({'text': current_topic, 'topic_title': None})
            return topics
        return [{'text': text, 'topic_title': None}]

    def _extract_topic_features(self, topic):
        text = topic.get('text', '')
        if not text:
            return None
        words = [w for w in jieba.cut(text) if w not in self.stopwords and len(w.strip()) > 1]
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:20]]
        entities = []
        for word in keywords:
            if len(word) >= 2 and not any(char.isdigit() for char in word):
                entities.append(word)
        sentences = re.split(r'[。！？.!?]+', text)
        core_sentences = [s.strip() for s in sentences[:3] if len(s.strip()) > 10]
        fingerprint = {
            'keywords': keywords,
            'entities': entities[:10],
            'core_sentences': core_sentences,
            'text': text,
            'topic_title': topic.get('topic_title') or self._generate_topic_title(text)
        }
        return fingerprint

    def _calculate_topic_similarity(self, topic1, topic2):
        features1 = topic1.get('features', {})
        features2 = topic2.get('features', {})
        if not features1 or not features2:
            return 0.0
        entities1 = set(features1.get('entities', []))
        entities2 = set(features2.get('entities', []))
        entity_sim = 0
        if entities1 and entities2:
            entity_intersection = entities1.intersection(entities2)
            entity_union = entities1.union(entities2)
            entity_sim = len(entity_intersection) / len(entity_union) if entity_union else 0
            if len(entity_intersection) >= 3:
                entity_sim = min(1.0, entity_sim * 1.3)
        keywords1 = features1.get('keywords', [])
        keywords2 = features2.get('keywords', [])
        keyword_sim = 0
        if keywords1 and keywords2:
            kw_set1 = set(keywords1)
            kw_set2 = set(keywords2)
            kw_intersection = kw_set1.intersection(kw_set2)
            kw_union = kw_set1.union(kw_set2)
            keyword_sim = len(kw_intersection) / len(kw_union) if kw_union else 0
            if len(kw_intersection) >= 5:
                keyword_sim = min(1.0, keyword_sim * 1.3)
        text1 = features1.get('text', '')
        text2 = features2.get('text', '')
        cosine_sim = 0
        if text1 and text2:
            try:
                tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except:
                words1 = set(self._tokenize(text1).split())
                words2 = set(self._tokenize(text2).split())
                overlap = words1.intersection(words2)
                cosine_sim = len(overlap) / (len(words1) + len(words2) - len(overlap)) if (len(words1) + len(words2) - len(overlap)) > 0 else 0
        title1 = features1.get('topic_title', '')
        title2 = features2.get('topic_title', '')
        title_sim = 0
        if title1 and title2:
            words1 = set(self._tokenize(title1).split())
            words2 = set(self._tokenize(title2).split())
            if words1 and words2:
                overlap = words1.intersection(words2)
                title_sim = len(overlap) / (len(words1) + len(words2) - len(overlap)) if (len(words1) + len(words2) - len(overlap)) > 0 else 0
                if title_sim > 0.7:
                    title_sim = min(1.0, title_sim * 1.5)
        combined_sim = (entity_sim * 0.35 + keyword_sim * 0.25 + cosine_sim * 0.2 + title_sim * 0.2)
        return combined_sim

    def _cluster_and_filter_topics(self, topics):
        """聚类相似话题，并根据少数派规则进行过滤"""
        if not topics or len(topics) <= 1:
            return topics, []

        n_topics = len(topics)
        similarity_matrix = np.zeros((n_topics, n_topics))
        for i in range(n_topics):
            for j in range(i + 1, n_topics):
                sim = self._calculate_topic_similarity(topics[i], topics[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        visited = [False] * n_topics
        clusters = []
        for i in range(n_topics):
            if visited[i]:
                continue
            cluster = [i]
            visited[i] = True
            queue = [i]
            while queue:
                current = queue.pop(0)
                for j in range(n_topics):
                    if not visited[j] and similarity_matrix[current, j] >= self.topic_similarity_threshold:
                        cluster.append(j)
                        visited[j] = True
                        queue.append(j)
            clusters.append(cluster)

        merged_topics = []
        unique_sspai_topics = []

        for cluster_indices in clusters:
            cluster_topics = [topics[idx] for idx in cluster_indices]
            
            # 检查簇中是否有非少数派话题
            has_non_sspai = any(not t.get('is_sspai', False) for t in cluster_topics)
            
            # 如果簇中只有少数派话题
            if not has_non_sspai:
                # 选择最长的话题作为代表，并标记为待处理的少数派话题
                representative_sspai = max(cluster_topics, key=lambda t: len(t.get('features', {}).get('text', '')))
                unique_sspai_topics.append(representative_sspai)
            else:
                # 如果簇中有非少数派话题，则只保留非少数派话题
                non_sspai_topics = [t for t in cluster_topics if not t.get('is_sspai', False)]
                # 合并这些非少数派话题
                merged_topic = self._merge_topic_cluster(non_sspai_topics)
                if merged_topic:
                    merged_topics.append(merged_topic)

        print(f"话题聚类与过滤后，保留 {len(merged_topics)} 个主要话题和 {len(unique_sspai_topics)} 个独特少数派话题")
        return merged_topics, unique_sspai_topics

    def _merge_topic_cluster(self, topic_cluster):
        if not topic_cluster:
            return None
        if len(topic_cluster) == 1:
            return topic_cluster[0]
        base_topic = max(topic_cluster, key=lambda t: len(t.get('features', {}).get('text', '')))
        sources = []
        for topic in topic_cluster:
            source = topic.get('source')
            if source and source not in sources:
                sources.append(source)
        merged_topic = {
            'text': base_topic.get('features', {}).get('text', ''),
            'sources': sources,
            'link': base_topic.get('link', ''),
            'published': base_topic.get('published', ''),
            'published_parsed': base_topic.get('published_parsed'),
            'features': base_topic.get('features', {}),
            'entry_id': base_topic.get('entry_id', ''),
            'source': base_topic.get('source', ''),
            'topic_title': base_topic.get('features', {}).get('topic_title', '')
        }
        base_text = merged_topic['text']
        for topic in topic_cluster:
            if topic == base_topic:
                continue
            topic_text = topic.get('features', {}).get('text', '')
            if topic_text and not self._is_text_contained(topic_text, base_text):
                supplement = self._extract_supplement(topic_text, base_text)
                if supplement:
                    merged_topic['text'] += "\n\n补充信息: " + supplement
        return merged_topic

    def _is_text_contained(self, text1, text2):
        words1 = set(self._tokenize(text1).split())
        words2 = set(self._tokenize(text2).split())
        if len(words1) == 0:
            return True
        overlap_ratio = len(words1.intersection(words2)) / len(words1)
        return overlap_ratio > 0.8

    def _extract_supplement(self, text, base_text):
        sentences1 = re.split(r'[。！？.!?]+', text)
        sentences2 = re.split(r'[。！？.!?]+', base_text)
        supplements = []
        for sentence in sentences1:
            sentence = sentence.strip()
            if not sentence:
                continue
            contained = False
            for base_sentence in sentences2:
                if sentence in base_sentence or self._is_text_contained(sentence, base_sentence):
                    contained = True
                    break
            if not contained:
                supplements.append(sentence)
        return "。".join(supplements) if supplements else ""

    def process_topics(self, entries):
        """处理RSS条目，提取话题，去重并处理少数派内容"""
        all_topics = []
        for entry in entries:
            topics = self._split_into_topics(entry)
            for topic in topics:
                features = self._extract_topic_features(topic)
                if features:
                    topic['features'] = features
                    all_topics.append(topic)
        print(f"从 {len(entries)} 个条目中提取出 {len(all_topics)} 个话题")
        
        # 聚类并根据少数派规则过滤
        merged_topics, unique_sspai_topics = self._cluster_and_filter_topics(all_topics)
        
        return merged_topics, unique_sspai_topics

    def get_github_pages_url(self):
        """获取GitHub Pages URL"""
        if self.github_username and self.github_repo:
            return f"https://{self.github_username}.github.io/{self.github_repo}"
        return None

    def generate_flat_html(self, main_topics, unique_sspai_topics):
        """生成平铺式HTML文章，将独特少数派内容附加到末尾"""
        today = datetime.now().strftime("%Y年%m月%d日")
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>每日新闻聚合 - {today}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f9f9f9; }}
        h1 {{ text-align: center; color: #2c3e50; margin-bottom: 30px; padding-bottom: 10px; border-bottom: 1px solid #eee; }}
        h2 {{ color: #3498db; margin-top: 30px; padding-bottom: 8px; border-bottom: 1px solid #eee; }}
        .topic-content {{ background-color: white; padding: 15px; margin-bottom: 20px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .meta {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 10px; }}
        a {{ color: #3498db; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .footer {{ text-align: center; margin-top: 50px; color: #7f8c8d; font-size: 0.9em; }}
        .toc {{ background-color: white; padding: 15px; margin-bottom: 30px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .toc ul {{ list-style-type: none; padding-left: 20px; }}
        .toc li {{ margin-bottom: 5px; }}
        .sspai-section {{ margin-top: 50px; padding-top: 30px; border-top: 2px dashed #ccc; }}
        .sspai-topic {{ background-color: #f0f0f0; padding: 15px; margin-bottom: 20px; border-radius: 5px; border-left: 5px solid #ccc; }}
    </style>
</head>
<body>
    <h1>每日新闻聚合 - {today}</h1>
"""
        
        # 添加日期过滤说明
        html += f"""<div class="topic-content">
    <p><strong>说明</strong>：本页面仅包含 {today} 发布的内容。</p>
</div>
"""
        
        # 按发布时间排序主要话题
        sorted_main_topics = sorted(
            main_topics,
            key=lambda x: time.mktime(x['published_parsed']) if x.get('published_parsed') else 0,
            reverse=True
        )
        
        # 添加目录 (只包含主要话题)
        html += "<div class='toc'>\n<h2>目录</h2>\n<ul>\n"
        for i, topic in enumerate(sorted_main_topics):
            topic_title = topic.get('topic_title') or topic.get('features', {}).get('topic_title', f'话题 {i+1}')
            html += f'<li><a href="#{self._make_id(topic_title)}">{topic_title}</a></li>\n'
        html += "</ul>\n</div>\n"
        
        # 添加每个主要话题的内容
        for topic in sorted_main_topics:
            topic_title = topic.get('topic_title') or topic.get('features', {}).get('topic_title', '未命名话题')
            topic_text = topic.get('text', '')
            topic_link = topic.get('link', '')
            topic_source = topic.get('source', '')
            html += f'<h2 id="{self._make_id(topic_title)}">{topic_title}</h2>\n'
            html += '<div class="topic-content">\n'
            if topic_link:
                html += f'<div class="meta">来源: <a href="{topic_link}" target="_blank">{topic_source}</a></div>\n'
            paragraphs = topic_text.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    html += f'<p>{paragraph}</p>\n'
            html += '</div>\n'
            
        # 添加独特的少数派话题部分
        if unique_sspai_topics:
            html += "<div class='sspai-section'>\n<h2>少数派独特内容（格式可能不佳）</h2>\n"
            # 按发布时间排序少数派话题
            sorted_sspai_topics = sorted(
                unique_sspai_topics,
                key=lambda x: time.mktime(x['published_parsed']) if x.get('published_parsed') else 0,
                reverse=True
            )
            for topic in sorted_sspai_topics:
                topic_title = topic.get('topic_title') or topic.get('features', {}).get('topic_title', '少数派话题')
                # 直接使用原始摘要内容
                original_entry = next((e for e in self.all_entries if e['id'] == topic['entry_id']), None)
                topic_content = original_entry['summary'] if original_entry else topic.get('text', '')
                topic_link = topic.get('link', '')
                html += f'<h3>{topic_title}</h3>\n'
                html += '<div class="sspai-topic">\n'
                if topic_link:
                    html += f'<div class="meta">来源: <a href="{topic_link}" target="_blank">少数派</a></div>\n'
                # 直接嵌入原始HTML内容
                html += f'<div>{topic_content}</div>\n'
                html += '</div>\n'
            html += "</div>\n"
        
        html += f"""
    <div class="footer">
        <p>内容自动聚合于 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>仅包含 {today} 发布的内容</p>
    </div>
</body>
</html>
"""
        return html

    def _make_id(self, text):
        id_text = re.sub(r'[^a-zA-Z0-9]', '_', text)
        if id_text and id_text[0].isdigit():
            id_text = 'id_' + id_text
        return id_text

    def reconstruct_rss(self, main_topics, unique_sspai_topics):
        """重构RSS，将独特少数派内容附加到主条目"""
        all_content_html = self.generate_flat_html(main_topics, unique_sspai_topics)
        today = datetime.now().strftime("%Y年%m月%d日")
        
        # 获取GitHub Pages URL或使用默认值
        github_pages_url = self.get_github_pages_url()
        if github_pages_url:
            html_url = f"{github_pages_url}/{self.html_output_file}"
        else:
            # 如果没有提供GitHub用户名和仓库名，使用相对路径
            html_url = self.html_output_file
        
        main_item = {
            'title': f"每日新闻聚合 - {today}",
            'link': html_url,  # 使用实际的GitHub Pages URL
            'summary': all_content_html,
            'published': datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0800"),
            'published_parsed': time.localtime(),
            'source': "新闻聚合器",
            'id': hashlib.md5(f"daily_news_{datetime.now().strftime('%Y%m%d')}".encode()).hexdigest()
        }
        return [main_item]

    def generate_rss(self, items):
        rss = ET.Element("rss", version="2.0")
        channel = ET.SubElement(rss, "channel")
        
        # 获取GitHub Pages URL或使用默认值
        github_pages_url = self.get_github_pages_url()
        if github_pages_url:
            channel_link = github_pages_url
        else:
            channel_link = "."  # 使用相对路径
            
        ET.SubElement(channel, "title").text = "每日新闻聚合"
        ET.SubElement(channel, "link").text = channel_link
        ET.SubElement(channel, "description").text = "多源新闻内容聚合，已进行话题级别去重"
        ET.SubElement(channel, "language").text = "zh-cn"
        ET.SubElement(channel, "lastBuildDate").text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0800")
        generator = ET.SubElement(channel, "generator")
        generator.text = "Final RSS Aggregator"
        for item in items:
            rss_item = ET.SubElement(channel, "item")
            ET.SubElement(rss_item, "title").text = item['title']
            ET.SubElement(rss_item, "link").text = item['link']
            # 使用CDATA包装HTML内容
            description = ET.SubElement(rss_item, "description")
            description.text = f"<![CDATA[{item['summary']}]]>"
            if item.get('published'):
                ET.SubElement(rss_item, "pubDate").text = item['published']
            guid = ET.SubElement(rss_item, "guid", isPermaLink="false")
            guid.text = item['id']
            source = ET.SubElement(rss_item, "source")
            source.text = item['source']
        rough_string = ET.tostring(rss, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ", encoding="utf-8").decode('utf-8')
        return pretty_xml

    def save_rss(self, xml_content):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        print(f"合并后的RSS已保存到 {self.output_file}")

    def save_html(self, main_topics, unique_sspai_topics):
        """保存HTML文章到文件"""
        html_content = self.generate_flat_html(main_topics, unique_sspai_topics)
        with open(self.html_output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"结构化HTML文章已保存到 {self.html_output_file}")

    def process(self):
        """处理全部RSS聚合流程"""
        self.all_entries = self.fetch_feeds() # 保存所有条目以供后续查找
        main_topics, unique_sspai_topics = self.process_topics(self.all_entries)
        self.save_html(main_topics, unique_sspai_topics)
        rss_items = self.reconstruct_rss(main_topics, unique_sspai_topics)
        merged_rss = self.generate_rss(rss_items)
        self.save_rss(merged_rss)
        return self.output_file, self.html_output_file

# 主函数
def main():
    # 仅保留三个源
    rss_feeds = [
        {"name": "爱范儿", "url": "https://appd.top/feed/ifanr"},
        {"name": "少数派", "url": "https://appd.top/feed/sspai"}, # 少数派仍然获取，但会特殊处理
        {"name": "极客公园", "url": "https://appd.top/feed/geekpark"}
    ]
    
    # 请替换为您的GitHub用户名和仓库名
    github_username = "hybridrbt"  # 例如: "johndoe"
    github_repo = "rss-merger"          # 例如: "rss-merger"
    
    aggregator = FinalRSSAggregator(
        rss_feeds,
        output_file="final_rss.xml",
        html_output_file="final_daily_news.html",
        github_username=github_username,
        github_repo=github_repo
    )
    rss_file, html_file = aggregator.process()
    print(f"RSS聚合与去重完成，结果保存在: {rss_file} 和 {html_file}")

if __name__ == "__main__":
    main()
