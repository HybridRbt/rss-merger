#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
话题级别RSS合并与去重系统 - 优化版
能够识别并去除RSS条目中的重复话题，即使它们出现在不同的条目或有不同的描述
增强了话题分割和相似度判定算法，提高去重准确率
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
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from bs4 import BeautifulSoup

class TopicRSSMerger:
    def __init__(self, rss_feeds, output_file="merged_rss.xml"):
        """
        初始化话题级别RSS合并器
        
        Args:
            rss_feeds: RSS源列表，每项包含name和url
            output_file: 输出的合并RSS文件路径
        """
        self.rss_feeds = rss_feeds
        self.output_file = output_file
        
        # 初始化TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize,
            min_df=1,
            max_df=0.95
        )
        
        # 设置相似度阈值 - 优化版降低阈值以提高聚类效果
        self.topic_similarity_threshold = 0.6
        
        # 设置时间窗口（天）
        self.time_window = 3
        
        # 停用词列表
        self.stopwords = self._load_stopwords()
        
        # 创建输出目录
        os.makedirs(os.path.dirname(os.path.abspath(output_file)) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    def _load_stopwords(self):
        """加载中文停用词"""
        # 常见中文停用词
        stopwords = set([
            '的', '了', '和', '是', '就', '都', '而', '及', '与', '这', '那', '有', '在',
            '中', '为', '对', '到', '以', '等', '上', '下', '由', '于', '或', '如', '我',
            '你', '他', '她', '它', '们', '个', '之', '也', '但', '还', '只', '又', '并',
            '很', '将', '没', '说', '被', '着', '给', '让', '从', '向', '地', '得', '更',
            '日前', '据悉', '据介绍', '据报道', '目前', '近日', '昨日', '今日', '此前',
            '另外', '同时', '此外', '值得一提', '具体来看', '据了解', '据称', '据悉'
        ])
        return stopwords
    
    def fetch_feeds(self):
        """获取所有RSS源的内容"""
        all_entries = []
        
        for feed_info in self.rss_feeds:
            try:
                print(f"正在获取 {feed_info['name']} 的RSS内容...")
                feed = feedparser.parse(feed_info['url'])
                
                # 处理每个条目
                for entry in feed.entries:
                    # 标准化条目
                    standardized_entry = self._standardize_entry(entry, feed_info['name'])
                    all_entries.append(standardized_entry)
                
                print(f"成功获取 {feed_info['name']} 的 {len(feed.entries)} 条内容")
            except Exception as e:
                print(f"获取 {feed_info['name']} 内容时出错: {str(e)}")
        
        return all_entries
    
    def _standardize_entry(self, entry, source):
        """将RSS条目标准化为统一格式"""
        # 获取发布时间
        published = entry.get('published', '')
        published_parsed = entry.get('published_parsed')
        
        # 获取内容
        content = ''
        if 'content' in entry:
            for content_item in entry.content:
                if 'value' in content_item:
                    content = content_item.value
                    break
        
        if not content and 'summary' in entry:
            content = entry.summary
        
        # 标准化条目
        standardized = {
            'title': entry.get('title', ''),
            'link': entry.get('link', ''),
            'summary': content,
            'published': published,
            'published_parsed': published_parsed,
            'source': source,
            # 生成唯一ID
            'id': hashlib.md5(entry.get('link', '').encode()).hexdigest()
        }
        
        return standardized
    
    def _tokenize(self, text):
        """对文本进行中文分词并去除停用词"""
        words = jieba.cut(text)
        return " ".join([w for w in words if w not in self.stopwords and len(w.strip()) > 1])
    
    def _clean_html(self, html_content):
        """清理HTML内容，保留文本"""
        if not html_content:
            return ""
        
        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 获取纯文本
        text = soup.get_text(separator=' ', strip=True)
        
        # 清理多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _split_into_topics(self, entry):
        """将RSS条目分割为多个话题 - 优化版"""
        # 获取内容
        content = entry.get('summary', '')
        if not content:
            # 如果没有摘要，使用标题作为唯一话题
            return [{
                'text': entry.get('title', ''),
                'source': entry.get('source', ''),
                'link': entry.get('link', ''),
                'published': entry.get('published', ''),
                'published_parsed': entry.get('published_parsed'),
                'entry_id': entry.get('id', '')
            }]
        
        # 清理HTML
        clean_content = self._clean_html(content)
        
        # 尝试基于HTML结构分割话题
        topics = self._split_by_html_structure(content)
        
        # 如果HTML结构分割失败或结果不理想，尝试基于文本特征分割
        if not topics or len(topics) <= 1:
            topics = self._split_by_text_features(clean_content)
        
        # 如果所有分割方法都失败，将整个条目作为一个话题
        if not topics:
            topics = [{
                'text': clean_content,
                'source': entry.get('source', ''),
                'link': entry.get('link', ''),
                'published': entry.get('published', ''),
                'published_parsed': entry.get('published_parsed'),
                'entry_id': entry.get('id', '')
            }]
        else:
            # 为每个话题添加元数据
            for topic in topics:
                topic['source'] = entry.get('source', '')
                topic['link'] = entry.get('link', '')
                topic['published'] = entry.get('published', '')
                topic['published_parsed'] = entry.get('published_parsed')
                topic['entry_id'] = entry.get('id', '')
                
                # 优化：确保每个话题有足够的内容
                if len(topic.get('text', '').strip()) < 30:
                    topic['text'] = entry.get('title', '') + ' ' + topic.get('text', '')
        
        # 优化：合并过短的话题
        merged_topics = self._merge_short_topics(topics)
        
        return merged_topics
    
    def _merge_short_topics(self, topics):
        """合并过短的话题片段"""
        if len(topics) <= 1:
            return topics
        
        result = []
        current_topic = None
        
        for topic in topics:
            if not current_topic:
                current_topic = topic
                continue
            
            # 如果当前话题或下一个话题太短，合并它们
            if len(current_topic.get('text', '').strip()) < 100 or len(topic.get('text', '').strip()) < 100:
                current_topic['text'] = current_topic.get('text', '') + ' ' + topic.get('text', '')
            else:
                result.append(current_topic)
                current_topic = topic
        
        if current_topic:
            result.append(current_topic)
        
        return result
    
    def _split_by_html_structure(self, html_content):
        """基于HTML结构分割话题 - 优化版"""
        if not html_content:
            return []
        
        try:
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 查找可能的话题分隔符
            topics = []
            
            # 方法1: 查找<hr>标签作为分隔符
            hr_tags = soup.find_all('hr')
            if hr_tags and len(hr_tags) > 1:
                # 使用<hr>标签分割内容
                segments = []
                current_segment = []
                
                for element in soup.body.children if soup.body else soup.children:
                    if element.name == 'hr':
                        if current_segment:
                            segments.append(current_segment)
                            current_segment = []
                    else:
                        current_segment.append(element)
                
                # 添加最后一个段落
                if current_segment:
                    segments.append(current_segment)
                
                # 处理每个段落
                for segment in segments:
                    segment_text = ' '.join([elem.get_text(strip=True) for elem in segment if elem.get_text(strip=True)])
                    if segment_text.strip():
                        topics.append({'text': segment_text})
                
                return topics
            
            # 方法2: 查找标题标签 (h1, h2, h3, ...)
            headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            if headers:
                # 使用标题作为分隔点
                for header in headers:
                    # 获取标题文本
                    title = header.get_text(strip=True)
                    
                    # 获取标题后的内容，直到下一个标题
                    content = []
                    for sibling in header.next_siblings:
                        if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            break
                        if sibling.string and sibling.string.strip():
                            content.append(sibling.get_text(strip=True))
                    
                    # 组合标题和内容
                    topic_text = title + " " + " ".join(content)
                    if len(topic_text.strip()) > 10:  # 确保话题有足够内容
                        topics.append({'text': topic_text})
            
            # 方法3: 查找分隔符（如<div>等）
            if not topics:
                divs = soup.find_all('div', recursive=False)
                if len(divs) > 1:
                    for div in divs:
                        text = div.get_text(strip=True)
                        if len(text) > 30:  # 确保有足够内容
                            topics.append({'text': text})
            
            return topics
        
        except Exception as e:
            print(f"HTML结构分割出错: {str(e)}")
            return []
    
    def _split_by_text_features(self, text):
        """基于文本特征分割话题 - 优化版"""
        if not text or len(text) < 100:  # 文本太短，不分割
            return [{'text': text}] if text else []
        
        topics = []
        
        # 方法1: 基于明显的分隔符
        separators = [
            r'\n\s*\n',  # 空行
            r'\*{3,}',   # 星号分隔线
            r'-{3,}',    # 破折号分隔线
            r'={3,}',    # 等号分隔线
            r'【.*?】',   # 中文方括号标题
            r'■.*?■',    # 方块符号标题
            r'▌.*?▌',    # 竖线符号标题
            r'\d+[\.\s、]+', # 数字编号
            r'•\s+',     # 项目符号
        ]
        
        # 尝试使用分隔符分割
        pattern = '|'.join(separators)
        segments = re.split(pattern, text)
        
        if len(segments) > 1:
            for segment in segments:
                segment = segment.strip()
                if len(segment) > 50:  # 确保片段有足够内容
                    topics.append({'text': segment})
            return topics
        
        # 方法2: 基于主题转换识别
        # 识别可能的主题转换标记词
        topic_shift_markers = [
            '另外', '此外', '同时', '值得一提的是', '据悉', '据报道', '据了解',
            '另据', '与此同时', '除此之外', '不仅如此', '更重要的是', '此外',
            '与此相关', '相关消息', '最新消息', '最新进展', '最新动态'
        ]
        
        # 使用主题转换标记词分割
        for marker in topic_shift_markers:
            if marker in text:
                segments = text.split(marker)
                if len(segments) > 1:
                    # 第一个片段
                    topics.append({'text': segments[0].strip()})
                    
                    # 其余片段，添加标记词以保持上下文
                    for i in range(1, len(segments)):
                        if segments[i].strip():
                            topics.append({'text': marker + ' ' + segments[i].strip()})
                    
                    return topics
        
        # 方法3: 基于句子长度和数量
        sentences = re.split(r'[。！？.!?]+', text)
        if len(sentences) > 5:  # 有足够多的句子
            current_topic = ""
            sentence_count = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                current_topic += sentence + "。"
                sentence_count += 1
                
                # 每3-5个句子可能形成一个话题
                if sentence_count >= 3 and len(current_topic) > 100:
                    topics.append({'text': current_topic})
                    current_topic = ""
                    sentence_count = 0
            
            # 添加最后一个话题
            if current_topic:
                topics.append({'text': current_topic})
            
            return topics
        
        # 如果所有方法都失败，将整个文本作为一个话题
        return [{'text': text}]
    
    def _extract_topic_features(self, topic):
        """提取话题特征 - 优化版"""
        text = topic.get('text', '')
        if not text:
            return None
        
        # 分词并去除停用词
        words = [w for w in jieba.cut(text) if w not in self.stopwords and len(w.strip()) > 1]
        
        # 提取关键词（基于词频）
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:15]]  # 增加关键词数量
        
        # 提取实体（简化版，基于关键词）
        entities = []
        for word in keywords:
            if len(word) >= 2 and not any(char.isdigit() for char in word):
                entities.append(word)
        
        # 提取核心句子（简化版，取前三句）
        sentences = re.split(r'[。！？.!?]+', text)
        core_sentences = [s.strip() for s in sentences[:3] if len(s.strip()) > 10]
        
        # 生成话题指纹
        fingerprint = {
            'keywords': keywords,
            'entities': entities[:8],  # 增加实体数量
            'core_sentences': core_sentences,
            'text': text
        }
        
        return fingerprint
    
    def _calculate_topic_similarity(self, topic1, topic2):
        """计算两个话题的相似度 - 优化版"""
        # 获取话题特征
        features1 = topic1.get('features', {})
        features2 = topic2.get('features', {})
        
        if not features1 or not features2:
            return 0.0
        
        # 1. 实体相似度
        entities1 = set(features1.get('entities', []))
        entities2 = set(features2.get('entities', []))
        
        if entities1 and entities2:
            entity_intersection = entities1.intersection(entities2)
            entity_union = entities1.union(entities2)
            entity_sim = len(entity_intersection) / len(entity_union) if entity_union else 0
            
            # 优化：如果有3个或以上共同实体，提高相似度权重
            if len(entity_intersection) >= 3:
                entity_sim = min(1.0, entity_sim * 1.2)
        else:
            entity_sim = 0
        
        # 2. 关键词相似度
        keywords1 = features1.get('keywords', [])
        keywords2 = features2.get('keywords', [])
        
        if keywords1 and keywords2:
            # Jaccard相似度
            kw_set1 = set(keywords1)
            kw_set2 = set(keywords2)
            kw_intersection = kw_set1.intersection(kw_set2)
            kw_union = kw_set1.union(kw_set2)
            keyword_sim = len(kw_intersection) / len(kw_union) if kw_union else 0
            
            # 优化：如果有5个或以上共同关键词，提高相似度权重
            if len(kw_intersection) >= 5:
                keyword_sim = min(1.0, keyword_sim * 1.2)
        else:
            keyword_sim = 0
        
        # 3. 文本相似度（使用TF-IDF和余弦相似度）
        text1 = features1.get('text', '')
        text2 = features2.get('text', '')
        
        if text1 and text2:
            try:
                # 使用TF-IDF向量化文本
                tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
                # 计算余弦相似度
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except:
                # 如果向量化失败，使用简单的词重叠率
                words1 = set(self._tokenize(text1).split())
                words2 = set(self._tokenize(text2).split())
                overlap = words1.intersection(words2)
                cosine_sim = len(overlap) / (len(words1) + len(words2) - len(overlap)) if (len(words1) + len(words2) - len(overlap)) > 0 else 0
        else:
            cosine_sim = 0
        
        # 综合相似度（加权平均）- 优化权重分配
        combined_sim = (entity_sim * 0.45 + keyword_sim * 0.35 + cosine_sim * 0.2)
        
        return combined_sim
    
    def _cluster_similar_topics(self, topics):
        """聚类相似话题 - 优化版"""
        if not topics or len(topics) <= 1:
            return topics
        
        # 计算话题间的相似度矩阵
        n_topics = len(topics)
        similarity_matrix = np.zeros((n_topics, n_topics))
        
        for i in range(n_topics):
            for j in range(i+1, n_topics):
                sim = self._calculate_topic_similarity(topics[i], topics[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        # 转换为距离矩阵（1 - 相似度）
        distance_matrix = 1 - similarity_matrix
        
        # 使用层次聚类
        clusters = []
        
        # 优化版聚类：使用传递性聚类
        visited = [False] * n_topics
        
        for i in range(n_topics):
            if visited[i]:
                continue
            
            # 创建新簇
            cluster = [i]
            visited[i] = True
            
            # 使用广度优先搜索找到所有相关话题
            queue = [i]
            while queue:
                current = queue.pop(0)
                
                for j in range(n_topics):
                    if not visited[j] and similarity_matrix[current, j] >= self.topic_similarity_threshold:
                        cluster.append(j)
                        visited[j] = True
                        queue.append(j)
            
            # 添加到聚类结果
            clusters.append(cluster)
        
        # 将索引转换为实际话题
        topic_clusters = []
        for cluster in clusters:
            topic_cluster = [topics[idx] for idx in cluster]
            topic_clusters.append(topic_cluster)
        
        return topic_clusters
    
    def _merge_topic_cluster(self, topic_cluster):
        """合并话题簇 - 优化版"""
        if not topic_cluster:
            return None
        
        if len(topic_cluster) == 1:
            return topic_cluster[0]
        
        # 选择基础话题（最长的版本）
        base_topic = max(topic_cluster, key=lambda t: len(t.get('features', {}).get('text', '')))
        
        # 记录所有来源
        sources = []
        for topic in topic_cluster:
            source = topic.get('source')
            if source and source not in sources:
                sources.append(source)
        
        # 合并后的话题
        merged_topic = {
            'text': base_topic.get('features', {}).get('text', ''),
            'sources': sources,
            'link': base_topic.get('link', ''),
            'published': base_topic.get('published', ''),
            'published_parsed': base_topic.get('published_parsed'),
            'features': base_topic.get('features', {}),
            'entry_id': base_topic.get('entry_id', '')
        }
        
        # 优化：尝试从其他话题提取补充信息
        base_text = merged_topic['text']
        for topic in topic_cluster:
            if topic == base_topic:
                continue
            
            topic_text = topic.get('features', {}).get('text', '')
            # 简单启发式方法：如果话题文本不完全包含在基础文本中，可能有补充信息
            if topic_text and not self._is_text_contained(topic_text, base_text):
                # 找出可能的补充信息
                supplement = self._extract_supplement(topic_text, base_text)
                if supplement:
                    merged_topic['text'] += "\n\n补充信息: " + supplement
        
        return merged_topic
    
    def _is_text_contained(self, text1, text2):
        """检查text1是否基本包含在text2中"""
        # 简化为关键词包含检查
        words1 = set(self._tokenize(text1).split())
        words2 = set(self._tokenize(text2).split())
        
        # 如果80%的关键词都包含在text2中，认为基本包含
        if len(words1) == 0:
            return True
        
        overlap_ratio = len(words1.intersection(words2)) / len(words1)
        return overlap_ratio > 0.8
    
    def _extract_supplement(self, text, base_text):
        """从text中提取相对于base_text的补充信息"""
        # 分句
        sentences1 = re.split(r'[。！？.!?]+', text)
        sentences2 = re.split(r'[。！？.!?]+', base_text)
        
        # 找出可能的补充句子
        supplements = []
        for sentence in sentences1:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 检查这个句子是否已经包含在base_text中
            contained = False
            for base_sentence in sentences2:
                if sentence in base_sentence or self._is_text_contained(sentence, base_sentence):
                    contained = True
                    break
            
            if not contained:
                supplements.append(sentence)
        
        return "。".join(supplements) if supplements else ""
    
    def process_topics(self, entries):
        """处理RSS条目，提取话题并去重"""
        # 1. 提取所有话题
        all_topics = []
        
        for entry in entries:
            # 分割条目为话题
            topics = self._split_into_topics(entry)
            
            # 提取每个话题的特征
            for topic in topics:
                features = self._extract_topic_features(topic)
                if features:
                    topic['features'] = features
                    all_topics.append(topic)
        
        print(f"从 {len(entries)} 个条目中提取出 {len(all_topics)} 个话题")
        
        # 2. 聚类相似话题
        topic_clusters = self._cluster_similar_topics(all_topics)
        
        # 3. 合并每个簇中的话题
        merged_topics = []
        for cluster in topic_clusters:
            merged_topic = self._merge_topic_cluster(cluster)
            if merged_topic:
                merged_topics.append(merged_topic)
        
        print(f"话题聚类与合并后，保留 {len(merged_topics)} 个唯一话题")
        
        return merged_topics
    
    def reconstruct_rss(self, topics):
        """重构RSS，生成新的RSS条目 - 优化版"""
        # 按来源分组话题
        topics_by_source = {}
        for topic in topics:
            sources = topic.get('sources', [])
            if not sources and topic.get('source'):
                sources = [topic.get('source')]
            
            # 使用第一个来源作为主要来源
            primary_source = sources[0] if sources else "未知来源"
            
            if primary_source not in topics_by_source:
                topics_by_source[primary_source] = []
            
            topics_by_source[primary_source].append(topic)
        
        # 为每个来源创建RSS条目
        rss_items = []
        
        for source, source_topics in topics_by_source.items():
            # 按发布时间排序（如果有）
            source_topics.sort(
                key=lambda x: time.mktime(x['published_parsed']) if x.get('published_parsed') else 0, 
                reverse=True
            )
            
            # 每3个话题组成一个RSS条目 - 优化为更小的分组
            for i in range(0, len(source_topics), 3):
                chunk = source_topics[i:i+3]
                
                # 创建条目标题
                if len(chunk) == 1:
                    title = chunk[0].get('text', '')[:50] + ('...' if len(chunk[0].get('text', '')) > 50 else '')
                else:
                    # 使用多个话题的关键词组合生成标题
                    keywords = []
                    for topic in chunk:
                        features = topic.get('features', {})
                        if features and features.get('keywords'):
                            keywords.extend(features.get('keywords')[:2])
                    
                    if keywords:
                        title = f"{source}：" + "、".join(keywords[:5])
                    else:
                        title = f"{source} 新闻汇总"
                
                # 创建条目内容
                content = ""
                for topic in chunk:
                    topic_text = topic.get('text', '')
                    if topic_text:
                        content += f"<div>{topic_text}</div><hr/>"
                
                # 使用第一个话题的链接和时间
                link = chunk[0].get('link', '')
                published = chunk[0].get('published', '')
                published_parsed = chunk[0].get('published_parsed')
                
                # 创建RSS条目
                item = {
                    'title': title,
                    'link': link,
                    'summary': content,
                    'published': published,
                    'published_parsed': published_parsed,
                    'source': source,
                    'id': hashlib.md5((link + title).encode()).hexdigest()
                }
                
                rss_items.append(item)
        
        return rss_items
    
    def generate_rss(self, items):
        """生成合并后的RSS XML"""
        # 创建RSS根元素
        rss = ET.Element("rss", version="2.0")
        
        # 创建channel元素
        channel = ET.SubElement(rss, "channel")
        
        # 添加channel基本信息
        ET.SubElement(channel, "title").text = "早报聚合（话题级别去重）"
        ET.SubElement(channel, "link").text = "https://example.com/merged_rss"
        ET.SubElement(channel, "description").text = "条目来自appd.top的早报RSS源，已进行话题级别去重"
        ET.SubElement(channel, "language").text = "zh-cn"
        ET.SubElement(channel, "lastBuildDate").text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0800")
        
        # 添加自定义标签
        generator = ET.SubElement(channel, "generator")
        generator.text = "Topic-level RSS Merger"
        
        # 添加items
        for item in items:
            rss_item = ET.SubElement(channel, "item")
            
            ET.SubElement(rss_item, "title").text = item['title']
            ET.SubElement(rss_item, "link").text = item['link']
            ET.SubElement(rss_item, "description").text = item['summary']
            
            if item.get('published'):
                ET.SubElement(rss_item, "pubDate").text = item['published']
            
            # 添加guid
            guid = ET.SubElement(rss_item, "guid", isPermaLink="false")
            guid.text = item['id']
            
            # 添加source
            source = ET.SubElement(rss_item, "source")
            source.text = item['source']
        
        # 生成漂亮的XML
        rough_string = ET.tostring(rss, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ", encoding="utf-8").decode('utf-8')
        
        return pretty_xml
    
    def save_rss(self, xml_content):
        """保存RSS到文件"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        print(f"合并后的RSS已保存到 {self.output_file}")
    
    def process(self):
        """处理全部RSS合并流程"""
        # 获取所有RSS源内容
        all_entries = self.fetch_feeds()
        
        # 话题级别处理
        topics = self.process_topics(all_entries)
        
        # 重构RSS条目
        rss_items = self.reconstruct_rss(topics)
        
        # 生成新的RSS
        merged_rss = self.generate_rss(rss_items)
        
        # 保存到文件
        self.save_rss(merged_rss)
        
        return self.output_file

# 主函数
def main():
    # 定义RSS源
    rss_feeds = [
        {"name": "爱范儿", "url": "https://appd.top/feed/ifanr"},
        {"name": "少数派", "url": "https://appd.top/feed/sspai"},
        {"name": "极客公园", "url": "https://appd.top/feed/geekpark"}
    ]
    
    # 创建合并器
    merger = TopicRSSMerger(rss_feeds, output_file="topic_merged_rss_optimized.xml")
    
    # 处理合并
    output_file = merger.process()
    
    print(f"话题级别RSS合并与去重完成，结果保存在: {output_file}")

if __name__ == "__main__":
    main()
