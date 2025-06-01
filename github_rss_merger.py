#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import feedparser
import jieba
import json
import os
import time
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import xml.etree.ElementTree as ET
from xml.dom import minidom
import hashlib

class RSSMerger:
    def __init__(self, rss_feeds, output_file="merged_rss.xml"):
        """
        初始化RSS合并器
        
        Args:
            rss_feeds: RSS源列表，每项包含name和url
            output_file: 输出的合并RSS文件路径
        """
        self.rss_feeds = rss_feeds
        self.output_file = output_file
        
        # 初始化TF-IDF向量化器
        self.vectorizer = TfidfVectorizer()
        
        # 设置相似度阈值
        self.tfidf_threshold = 0.6
        self.jaccard_threshold = 0.5
        
        # 设置时间窗口（天）
        self.time_window = 3
    
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
        
        # 标准化条目
        standardized = {
            'title': entry.get('title', ''),
            'link': entry.get('link', ''),
            'summary': entry.get('summary', ''),
            'published': published,
            'published_parsed': published_parsed,
            'source': source,
            # 生成唯一ID
            'id': hashlib.md5(entry.get('link', '').encode()).hexdigest()
        }
        
        return standardized
    
    def _tokenize(self, text):
        """对文本进行中文分词"""
        return " ".join(jieba.cut(text))
    
    def _calculate_tfidf_similarity(self, entries):
        """计算条目间的TF-IDF余弦相似度"""
        # 提取标题
        titles = [entry['title'] for entry in entries]
        
        # 对标题进行分词
        tokenized_titles = [self._tokenize(title) for title in titles]
        
        # 计算TF-IDF向量
        tfidf_matrix = self.vectorizer.fit_transform(tokenized_titles)
        
        # 计算余弦相似度矩阵
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        return cosine_sim_matrix
    
    def _calculate_jaccard_similarity(self, title1, title2):
        """计算两个标题的Jaccard相似度"""
        # 对标题进行分词
        words1 = set(jieba.cut(title1))
        words2 = set(jieba.cut(title2))
        
        # 计算Jaccard相似度
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _is_within_time_window(self, entry1, entry2):
        """判断两个条目的发布时间是否在指定时间窗口内"""
        # 如果没有解析过的时间，则认为在时间窗口内
        if not entry1.get('published_parsed') or not entry2.get('published_parsed'):
            return True
        
        time1 = entry1['published_parsed']
        time2 = entry2['published_parsed']
        
        # 计算时间差（天）
        time_diff = abs(time.mktime(time1) - time.mktime(time2)) / (24 * 3600)
        
        return time_diff <= self.time_window
    
    def deduplicate(self, entries):
        """对RSS条目进行去重"""
        if not entries:
            return []
        
        print(f"开始对 {len(entries)} 条内容进行去重...")
        
        # 按发布时间排序（如果有）
        entries.sort(key=lambda x: time.mktime(x['published_parsed']) if x.get('published_parsed') else 0, reverse=True)
        
        # 计算TF-IDF相似度矩阵
        cosine_sim_matrix = self._calculate_tfidf_similarity(entries)
        
        # 去重结果
        unique_entries = []
        duplicate_indices = set()
        
        # 记录重复项信息
        duplicate_info = {}
        
        # 去重过程
        for i in range(len(entries)):
            if i in duplicate_indices:
                continue
            
            current_entry = entries[i]
            current_duplicates = []
            
            for j in range(len(entries)):
                if i == j or j in duplicate_indices:
                    continue
                
                # 只比较时间窗口内的条目
                if not self._is_within_time_window(current_entry, entries[j]):
                    continue
                
                # 检查链接是否完全相同
                if current_entry['link'] == entries[j]['link']:
                    duplicate_indices.add(j)
                    current_duplicates.append(j)
                    continue
                
                # 检查TF-IDF相似度
                similarity = cosine_sim_matrix[i, j]
                
                # 如果TF-IDF相似度高于阈值，判定为重复
                if similarity >= self.tfidf_threshold:
                    duplicate_indices.add(j)
                    current_duplicates.append(j)
                # 如果TF-IDF相似度在边界值，使用Jaccard相似度进一步判断
                elif similarity >= 0.4:
                    jaccard_sim = self._calculate_jaccard_similarity(
                        current_entry['title'], entries[j]['title']
                    )
                    if jaccard_sim >= self.jaccard_threshold:
                        duplicate_indices.add(j)
                        current_duplicates.append(j)
            
            # 记录重复信息
            if current_duplicates:
                duplicate_info[i] = current_duplicates
            
            # 添加到唯一条目列表
            unique_entries.append(current_entry)
        
        # 打印去重结果
        print(f"去重完成，从 {len(entries)} 条内容中保留了 {len(unique_entries)} 条唯一内容")
        if duplicate_info:
            print(f"发现 {len(duplicate_info)} 组重复内容")
            for idx, duplicates in duplicate_info.items():
                print(f"条目 '{entries[idx]['title']}' 与以下条目重复:")
                for dup_idx in duplicates:
                    print(f"  - '{entries[dup_idx]['title']}' (来源: {entries[dup_idx]['source']})")
        
        return unique_entries
    
    def generate_rss(self, entries):
        """生成合并后的RSS XML"""
        # 创建RSS根元素
        rss = ET.Element("rss", version="2.0")
        
        # 创建channel元素
        channel = ET.SubElement(rss, "channel")
        
        # 添加channel基本信息
        ET.SubElement(channel, "title").text = "科技新闻聚合"
        ET.SubElement(channel, "link").text = "https://example.com/merged_rss"
        ET.SubElement(channel, "description").text = "爱范儿、少数派、极客公园内容聚合，已去除重复"
        ET.SubElement(channel, "language").text = "zh-cn"
        ET.SubElement(channel, "lastBuildDate").text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0800")
        
        # 添加自定义标签
        generator = ET.SubElement(channel, "generator")
        generator.text = "RSS Merger"
        
        # 添加items
        for entry in entries:
            item = ET.SubElement(channel, "item")
            
            ET.SubElement(item, "title").text = entry['title']
            ET.SubElement(item, "link").text = entry['link']
            ET.SubElement(item, "description").text = entry['summary']
            
            if entry.get('published'):
                ET.SubElement(item, "pubDate").text = entry['published']
            
            # 添加guid
            guid = ET.SubElement(item, "guid", isPermaLink="false")
            guid.text = entry['id']
            
            # 添加source
            source = ET.SubElement(item, "source")
            source.text = entry['source']
        
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
        
        # 去重
        unique_entries = self.deduplicate(all_entries)
        
        # 生成新的RSS
        merged_rss = self.generate_rss(unique_entries)
        
        # 保存到文件
        self.save_rss(merged_rss)
        
        return self.output_file

# 主函数
def main():
    # 定义RSS源
    rss_feeds = [
        {"name": "Readhub早报", "url": "https://appd.top/feed/readhub"},
        {"name": "8点1氪", "url": "https://appd.top/feed/kr36"},
        {"name": "科技昨夜今晨", "url": "https://appd.top/feed/ithome"},
        {"name": "ifanr早报", "url": "https://appd.top/feed/ifanr"},
        {"name": "派早报", "url": "https://appd.top/feed/sspai"},
        {"name": "极客早知道", "url": "https://appd.top/feed/geekpark"}
    ]
    
    # 创建合并器
    merger = RSSMerger(rss_feeds, output_file="merged_rss.xml")
    
    # 处理合并
    output_file = merger.process()
    
    print(f"RSS合并与去重完成，结果保存在: {output_file}")

if __name__ == "__main__":
    main()
