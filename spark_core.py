import os
import json
import numpy as np
import google.generativeai as genai
from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from models import SmartBlock
import prompts

# 配置 API KEY
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class SparkEngine:
    def __init__(self):
        self.database: List[SmartBlock] = []
        # 使用你刚才验证通过的最强模型
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def _extract_video_id(self, url):
        """从 YouTube URL 中提取 Video ID"""
        query = urlparse(url)
        if query.hostname == 'youtu.be':
            return query.path[1:]
        if query.hostname in ('www.youtube.com', 'youtube.com'):
            if query.path == '/watch':
                p = parse_qs(query.query)
                return p['v'][0]
            if query.path[:7] == '/embed/':
                return query.path.split('/')[2]
            if query.path[:3] == '/v/':
                return query.path.split('/')[2]
        return None

    def _fetch_transcript(self, url, start_min=None, end_min=None):
        """抓取字幕并根据时间过滤"""
        try:
            video_id = self._extract_video_id(url)
            if not video_id:
                return "Error: 无效的 YouTube 链接"
            
            # 获取字幕列表 (自动尝试中文和英文)
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['zh-Hans', 'zh-Hant', 'en'])
            
            full_text = []
            for item in transcript_list:
                start_time = item['start']
                text = item['text']
                
                # 如果指定了时间范围 (精研模式)
                if start_min is not None and end_min is not None:
                    if start_time < start_min * 60: continue
                    if start_time > end_min * 60: break
                
                full_text.append(text)
            
            return " ".join(full_text)
        except Exception as e:
            return f"字幕抓取失败 (可能该视频没有CC字幕): {str(e)}"

    def _call_llm(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error processing AI: {e}"

    def _get_embedding(self, text):
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document", 
                title="Spark Block" 
            )
            return result['embedding']
        except Exception as e:
            print(f"Embedding Error: {e}")
            return []

    def process_block(self, block: SmartBlock):
        print(f"🔄 [Gemini] 正在处理: {block.source_type} ...")
        
        # --- 核心改动：如果是视频，先抓取内容 ---
        content_to_process = block.raw_content
        
        if block.source_type == "video_snippet":
            # 检查 metadata 里有没有 URL
            url = block.metadata.get('url')
            if url:
                print(f"📺 正在抓取 YouTube 字幕: {url}")
                # 获取时间范围设置
                s_min = block.metadata.get('start_min')
                e_min = block.metadata.get('end_min')
                
                # 抓取字幕覆盖掉原始的 raw_content
                fetched_text = self._fetch_transcript(url, s_min, e_min)
                if "Error" in fetched_text or "失败" in fetched_text:
                    block.processed_content = f"❌ {fetched_text}"
                    return # 终止处理
                
                content_to_process = fetched_text
                # 把抓到的文字存回去，方便查看
                block.raw_content = f"[已提取字幕] {url}\n\n{fetched_text[:200]}..."

        # 1. 文本整形
        if block.source_type == "video_snippet":
            final_prompt = prompts.VIDEO_PROCESS_PROMPT.format(text=content_to_process)
        elif block.source_type == "chat_log":
            final_prompt = prompts.CHAT_PROCESS_PROMPT.format(text=content_to_process)
        else:
            final_prompt = content_to_process
            
        block.processed_content = self._call_llm(final_prompt)
        
        # 2. 自动打标
        tag_prompt = prompts.TAGGING_PROMPT.format(content=block.processed_content)
        tags_raw = self._call_llm(tag_prompt)
        try:
            clean_json = tags_raw.replace("```json", "").replace("```", "").strip()
            block.ai_tags = json.loads(clean_json)
        except:
            block.ai_tags = ["#AI_Tag_Error"]

        # 3. 向量化
        if block.processed_content:
            block.embedding = self._get_embedding(block.processed_content)
            
        self.database.append(block)
        print(f"✅ 处理完成: ID {block.id[:6]}")

    def find_related(self, target_block: SmartBlock, top_k=3):
        # ... (这部分代码保持不变) ...
        if not target_block.embedding or not self.database:
            return []
        db_embeddings = [b.embedding for b in self.database if b.id != target_block.id and b.embedding]
        db_blocks = [b for b in self.database if b.id != target_block.id and b.embedding]
        if not db_embeddings: return []
        target_vec = np.array(target_block.embedding).reshape(1, -1)
        db_matrix = np.array(db_embeddings)
        similarities = cosine_similarity(target_vec, db_matrix)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score > 0.3: results.append((db_blocks[idx], score))
        return results