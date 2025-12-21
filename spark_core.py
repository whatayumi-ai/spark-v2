import os
import json
import time
import google.generativeai as genai
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models import SmartBlock
import prompts

# --- 库导入: 只保留字幕库，移除 yt_dlp ---
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None

# 配置 API KEY
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class SparkEngine:
    def __init__(self):
        self.database: List[SmartBlock] = []
        # 保留修复: 使用最新的 2.5 版本
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def _call_llm(self, prompt):
        """调用大模型"""
        # 保留修复: 强制休息 2 秒，防止 429 报错
        print("⏳ 正在等待 API 冷却 (2s)...")
        time.sleep(2) 
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error processing AI: {e}"

    def _get_youtube_transcript(self, url):
        """只抓取字幕"""
        if not YouTubeTranscriptApi:
            return None, "❌ 未安装 transcript 库"
            
        try:
            video_id = None
            if "v=" in url:
                video_id = url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                video_id = url.split("youtu.be/")[1].split("?")[0]
            
            if not video_id:
                return None, "无法解析 Video ID"

            # 尝试抓取多语言字幕
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['zh-CN', 'zh-Hans', 'zh-Hant', 'en'])
            full_text = " ".join([t['text'] for t in transcript_list])
            return f"[自动抓取的字幕] {full_text}", None
            
        except Exception as e:
            return None, str(e)

    def _get_embedding(self, text):
        try:
            time.sleep(1)
            truncated_text = text[:9000]
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=truncated_text,
                task_type="retrieval_document", 
                title="Spark Block" 
            )
            return result['embedding']
        except Exception as e:
            return []

    def process_block(self, block: SmartBlock, file_bytes=None):
        print(f"🔄 [Gemini] 正在处理: {block.source_type} ...")
        
        prompt_text = block.raw_content
        status_msg = ""

        # === 核心逻辑: 只处理字幕 ===
        if block.source_type == "video_snippet" and ("youtube.com" in block.raw_content or "youtu.be" in block.raw_content):
            transcript_text, error = self._get_youtube_transcript(block.raw_content)
            
            if transcript_text:
                print("✅ 成功抓取字幕")
                prompt_text = transcript_text
                status_msg = "(基于CC字幕)"
            else:
                # 如果没有字幕，直接报错，不再尝试下载音频
                print(f"❌ 字幕获取失败: {error}")
                block.processed_content = f"❌ 此视频没有CC字幕，且音频下载功能已关闭。\n错误信息: {error}"
                return

        # === 准备 Prompt ===
        if block.source_type == "video_snippet":
            # 简化 Prompt，不再需要处理音频的逻辑
            final_prompt = prompts.VIDEO_PROCESS_PROMPT.format(text=prompt_text)
        elif block.source_type == "chat_log":
            final_prompt = prompts.CHAT_PROCESS_PROMPT.format(text=prompt_text)
        else:
            final_prompt = prompt_text
        
        # --- 保留修复: 合并 Prompt (总结+标签 一次搞定) ---
        combined_prompt = final_prompt + "\n\n" + "-"*20 + "\n【附加任务】在笔记的最后，请务必另起一行，以 JSON 格式输出 3-5 个核心标签，格式严格如下：\nTagsJSON: [\"#标签1\", \"#标签2\", \"#标签3\"]"

        # 1. 调用 LLM
        full_response = self._call_llm(combined_prompt)
        
        # 2. 解析结果
        if full_response and "TagsJSON:" in full_response:
            try:
                parts = full_response.split("TagsJSON:")
                content_part = parts[0].strip()
                tags_json_str = parts[1].strip().replace("```json", "").replace("```", "").strip()
                
                block.processed_content = f"{status_msg}\n\n{content_part}"
                block.ai_tags = json.loads(tags_json_str)
            except:
                block.processed_content = f"{status_msg}\n\n{full_response}"
                block.ai_tags = ["#TagParseError"]
        else:
            block.processed_content = f"{status_msg}\n\n{full_response}"
            block.ai_tags = []

        # 3. Embedding
        if block.processed_content and "Error" not in block.processed_content:
            block.embedding = self._get_embedding(block.processed_content)
            self.database.append(block)
            print(f"✅ 处理完成: ID {block.id[:6]}")

    def find_related(self, target_block: SmartBlock, top_k=3):
        # ... (这部分保持不变) ...
        if not target_block.embedding or not self.database:
            return []
        db_embeddings = [b.embedding for b in self.database if b.id != target_block.id and b.embedding]
        db_blocks = [b for b in self.database if b.id != target_block.id and b.embedding]
        if not db_embeddings:
            return []
        target_vec = np.array(target_block.embedding).reshape(1, -1)
        db_matrix = np.array(db_embeddings)
        similarities = cosine_similarity(target_vec, db_matrix)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score > 0.3:
                results.append((db_blocks[idx], score))
        return results