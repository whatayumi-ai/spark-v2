import os
import json
import time
import glob
import google.generativeai as genai
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models import SmartBlock
import prompts

# --- 库导入检查 ---
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

# 配置 API KEY
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class SparkEngine:
    def __init__(self):
        self.database: List[SmartBlock] = []
        # 使用 Gemini 2.5 Flash (目前最稳)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def _call_llm(self, prompt, audio_file=None):
        """调用大模型，增加强制休眠以防止 429 报错"""
        # --- 优化点 2: 强制休息 2 秒 ---
        print("⏳ 正在等待 API 冷却 (2s)...")
        time.sleep(2) 
        
        try:
            content_parts = [prompt]
            if audio_file:
                content_parts.append(audio_file)
            
            response = self.model.generate_content(content_parts)
            return response.text
        except Exception as e:
            return f"Error processing AI: {e}"

    def _download_youtube_audio(self, url):
        """Plan B: 使用 yt-dlp 下载音频"""
        if not yt_dlp:
            return None, "❌ 未安装 yt-dlp 库"
            
        print(f"正在尝试下载音频: {url} ...")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '128',
            }],
            'outtmpl': '/tmp/%(id)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_id = info['id']
                files = glob.glob(f"/tmp/{video_id}.mp3")
                if files:
                    return files[0], None
                else:
                    return None, "❌ 下载显示完成，但在文件夹里找不到文件"
        except Exception as e:
            return None, f"❌ 音频下载失败: {e}"

    def _get_youtube_transcript(self, url):
        """Plan A: 抓取字幕"""
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

            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['zh-CN', 'zh-Hans', 'zh-Hant', 'en'])
            full_text = " ".join([t['text'] for t in transcript_list])
            return f"[自动抓取的字幕] {full_text}", None
            
        except Exception as e:
            return None, str(e)

    def _upload_audio(self, file_path_or_bytes, mime_type="audio/mp3"):
        """上传音频"""
        try:
            print("正在上传音频到 Gemini...")
            
            if isinstance(file_path_or_bytes, str):
                uploaded_file = genai.upload_file(file_path_or_bytes, mime_type=mime_type)
            else:
                import tempfile
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tfile.write(file_path_or_bytes)
                tfile.close()
                uploaded_file = genai.upload_file(tfile.name, mime_type=mime_type)
            
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_file = genai.get_file(uploaded_file.name)
            
            return uploaded_file
        except Exception as e:
            print(f"Upload Error: {e}")
            return None

    def _get_embedding(self, text):
        try:
            # Embedding 调用也加个小延迟，更稳
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
        
        audio_resource = None
        prompt_text = block.raw_content
        status_msg = ""

        # === 核心逻辑分支 ===
        if file_bytes:
            audio_resource = self._upload_audio(file_bytes)
            if not audio_resource:
                block.processed_content = "❌ 用户音频上传失败"
                return

        elif block.source_type == "video_snippet" and ("youtube.com" in block.raw_content or "youtu.be" in block.raw_content):
            transcript_text, error = self._get_youtube_transcript(block.raw_content)
            
            if transcript_text:
                print("✅ 成功抓取字幕")
                prompt_text = transcript_text
                status_msg = "(基于CC字幕)"
            else:
                print(f"⚠️ 启动 Plan B 音频下载... (原因: {error})")
                block.processed_content = "⚠️ 正在下载视频音频(Plan B)，这可能需要 1-2 分钟，请耐心等待..."
                
                mp3_path, dl_error = self._download_youtube_audio(block.raw_content)
                
                if mp3_path:
                    print("✅ 音频下载成功，上传给 AI...")
                    audio_resource = self._upload_audio(mp3_path)
                    if not audio_resource:
                        block.processed_content = "❌ 下载成功但上传 AI 失败"
                        return
                    status_msg = "(基于AI听写 - Plan B)"
                else:
                    block.processed_content = f"❌ 字幕抓取失败，且音频下载也失败: {dl_error}"
                    return

        # === 准备 Prompt ===
        if block.source_type == "video_snippet":
            base_prompt = prompts.VIDEO_PROCESS_PROMPT if not audio_resource else "请认真听这段音频，整理出详细的笔记。忽略口语废话，保留核心观点，按 Markdown 格式输出。"
            final_prompt = base_prompt.format(text=prompt_text) if not audio_resource else base_prompt
        elif block.source_type == "chat_log":
            final_prompt = prompts.CHAT_PROCESS_PROMPT.format(text=prompt_text)
        else:
            final_prompt = prompt_text
        
        # --- 优化 1: 合并 Prompt (总结+标签 一次搞定) ---
        # 拼接指令：让 AI 在总结最后，单独输出 JSON 格式的标签
        combined_prompt = final_prompt + "\n\n" + "-"*20 + "\n【附加任务】在笔记的最后，请务必另起一行，以 JSON 格式输出 3-5 个核心标签（用于分类），格式严格如下：\nTagsJSON: [\"#标签1\", \"#标签2\", \"#标签3\"]"

        # 1. 调用 LLM (总结 + 标签) - 减少一次调用
        full_response = self._call_llm(combined_prompt, audio_file=audio_resource)
        
        # 2. 解析结果
        if full_response and "TagsJSON:" in full_response:
            try:
                parts = full_response.split("TagsJSON:")
                content_part = parts[0].strip()
                tags_json_str = parts[1].strip().replace("```json", "").replace("```", "").strip()
                
                block.processed_content = f"{status_msg}\n\n{content_part}"
                block.ai_tags = json.loads(tags_json_str)
            except:
                # 如果解析失败，保留全文，给个错误标签
                block.processed_content = f"{status_msg}\n\n{full_response}"
                block.ai_tags = ["#TagParseError"]
        else:
            block.processed_content = f"{status_msg}\n\n{full_response}"
            block.ai_tags = []

        # 3. Embedding (必须保留，用于搜索，但现在总共只调 2 次 API)
        if block.processed_content and "Error" not in block.processed_content:
            block.embedding = self._get_embedding(block.processed_content)
            self.database.append(block)
            print(f"✅ 处理完成: ID {block.id[:6]}")

    def find_related(self, target_block: SmartBlock, top_k=3):
        """关联实验室核心算法"""
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