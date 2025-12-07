import os
import json
import time
import glob
import google.generativeai as genai
from typing import List
from models import SmartBlock
import prompts

# --- 库导入检查 (防止本地没装报错) ---
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
        # 使用 Gemini 2.0 Flash (目前最快且免费)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def _call_llm(self, prompt, audio_file=None):
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
        
        # 配置下载选项 (只下载音频，最低音质以加快速度，保存到临时文件夹)
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '128',
            }],
            'outtmpl': '/tmp/%(id)s.%(ext)s', # Linux/Mac 临时目录
            'quiet': True,
            'no_warnings': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_id = info['id']
                # 找到下载好的文件
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

            # 尝试抓取中文或英文字幕
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['zh-CN', 'zh-Hans', 'zh-Hant', 'en'])
            full_text = " ".join([t['text'] for t in transcript_list])
            return f"[自动抓取的字幕] {full_text}", None
            
        except Exception as e:
            return None, str(e)

    def _upload_audio(self, file_path_or_bytes, mime_type="audio/mp3"):
        """上传音频 (支持文件路径 或 二进制流)"""
        try:
            print("正在上传音频到 Gemini...")
            
            if isinstance(file_path_or_bytes, str):
                # 如果是文件路径 (YouTube 下载的 /tmp/xxx.mp3)
                uploaded_file = genai.upload_file(file_path_or_bytes, mime_type=mime_type)
            else:
                # 如果是二进制流 (前端用户拖拽上传的)
                import tempfile
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tfile.write(file_path_or_bytes)
                tfile.close()
                uploaded_file = genai.upload_file(tfile.name, mime_type=mime_type)
            
            # 等待 Google 处理
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_file = genai.get_file(uploaded_file.name)
            
            return uploaded_file
        except Exception as e:
            print(f"Upload Error: {e}")
            return None

    def _get_embedding(self, text):
        try:
            # 截断过长的文本以防 embedding 报错
            truncated_text = text[:9000]
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=truncated_text,
                task_type="retrieval_document", 
                title="Spark Block" 
            )
            return result['embedding']
        except Exception as e:
            # print(f"Embedding Error: {e}") # 忽略非关键报错
            return []

    def process_block(self, block: SmartBlock, file_bytes=None):
        print(f"🔄 [Gemini] 正在处理: {block.source_type} ...")
        
        audio_resource = None
        prompt_text = block.raw_content
        status_msg = ""

        # === 核心逻辑分支 ===
        
        # 1. 优先：用户直接上传了音频文件
        if file_bytes:
            audio_resource = self._upload_audio(file_bytes)
            if not audio_resource:
                block.processed_content = "❌ 用户音频上传失败"
                return

        # 2. 其次：处理 YouTube 链接
        elif block.source_type == "video_snippet" and ("youtube.com" in block.raw_content or "youtu.be" in block.raw_content):
            # >> Plan A: 试着抓字幕
            transcript_text, error = self._get_youtube_transcript(block.raw_content)
            
            if transcript_text:
                print("✅ 成功抓取字幕")
                prompt_text = transcript_text
                status_msg = "(基于CC字幕)"
            else:
                # >> Plan B: 字幕失败，启动下载音频
                print(f"⚠️ 字幕抓取失败，启动 Plan B 音频下载... (原因: {error})")
                
                # 在前端显示的临时占位符
                block.processed_content = "⚠️ 正在下载视频音频，这可能需要 1-2 分钟，请耐心等待..."
                
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
            # 如果有音频资源，提示词改一下
            base_prompt = prompts.VIDEO_PROCESS_PROMPT if not audio_resource else "请认真听这段音频，整理出详细的笔记。忽略口语废话，保留核心观点，按 Markdown 格式输出。"
            final_prompt = base_prompt.format(text=prompt_text) if not audio_resource else base_prompt
        elif block.source_type == "chat_log":
            final_prompt = prompts.CHAT_PROCESS_PROMPT.format(text=prompt_text)
        else:
            final_prompt = prompt_text
            
        # === 调用 AI ===
        result = self._call_llm(final_prompt, audio_file=audio_resource)
        
        # 加上来源标记
        block.processed_content = f"{status_msg}\n\n{result}"
        
        # === 后处理 (打标/存储) ===
        if block.processed_content and "Error" not in block.processed_content:
            try:
                tag_prompt = prompts.TAGGING_PROMPT.format(content=block.processed_content)
                tags_raw = self._call_llm(tag_prompt)
                clean_json = tags_raw.replace("```json", "").replace("```", "").strip()
                block.ai_tags = json.loads(clean_json)
            except:
                block.ai_tags = ["#AI_Tag_Error"]

            block.embedding = self._get_embedding(block.processed_content)
            self.database.append(block)
            print(f"✅ 处理完成: ID {block.id[:6]}")