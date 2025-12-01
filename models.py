import uuid
from datetime import datetime
from typing import List, Dict, Optional

class SmartBlock:
    def __init__(self, source_type: str, raw_content: str, metadata: Dict = None):
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        
        # source_type: "video_snippet", "chat_log", "article_highlight"
        self.source_type = source_type 
        
        # 原始数据 (URL, 原始聊天记录, 原始OCR文本)
        self.raw_content = raw_content
        
        # 元数据 (时间戳范围, 发言人列表, 书名等)
        self.metadata = metadata or {}
        
        # AI 处理后的结构化内容 (即 "阅读级文本块")
        self.processed_content: Optional[str] = None
        
        # 标签系统 (Phase 2 需求: 混合标签)
        self.ai_tags: List[str] = []      # AI 自动生成的
        self.user_tags: List[str] = []    # 用户手动打的
        
        # 向量嵌入 (用于语义关联)
        self.embedding: List[float] = []

    def __repr__(self):
        return f"<Block {self.id[:6]}: {self.source_type} | Tags: {self.ai_tags + self.user_tags}>"