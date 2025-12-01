from models import SmartBlock
from spark_core import SparkEngine

def main():
    engine = SparkEngine()

    # --- 模拟输入数据 ---
    
    # 1. 模拟 YouTube 视频 (来自李厚辰视频的 Whisper 原始转录片段，模拟口语化)
    video_raw = """
    这个这个转型问题对大家其实都不是一个陌生问题了... 
    因为中国是一个这个极权国家，这个极权国家呢，而且是一个相对样表一人极权的这个体制...
    所以很多人会认为呢，中国的事都是一个帝王意志的问题...
    """
    video_block = SmartBlock(
        source_type="video_snippet", 
        raw_content=video_raw,
        metadata={"url": "youtube.com/watch?v=2zUak31UmZ0", "time_range": "00:00-05:00"}
    )

    # 2. 模拟群聊记录 (来自我们刚才测试的文本)
    chat_raw = """
    郑鹏（Pen）: 刚刚在想“名实分离”的问题...
    傻蛋: 因为中国的名实分裂完全是给上位者卸责的...
    雪绒鹅岛: 一个很明显的例子，是孟子说...韩非变成了...
    """
    chat_block = SmartBlock(
        source_type="chat_log",
        raw_content=chat_raw
    )

    # --- Step 1: 运行 AI 引擎处理 ---
    print("🚀 启动 Spark v2.0 引擎...")
    
    engine.process_block(video_block)
    engine.process_block(chat_block)

    # --- Step 2: 展示处理结果 (模拟前端渲染) ---
    print("\n" + "="*50)
    print("📄 [视频] 精研模式输出:")
    print("="*50)
    print(f"🏷️ 标签: {video_block.ai_tags}")
    print("-" * 20)
    print(video_block.processed_content[:500] + "...\n(略)") # 只显示前500字

    print("\n" + "="*50)
    print("💬 [群聊] 清洗模式输出:")
    print("="*50)
    print(f"🏷️ 标签: {chat_block.ai_tags}")
    print("-" * 20)
    print(chat_block.processed_content[:500] + "...\n(略)")

    # --- Step 3: 测试 Phase 2 用户自定义标签 ---
    print("\n🔧 测试用户手动打标...")
    video_block.user_tags.append("#Project-Politics")
    print(f"更新后的视频标签: {video_block.ai_tags + video_block.user_tags}")

    # --- Step 4: 测试 Auto-Linking (关联) ---
    print("\n🔗 正在计算关联...")
    # 查找与“群聊记录”最相关的“视频片段”
    related_items = engine.find_related(chat_block)
    
    if related_items:
        print(f"发现 {len(related_items)} 个关联内容:")
        for block, score in related_items:
            print(f"   -> 关联度 {score:.4f}: {block.source_type} (ID: {block.id[:6]})")
            print(f"      共性猜测: {set(block.ai_tags) & set(chat_block.ai_tags)} 等")
    else:
        print("暂无强关联内容。")

if __name__ == "__main__":
    main()