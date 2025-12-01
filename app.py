import streamlit as st
from models import SmartBlock
from spark_core import SparkEngine
import os

# --- 页面配置 ---
st.set_page_config(page_title="Spark v2.0", page_icon="✨", layout="wide", initial_sidebar_state="expanded")

# --- 初始化 ---
if 'engine' not in st.session_state:
    if not os.getenv("GOOGLE_API_KEY"):
        # 为了方便本地测试，如果环境变量没设，尝试读取 secrets (云端模式)
        if "GOOGLE_API_KEY" in st.secrets:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        else:
            st.error("⚠️ 未检测到 API Key！")
            st.stop()
    st.session_state.engine = SparkEngine()

if 'blocks' not in st.session_state:
    st.session_state.blocks = []

# --- 侧边栏：输入区 ---
with st.sidebar:
    st.header("📥 采集流 (Input Stream)")
    
    source_type = st.selectbox(
        "内容来源",
        ("video_snippet", "chat_log", "article_highlight"),
        format_func=lambda x: {"video_snippet": "📹 YouTube 精研 (URL)", "chat_log": "💬 群聊清洗 (Text)", "article_highlight": "📝 阅读模式 (Text)"}[x]
    )
    
    # === 动态输入界面 ===
    url_input = ""
    start_min = 0
    end_min = 0
    raw_text = ""
    
    if source_type == "video_snippet":
        st.info("🔗 支持 YouTube 链接自动抓取")
        url_input = st.text_input("YouTube 链接", placeholder="https://www.youtube.com/watch?v=...")
        
        use_time_range = st.checkbox("启用精研模式 (指定时间段)", value=False)
        if use_time_range:
            col_t1, col_t2 = st.columns(2)
            with col_t1: start_min = st.number_input("开始 (分钟)", min_value=0, value=0)
            with col_t2: end_min = st.number_input("结束 (分钟)", min_value=1, value=10)
    else:
        # 其他模式保持粘贴文本
        placeholder = "粘贴群聊记录..." if source_type == "chat_log" else "粘贴文章内容..."
        raw_text = st.text_area("原始文本", height=300, placeholder=placeholder)
    
    # === 提交按钮 ===
    if st.button("✨ Spark It!", type="primary"):
        process_flag = False
        meta_data = {}
        content_payload = ""

        # 校验输入
        if source_type == "video_snippet":
            if not url_input:
                st.warning("请输入 YouTube 链接")
            else:
                process_flag = True
                content_payload = "Waiting for fetch..." # 占位符
                meta_data = {"url": url_input}
                if use_time_range:
                    meta_data.update({"start_min": start_min, "end_min": end_min})
        else:
            if not raw_text:
                st.warning("请输入内容")
            else:
                process_flag = True
                content_payload = raw_text

        if process_flag:
            with st.spinner("AI 正在抓取字幕、阅读、清洗、关联..."):
                # 创建块
                new_block = SmartBlock(source_type=source_type, raw_content=content_payload, metadata=meta_data)
                # 处理
                st.session_state.engine.process_block(new_block)
                # 更新
                st.session_state.blocks.insert(0, new_block)
                st.success("完成！")

# --- 主界面 ---
st.title("✨ Spark v2.0 知识内化引擎")
st.markdown("---")

if not st.session_state.blocks:
    st.info("👈 请在左侧输入 YouTube 链接或群聊记录，开始体验。")

for block in st.session_state.blocks:
    col1, col2 = st.columns([7, 3])
    
    with col1:
        # 标题处理
        title = "未命名片段"
        if block.processed_content and not block.processed_content.startswith("❌"):
            lines = block.processed_content.splitlines()
            for line in lines:
                if line.strip().startswith("#"):
                    title = line.strip("# ")
                    break
        
        st.markdown(f"### {title}")
        
        # 元数据显示
        src_label = block.source_type
        if block.source_type == "video_snippet":
            url = block.metadata.get('url', '')
            src_label = f"[YouTube]({url})"
        
        st.caption(f"ID: {block.id[:6]} | 来源: {src_label} | 📅 {block.created_at.strftime('%H:%M')}")
        
        # 标签
        if block.ai_tags:
            st.markdown(" ".join([f"`{t}`" for t in block.ai_tags]))
        
        # 内容
        with st.expander("📖 深度阅读", expanded=True):
            st.markdown(block.processed_content)

    with col2:
        st.markdown("#### 🔗 关联实验室")
        related = st.session_state.engine.find_related(block)
        if related:
            for r_block, score in related:
                with st.container(border=True):
                    st.markdown(f"**关联度: {score:.0%}**")
                    st.caption(f"ID: {r_block.id[:6]}")
                    preview = r_block.processed_content[:40] + "..." if r_block.processed_content else ""
                    st.markdown(f"_{preview}_")
        else:
            st.markdown("*暂无强关联*")
    
    st.markdown("---")