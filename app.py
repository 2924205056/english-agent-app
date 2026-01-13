import streamlit as st
import io
import re
import zipfile
import math
import chardet

# NLP Imports
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag

# DOCX Import
from docx import Document

# =========== [新增] 引入 OpenAI 库 ===========
from openai import OpenAI
# ============================================

# Optional Spacy
try:
    import spacy
    _HAS_SPACY = True
except ImportError:
    _HAS_SPACY = False

# ------------------ 页面配置 ------------------
st.set_page_config(page_title="单词提取器 & AI 助教", page_icon="📘", layout="wide")

# ------------------ 缓存资源加载 ------------------
@st.cache_resource
def download_nltk_resources():
    """静默下载 NLTK 资源"""
    resources = ["punkt", "averaged_perceptron_tagger", "wordnet", "omw-1.4", "stopwords"]
    for r in resources:
        try:
            nltk.data.find(f'tokenizers/{r}')
        except LookupError:
            nltk.download(r, quiet=True)
        except ValueError:
            nltk.download(r, quiet=True)

@st.cache_resource
def load_spacy_model():
    if _HAS_SPACY:
        try:
            return spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except Exception:
            return None
    return None

# 初始化资源
download_nltk_resources()
nlp_spacy = load_spacy_model()

# ------------------ 核心工具函数 (保持原样) ------------------

def extract_text_from_bytes(file_obj, filename):
    """从内存文件对象中提取文本"""
    ext = filename.split('.')[-1].lower()
    text = ""
    try:
        if ext == 'docx':
            doc = Document(file_obj)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n".join(paragraphs)
        else:
            raw = file_obj.read()
            enc = chardet.detect(raw).get('encoding') or 'utf-8'
            text = raw.decode(enc, errors='ignore')
    except Exception as e:
        st.warning(f"⚠️ 读取 {filename} 失败: {e}")
        return ""

    if ext == 'srt': return extract_english_from_srt(text)
    elif ext == 'ass': return extract_english_from_ass(text)
    elif ext == 'vtt': return extract_english_from_vtt(text)
    else: return text

def extract_english_from_srt(text):
    lines = []
    SRT_TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}")
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.isdigit() or SRT_TIME_RE.match(s): continue
        s = re.sub(r"<.*?>", "", s)
        s = re.sub(r"\[.*?\]", "", s)
        parts = re.findall(r"[A-Za-z0-9'\",.?!:;()\- ]+", s)
        if parts: lines.append("".join(parts).strip())
    return " ".join(lines)

def extract_english_from_ass(text):
    lines = []
    for ln in text.splitlines():
        if ln.startswith("Dialogue:"):
            parts = ln.split(",", 9)
            if len(parts) >= 10:
                t = re.sub(r"\{.*?\}", "", parts[-1])
                t = re.sub(r"<.*?>", "", t)
                parts2 = re.findall(r"[A-Za-z0-9'\",.?!:;()\- ]+", t)
                if parts2: lines.append("".join(parts2).strip())
    return " ".join(lines)

def extract_english_from_vtt(text):
    lines = []
    VTT_TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}")
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("WEBVTT") or VTT_TIME_RE.match(s): continue
        s = re.sub(r"<.*?>", "", s)
        parts = re.findall(r"[A-Za-z0-9'\",.?!:;()\- ]+", s)
        if parts: lines.append("".join(parts).strip())
    return " ".join(lines)

def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    if tag.startswith('V'): return wordnet.VERB
    if tag.startswith('N'): return wordnet.NOUN
    if tag.startswith('R'): return wordnet.ADV
    return None

def process_words(all_text, mode, min_len, filter_set=None):
    """处理核心逻辑"""
    TOKEN_RE = re.compile(r"[A-Za-z-]+")
    raw_tokens = TOKEN_RE.findall(all_text)
    cleaned = [re.sub(r'[^a-z]', '', w.lower()) for w in raw_tokens]
    cleaned = [w for w in cleaned if w]

    lemmatized = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    if mode == "spacy" and nlp_spacy is not None:
        status_text.text("正在使用 spaCy (精准模式)...")
        chunk_size = 50000
        chunks = [cleaned[i:i + chunk_size] for i in range(0, len(cleaned), chunk_size)]
        for i, chunk in enumerate(chunks):
            doc = nlp_spacy(" ".join(chunk))
            for token in doc:
                lw = token.lemma_.lower()
                if lw.isalpha() and wordnet.synsets(lw):
                    lemmatized.append(lw)
            progress_bar.progress((i + 1) / len(chunks))
    else:
        status_text.text("正在使用 NLTK (快速模式)...")
        lemmatizer = WordNetLemmatizer()
        tagged = pos_tag(cleaned)
        total = len(tagged)
        for i, (w, tag) in enumerate(tagged):
            wn = get_wordnet_pos(tag)
            lw = lemmatizer.lemmatize(w, wn) if wn else lemmatizer.lemmatize(w)
            if wordnet.synsets(lw):
                lemmatized.append(lw)
            if i % 5000 == 0:
                progress_bar.progress(min(i / total, 1.0))
        progress_bar.progress(1.0)

    status_text.text("正在去重和过滤...")
    seen = set()
    final_words = []
    sys_stopwords = set(stopwords.words('english'))
    
    for w in lemmatized:
        if len(w) < min_len: continue
        if w in sys_stopwords: continue
        if filter_set and w in filter_set: continue
        if w not in seen:
            seen.add(w)
            final_words.append(w)
            
    status_text.empty()
    progress_bar.empty()
    return final_words

# ------------------ UI 布局 ------------------

st.title("📘 英语生词本 & AI 助教")
st.markdown("上传字幕/文档 -> 提取单词 -> **AI 辅助学习**")

with st.sidebar:
    st.header("⚙️ 提取设置")
    nlp_mode = st.selectbox("NLP 引擎", ["nltk (快速)", "spacy (精准)"], index=0)
    mode_key = "spacy" if "spacy" in nlp_mode else "nltk"
    min_len = st.number_input("最小长度", min_value=1, value=3)
    chunk_size = st.number_input("切分大小", min_value=100, value=5000)
    sort_order = st.radio("排序", ["按出现顺序", "A-Z 排序", "随机打乱"])
    
    st.divider()
    filter_file = st.file_uploader("过滤词表 (.txt)", type=['txt'])
    filter_set = set()
    if filter_file:
        content = filter_file.getvalue().decode("utf-8", errors='ignore')
        filter_set = set(line.strip().lower() for line in content.splitlines() if line.strip())
        st.success(f"已加载 {len(filter_set)} 个过滤词")

    # =========== [新增] Agent 设置 ===========
    st.divider()
    st.header("🤖 AI 设置")
    # 优先从 Secrets 读取 API Key，方便部署后不用每次输入
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.success("已检测到配置的 API Key")
    else:
        api_key = st.text_input("API Key", type="password", help="输入 OpenAI/DeepSeek Key")
        
    base_url = st.text_input("API URL", value="https://api.deepseek.com", help="例如 DeepSeek 或 OpenAI 地址")
    model_name = st.text_input("模型名称", value="deepseek-chat")
    # ========================================

# =========== [新增] 初始化记忆 (Session State) ===========
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = None
if "messages" not in st.session_state:
    st.session_state.messages = []
# =======================================================

uploaded_files = st.file_uploader("拖拽文件到此处", type=['txt', 'srt', 'ass', 'vtt', 'docx'], accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 开始提取", type="primary"):
        all_raw_text = []
        read_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            file.seek(0)
            text = extract_text_from_bytes(file, file.name)
            all_raw_text.append(text)
            read_bar.progress((i + 1) / len(uploaded_files))
        read_bar.empty()
        
        full_text = "\n".join(all_raw_text)
        result_words = process_words(full_text, mode_key, min_len, filter_set)
        
        if sort_order == "A-Z 排序":
            result_words.sort()
        elif sort_order == "随机打乱":
            import random
            random.shuffle(result_words)
            
        st.success(f"🎉 提取完成！共 {len(result_words)} 个单词。")
        
        # 结果下载
        if result_words:
            zip_buffer = io.BytesIO()
            num_files = math.ceil(len(result_words) / chunk_size)
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for i in range(num_files):
                    start = i * chunk_size
                    end = min(start + chunk_size, len(result_words))
                    fname = f"word_list_{i+1}.txt"
                    zf.writestr(fname, "\n".join(result_words[start:end]))
            
            st.download_button("📥 下载生词本 (ZIP)", zip_buffer.getvalue(), "words.zip", "application/zip")
            
            # =========== [新增] 注入记忆 ===========
            st.session_state.agent_memory = {
                "text": full_text,
                "words": result_words
            }
            st.session_state.messages = [] # 清空旧对话
            st.toast("AI 已准备好，请下滑查看！", icon="🤖")
            # ======================================
        else:
            st.warning("未提取到单词。")

# =========== [新增] AI 聊天交互区 ===========
st.divider()

if st.session_state.agent_memory:
    st.subheader("🤖 AI 助教")
    st.caption("基于你上传的文档内容回答")
    
    # 快捷按钮
    col1, col2 = st.columns(2)
    if col1.button("📝 用前10个生词写故事"):
        st.session_state.messages.append({"role": "user", "content": "请用提取出的前10个单词写一个短篇英文故事，并附带中文翻译，生词请加粗。"})
    if col2.button("🧐 出3道阅读理解题"):
        st.session_state.messages.append({"role": "user", "content": "根据原文内容出3道单项选择题，并在最后附上答案和解析。"})

    # 显示历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入框
    if prompt := st.chat_input("例如：'distinguish' 在文中是什么意思？"):
        if not api_key:
            st.error("请先在左侧设置 API Key")
            st.stop()
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 构建 Prompt (RAG)
        mem = st.session_state.agent_memory
        # 为了防止 Token 超限，只截取前 3000 字符和前 100 个单词作为上下文
        context_text = mem["text"][:3000] + "..."
        context_words = ", ".join(mem["words"][:100])
        
        system_prompt = f"""
        你是一个英语助教。
        【资料】：
        1. 核心生词：{context_words}
        2. 原文片段：{context_text}
        
        【任务】：
        基于资料回答用户问题。若询问单词含义，请结合原文语境解释。
        """
        
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            with st.chat_message("assistant"):
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *st.session_state.messages
                    ],
                    stream=True
                )
                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"API 错误: {e}")

elif uploaded_files:
    st.info("👆 提取完成后，这里会出现 AI 聊天界面。")
