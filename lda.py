import os
import nltk
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords

# ====================== 0. 全局配置 ======================
CLUSTER_DIR = "cluster_results\\CountVectorizer"
RESULT_DIR = "theme_results"
os.makedirs(RESULT_DIR, exist_ok=True)  # 自动创建结果目录

NUM_TOPICS_PER_CLUSTER = 3  # 每个簇提取的主题数
N_TOP_WORDS = 5  # 每个主题的Top词数

# 增强版停用词表（包含用户新增词汇）
stop_words = set(stopwords.words('english'))
extra_stopwords = {
    'work', 'job', 'career', 'opportunity', 'employment', 'status', 'may', 'equal', 'using',
    'verizon', 'united', 'paid', 'status,,', 'law.',
    'search', 'offer', 'center', 'team', 'part', 'join', 'time', 'year', 'make', 'please', 'must'
}
ALL_STOPWORDS = stop_words.union(extra_stopwords)


# ====================== 1. 簇文件加载 ======================
def load_cluster_files():
    """按顺序加载簇文件，返回每个簇的文本列表"""
    clusters = []
    cluster_num = 0
    while True:
        file_path = os.path.join(CLUSTER_DIR, f"cluster_{cluster_num}.txt")
        if not os.path.exists(file_path):
            break  # 停止于第一个缺失的文件编号
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]  # 过滤空行
            if not texts:
                raise ValueError(f"簇{cluster_num}文件为空，跳过该簇")
            clusters.append(texts)
        cluster_num += 1
    if not clusters:
        raise ValueError("未找到任何有效簇文件，请检查CLUSTER_DIR路径")
    return clusters


# ====================== 2. 文本预处理 ======================
def preprocess_cluster_texts(texts):
    """对单个簇的文本进行预处理：清洗、分词、去停用词"""
    processed_tokens = []
    for text in texts:
        # 1. 清洗：去除非字母字符，转为小写
        cleaned_text = ''.join([c.lower() for c in text if c.isalpha() or c == ' '])
        # 2. 分词：按空格分割并过滤停用词和短词（长度<4）
        tokens = [token for token in cleaned_text.split()
                  if token not in ALL_STOPWORDS and len(token) >= 4]
        if tokens:  # 过滤无有效token的文档
            processed_tokens.append(tokens)
    if not processed_tokens:
        raise ValueError("预处理后簇内无有效token，请检查文本内容")
    return processed_tokens


# ====================== 3. LDA模型训练（移除困惑度计算） ======================
def train_lda_model(token_list, num_topics):
    """返回 (lda模型, 词典)"""
    dictionary = Dictionary(token_list)
    corpus = [dictionary.doc2bow(tokens) for tokens in token_list]

    lda_model = LdaModel(
        corpus=corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=20,  # 训练轮数
        random_state=42,  # 固定随机种子确保可复现
        chunksize=1000  # 分块大小，优化内存使用
    )
    return lda_model, dictionary  # 仅返回模型和词典


# ====================== 4. 主题词清洗 ======================
def clean_topic_keywords(raw_topics):
    """对原始主题词进行二次清洗，去除残留无意义词"""
    return [
        [word for word, _ in topic
         if word not in ALL_STOPWORDS and len(word) >= 4]
        for topic in raw_topics
    ]


# ====================== 5. 结果保存（移除困惑度相关内容） ======================
def save_theme_results(cluster_idx, topics):
    """将主题结果保存到独立文件"""
    save_path = os.path.join(RESULT_DIR, f"cluster_{cluster_idx}_themes.txt")
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"=== 簇{cluster_idx} 主题分析报告 ===\n")
        f.write(f"• 主题数: {NUM_TOPICS_PER_CLUSTER}\n\n")  # 移除困惑度相关行

        for theme_idx, words in enumerate(topics):
            f.write(f"【主题{theme_idx + 1}】Top{N_TOP_WORDS}关键词:\n")
            if words:
                f.write("  • " + "\n  • ".join(words) + "\n")
            else:
                f.write("  无有效关键词（可能文本预处理过严）\n")
            f.write("-" * 50 + "\n")
    print(f"已保存簇{cluster_idx}结果到 {save_path}")


# ====================== 6. 主执行流程（移除困惑度相关逻辑） ======================
def main():
    print("=== 开始主题提取 ===")

    # 下载NLTK停用词（首次运行时自动下载）
    nltk.download('stopwords', quiet=True)  # quiet模式避免重复提示

    # 1. 加载簇文件
    print(f"正在加载簇文件（目录：{CLUSTER_DIR}）...")
    try:
        cluster_texts = load_cluster_files()
        print(f"成功加载{len(cluster_texts)}个簇文件")
    except Exception as e:
        print(f"加载失败：{str(e)}")
        return

    # 2. 预处理每个簇
    print("\n正在预处理文本（去停用词+清洗）...")
    cluster_tokens = []
    for idx, texts in enumerate(cluster_texts):
        try:
            tokens = preprocess_cluster_texts(texts)
            cluster_tokens.append(tokens)
            print(f"簇{idx}预处理完成（有效文档数：{len(tokens)}）")
        except Exception as e:
            print(f"簇{idx}预处理失败：{str(e)}，跳过该簇")

    # 3. 提取每个簇的主题（移除困惑度计算）
    print("\n开始训练LDA模型并提取主题...")
    results = []
    for idx, tokens in enumerate(cluster_tokens):
        try:
            lda_model, dictionary = train_lda_model(tokens, NUM_TOPICS_PER_CLUSTER)
            raw_topics = [lda_model.show_topic(tid, topn=N_TOP_WORDS)
                          for tid in range(NUM_TOPICS_PER_CLUSTER)]
            cleaned_topics = clean_topic_keywords(raw_topics)
            results.append(cleaned_topics)
            print(f"簇{idx}主题提取完成")
        except Exception as e:
            print(f"簇{idx}训练失败：{str(e)}，记录空主题")
            results.append([[]] * NUM_TOPICS_PER_CLUSTER)  # 记录空主题

    # 4. 保存结果（无困惑度）
    print("\n正在保存主题结果...")
    for idx, topics in enumerate(results):
        save_theme_results(idx, topics)

    # 5. 控制台摘要（仅显示主题）
    print("\n=== 最终主题结果汇总 ===")
    for idx, topics in enumerate(results):
        print(f"\n簇{idx}:")
        for theme_idx, words in enumerate(topics):
            if words:
                print(f"  主题{theme_idx + 1}: {', '.join(words)}")
            else:
                print(f"  主题{theme_idx + 1}: 无有效关键词（已过滤无意义词）")


# ====================== 7. 执行主程序 ======================
if __name__ == "__main__":
    main()