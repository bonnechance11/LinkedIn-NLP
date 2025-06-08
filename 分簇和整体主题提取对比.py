import os
import pandas as pd
import numpy as np
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# ====================== 0. 配置与文件路径 ======================
CLUSTER_DIR = "cluster_results\\CountVectorizer"  # 聚类后的txt文件目录
EXCEL_FILE = "linkedin_jobs.xlsx"  # 整体数据Excel文件
TEXT_COLUMN = "description"  # Excel中的文本列名
NUM_TOPICS = 3  # 每个簇/整体提取的主题数
N_TOP_WORDS = 5  # 主题词数量

# 停用词表（与聚类时保持一致）
stop_words = set(stopwords.words('english'))
extra_stopwords = {
    'work', 'job', 'career', 'opportunity', 'employment', 'status', 'may', 'equal', 'using',
    'verizon', 'united', 'paid', 'status,,', 'law.', 'search', 'offer', 'center', 'team',
    'part', 'join', 'time', 'year', 'make', 'please', 'must'
}
ALL_STOPWORDS = stop_words.union(extra_stopwords)


# ====================== 1. 数据加载函数 ======================
def load_cluster_texts():
    """加载聚类后的每个簇文本（从txt文件）"""
    clusters = []
    cluster_num = 0
    while True:
        file_path = os.path.join(CLUSTER_DIR, f"cluster_{cluster_num}.txt")
        if not os.path.exists(file_path):
            break
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if "【岗位描述】" in line]  # 提取描述部分
            texts = [line.split("【岗位描述】")[1].strip() for line in texts if line]
            clusters.append(texts)
        cluster_num += 1
    return clusters


def load_global_texts():
    """从Excel加载整体文本（原始数据）"""
    data = pd.read_excel(EXCEL_FILE)
    return data[TEXT_COLUMN].dropna().tolist()  # 提取描述列并去空


# ====================== 2. 文本预处理函数（通用） ======================
def preprocess(texts):
    """清洗文本、分词、去停用词（与聚类时完全一致）"""
    processed_tokens = []
    for text in texts:
        # 1. 清洗：保留字母和空格，转小写，去除多余符号
        cleaned_text = ''.join([c.lower() for c in str(text) if c.isalpha() or c == ' '])
        tokens = cleaned_text.split()
        # 2. 过滤停用词和短词（长度<4）
        filtered_tokens = [token for token in tokens
                           if token not in ALL_STOPWORDS and len(token) >= 4]
        if filtered_tokens:
            processed_tokens.append(filtered_tokens)
    return processed_tokens


# ====================== 3. 分簇主题提取及评估 ======================
def process_cluster_topics(cluster_texts_list):
    """处理每个簇的主题提取并计算一致性"""
    cluster_coherences = []
    cluster_topics_all = []

    for idx, texts in enumerate(cluster_texts_list):
        if not texts:
            print(f"簇{idx + 1}无有效文本，跳过")
            continue

        tokens = preprocess(texts)
        if not tokens:
            print(f"簇{idx + 1}预处理后无有效token，跳过")
            continue

        # 提取主题并计算一致性
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(tok_list) for tok_list in tokens]
        lda_model = LdaModel(
            corpus,
            num_topics=NUM_TOPICS,
            id2word=dictionary,
            passes=20,
            random_state=42
        )
        topics = [lda_model.show_topic(tid, topn=N_TOP_WORDS) for tid in range(NUM_TOPICS)]

        coherence = CoherenceModel(
            topics=[[word for word, _ in topic] for topic in topics],
            texts=tokens,
            dictionary=dictionary,
            coherence='c_v'
        ).get_coherence()

        cluster_coherences.append(coherence)
        cluster_topics_all.append(topics)

    return cluster_coherences, cluster_topics_all


# ====================== 4. 整体主题提取及评估 ======================
def process_global_topics(global_texts):
    """处理Excel整体文本的主题提取并计算一致性"""
    tokens = preprocess(global_texts)
    if not tokens:
        raise ValueError("整体文本预处理后无有效token")

    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(tok_list) for tok_list in tokens]
    lda_model = LdaModel(
        corpus,
        num_topics=NUM_TOPICS,
        id2word=dictionary,
        passes=20,
        random_state=42
    )
    topics = [lda_model.show_topic(tid, topn=N_TOP_WORDS) for tid in range(NUM_TOPICS)]

    coherence = CoherenceModel(
        topics=[[word for word, _ in topic] for topic in topics],
        texts=tokens,
        dictionary=dictionary,
        coherence='c_v'
    ).get_coherence()

    return topics, coherence


# ====================== 5. 主执行流程 ======================
def main():
    # 1. 加载数据
    print("=== 加载数据 ===")
    cluster_texts = load_cluster_texts()  # 从txt加载分簇文本
    global_texts = load_global_texts()  # 从Excel加载整体文本
    print(f"分簇数据：{len(cluster_texts)}个簇，整体数据：{len(global_texts)}条文本")

    # 2. 分簇主题处理
    print("\n=== 分簇主题提取 ===")
    cluster_coherences, cluster_topics = process_cluster_topics(cluster_texts)
    avg_cluster_coherence = np.mean(cluster_coherences)
    print(f"分簇平均一致性分数: {avg_cluster_coherence:.4f}")

    # 3. 整体主题处理
    print("\n=== 整体主题提取 ===")
    global_topics, global_coherence = process_global_topics(global_texts)
    print(f"整体一致性分数: {global_coherence:.4f}")

    # 4. 可视化对比
    plt.figure(figsize=(12, 6))

    # 分簇分数（每个簇单独显示）
    plt.bar(
        [f"簇{i + 1}" for i in range(len(cluster_coherences))],
        cluster_coherences,
        width=0.4,
        label="分簇主题一致性",
        color='#2ca02c',
        alpha=0.9
    )

    # 整体分数（对比线）
    plt.axhline(
        y=global_coherence,
        color='#1f77b4',
        linestyle='--',
        linewidth=2,
        label=f"整体主题一致性 ({global_coherence:.4f})"
    )

    plt.title("分簇 vs 整体主题提取效果对比（C-V一致性分数）")
    plt.ylabel("主题一致性分数（范围：-1 ~ 1，越高越好）")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # 5. 主题词对比（各选第一个簇和整体前2个主题）
    print("\n=== 主题词对比 ===")
    print("【分簇主题示例（簇1）】")
    for tid, topic in enumerate(cluster_topics[0][:2]):  # 分簇第一个簇的前2个主题
        words = [word for word, _ in topic]
        print(f"  主题{tid + 1}: {', '.join(words)}")

    print("\n【整体主题示例】")
    for tid, topic in enumerate(global_topics[:2]):  # 整体前2个主题
        words = [word for word, _ in topic]
        print(f"  主题{tid + 1}: {', '.join(words)}")


# ====================== 6. 执行主程序 ======================
if __name__ == "__main__":
    main()