import os
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, normalize
from transformers import BertTokenizer, BertModel
import torch
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ====================== 0. 基础配置（效果与效率平衡） ======================
os.environ['OMP_NUM_THREADS'] = '8'  # CPU多线程优化
BATCH_SIZE = 64  # GPU批次大小（RTX3060及以上推荐）
MODELS = ["BERT", "TF-IDF", "CountVectorizer"]  # 三模型全保留
OUTPUT_DIR = "cluster_results"
K_RANGE = range(2, 11)  # 肘部法则k值范围
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 使用设备: {DEVICE}, BATCH_SIZE={BATCH_SIZE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 中文显示
plt.rcParams["axes.unicode_minus"] = False

# ====================== 1. 数据预处理（增强停用词+严格清洗） ======================
print("[STEP 1] 数据预处理")
data = pd.read_excel('linkedin_jobs.xlsx', dtype={'title': 'string', 'description': 'string'})
data[['title', 'description']] = data[['title', 'description']].fillna("empty")  # 填充缺失值

lemmatizer = WordNetLemmatizer()
# 增强停用词表：NLTK基础词表 + 职位描述高频无意义词
stop_words = set(stopwords.words('english')) | {
    'using', 'use', 'used', 'also', 'would', 'could', 'might', 'get', 'need',
    'required', 'require', 'requirements', 'responsibility', 'responsibilities',
    'role', 'amp', 'nbsp', 'including', 'include', 'within', 'across', 'over',
    'into', 'onto', 'via', 'without', 'among', 'along', 'during', 'towards',
    'job', 'post', 'position', 'apply', 'application'  # 新增职位相关通用词
}
CLEAN_REGEX = re.compile(r'[^a-zA-Z0-9\s]', re.IGNORECASE)  # 清洗特殊字符


def text_preprocess(text):
    """深度文本清洗与词形还原"""
    text = text.lower()  # 转小写
    text = CLEAN_REGEX.sub(' ', text)  # 移除特殊符号
    text = re.sub(r'\b\d+\b', ' ', text)  # 移除纯数字词
    text = re.sub(r'\s+', ' ', text).strip()  # 合并空格

    tokens = [
        lemmatizer.lemmatize(token)
        for token in text.split()
        if token not in stop_words and len(token) > 2 and not token.isdigit()
    ]
    return ' '.join(tokens) if len(tokens) >= 4 else "empty"  # 保留至少4个词的文本


data['desc_processed'] = data['description'].apply(text_preprocess)
valid_count = data['desc_processed'].ne("empty").sum()
print(f"[INFO] 有效文本数: {valid_count}/{len(data)} ({valid_count / len(data):.1%})")


# ====================== 2. 特征提取（高质量特征构建） ======================
def get_bert_embeddings():
    """BERT特征提取（大模型+GPU加速）"""
    print("\n[STEP 2-1] BERT特征提取")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=True)  # 快速分词器
    model = BertModel.from_pretrained(
        'bert-base-uncased',
        device_map=DEVICE,
        output_hidden_states=False,
        torch_dtype=torch.float16  # 半精度加速
    ).eval()

    cls_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(data), BATCH_SIZE), desc="BERT批次处理"):
            batch = data.iloc[i:i + BATCH_SIZE]['desc_processed'].tolist()
            if not batch or all(t == "empty" for t in batch):
                continue

            # 分词并转移至GPU
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=384).to(DEVICE)
            # 提取CLS向量并转换为numpy
            cls_emb = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
            cls_embeddings.extend(cls_emb)

    # 特征后处理：标准化+PCA降维
    cls_embeddings = normalize(cls_embeddings, axis=1)  # L2标准化（余弦距离适配）
    pca = PCA(n_components=200, random_state=42)  # 保留更多主成分
    return pca.fit_transform(cls_embeddings)


def get_tfidf_features():
    """TF-IDF特征提取（高维+停用词增强）"""
    print("\n[STEP 2-2] TF-IDF特征提取")
    vectorizer = TfidfVectorizer(
        max_features=15000,  # 更多特征
        stop_words=list(stop_words),  # 使用增强停用词表
        token_pattern=r'\b\w{3,}\b',  # 至少3字符
        ngram_range=(1, 3),  # 包含1-3元语法
        max_df=0.95, min_df=2  # 过滤极端频率词
    )
    return vectorizer.fit_transform(data['desc_processed'])


def get_count_features():
    """CountVectorizer特征提取"""
    print("\n[STEP 2-3] CountVectorizer特征提取")
    vectorizer = CountVectorizer(
        max_features=15000,
        stop_words=list(stop_words),
        token_pattern=r'\b\w{3,}\b',
        ngram_range=(1, 3)
    )
    return vectorizer.fit_transform(data['desc_processed'])


# ====================== 3. 聚类分析（稳定性优化） ======================
def find_best_cluster_k(features, model_type):
    """基于轮廓系数的最优k值选择"""
    best_score = -1
    best_k = 3
    for k in K_RANGE:
        kmeans = KMeans(
            n_clusters=k,
            n_init=20,  # 更多初始化避免局部最优
            random_state=42,
            verbose=0
        )
        labels = kmeans.fit_predict(features)
        # 距离度量适配
        metric = 'cosine' if model_type == "BERT" else 'euclidean'
        score = silhouette_score(features, labels, metric=metric)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def cluster_and_visualize(model_name, features, optimal_k):
    """聚类与可视化"""
    kmeans = KMeans(
        n_clusters=optimal_k,
        n_init=20,
        random_state=42
    )
    labels = kmeans.fit_predict(features)

    # 计算轮廓系数
    metric = 'cosine' if model_name == "BERT" else 'euclidean'
    score = silhouette_score(features, labels, metric=metric)

    # TSNE降维（平衡效果与速度）
    if model_name == "BERT":
        tsne_input = PCA(n_components=100, random_state=42).fit_transform(features)
    else:
        tsne_input = StandardScaler().fit_transform(features.toarray())  # 标准化稀疏矩阵

    tsne = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=800)  # 更多迭代提升精度
    emb_2d = tsne.fit_transform(tsne_input)

    # 绘制聚类图
    plt.figure(figsize=(12, 8))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', alpha=0.8, edgecolors='white')
    plt.title(f"{model_name} 聚类结果 (k={optimal_k}, 轮廓系数={score:.4f})", fontsize=14)
    plt.savefig(f"{OUTPUT_DIR}/tsne_{model_name.lower()}.png", dpi=300)
    return labels, score


# ====================== 4. 结果保存 ======================
def save_cluster_results(model_name, labels, optimal_k):
    """保存聚类结果"""
    model_dir = f"{OUTPUT_DIR}/{model_name}"
    os.makedirs(model_dir, exist_ok=True)

    # 保存簇详情
    for c in range(optimal_k):
        cluster_data = data[labels == c][['title', 'description', 'desc_processed']]
        with open(f"{model_dir}/cluster_{c}_details.txt", 'w', encoding='utf-8') as f:
            f.write(f"=== {model_name} 簇 {c} 分析 ===\n")
            f.write(f"样本数: {len(cluster_data)}\n\n")
            f.write("典型标题示例:\n")
            for title in cluster_data['title'].head(5):
                f.write(f"- {title}\n")

    # 保存标签到Excel
    data[f"{model_name}_cluster"] = labels
    with pd.ExcelWriter(f"{OUTPUT_DIR}/cluster_labels.xlsx", mode='a', if_sheet_exists='new') as writer:
        data[['title', 'description', f"{model_name}_cluster"]].to_excel(writer, sheet_name=model_name, index=False)


# ====================== 5. 主流程控制 ======================
if __name__ == "__main__":
    feature_functions = {
        "BERT": get_bert_embeddings,
        "TF-IDF": get_tfidf_features,
        "CountVectorizer": get_count_features
    }

    results = []
    for model in MODELS:
        print(f"\n[MODEL] 处理 {model} 模型")
        features = feature_functions[model]()

        # 非BERT特征标准化（欧式距离需要）
        if model != "BERT":
            features = StandardScaler(with_mean=False).fit_transform(features)

        # 寻找最优k值
        optimal_k = find_best_cluster_k(features, model)
        print(f"[INFO] {model} 最优簇数: {optimal_k}")

        # 聚类与可视化
        labels, score = cluster_and_visualize(model, features, optimal_k)

        # 保存结果
        save_cluster_results(model, labels, optimal_k)
        results.append((model, score, optimal_k))
        print(f"[DONE] {model} 轮廓系数: {score:.4f}")

    # 绘制模型效果对比图
    plt.figure(figsize=(10, 6))
    models, scores, ks = zip(*results)
    bars = plt.bar(models, scores, color=['#2c3e50', '#3498db', '#27ae60'], edgecolor='white')

    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'k={ks[i]}', ha='center', va='bottom', fontsize=12)

    plt.title("不同模型聚类效果对比", fontsize=16)
    plt.xlabel("模型", fontsize=14)
    plt.ylabel("轮廓系数（值越大聚类质量越高）", fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{OUTPUT_DIR}/silhouette_comparison.png", dpi=300)

    print("\n[FINISH] 所有任务完成，结果保存在cluster_results目录")