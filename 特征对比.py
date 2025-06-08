import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from transformers import BertTokenizer, BertModel
import torch
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ====================== 0. 基础配置 ======================
os.environ['OMP_NUM_THREADS'] = '1'  # 解决Windows内存泄漏
BATCH_SIZE = 8  # BERT批量处理大小
MODELS = ["BERT", "TF-IDF", "CountVectorizer"]  # 对比模型列表
OUTPUT_DIR = "cluster_results"  # 输出目录
K_RANGE = range(2, 11)  # 搜索簇数范围（k=2到k=10）

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文显示问题
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# ====================== 1. 数据预处理（使用全部数据） ======================
print("=== 数据预处理 ===")
data = pd.read_excel('linkedin_jobs.xlsx')  # 使用全部数据
data[['title', 'description']] = data[['title', 'description']].astype(str).fillna("empty")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) | {'using', 'use', 'used', 'also', 'would', 'could', 'might', 'get', 'need',
    'required', 'require', 'requirements', 'responsibility', 'responsibilities',
    'role', 'amp', 'nbsp', 'including', 'include', 'within', 'across', 'over',
    'into', 'onto', 'via', 'without', 'among', 'along', 'during', 'towards',
    'job', 'post', 'position', 'apply', 'application'}# 新增职位相关通用词}


def preprocess(text):
    tokens = [lemmatizer.lemmatize(token.lower()) for token in text.split()
              if token.lower() not in stop_words and len(token) > 2]
    return ' '.join(tokens) if tokens else "empty"


data['desc_processed'] = data['description'].apply(preprocess)
print(f"有效文本数: {len(data)}")


# ====================== 2. 特征提取（不变） ======================
def get_features(model_name):
    if model_name == "BERT":
        print(f"\n=== {model_name} 特征提取 ===")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
        model = BertModel.from_pretrained('bert-base-uncased', device_map='cpu').eval()
        embeddings = []
        for i in tqdm(range(0, len(data), BATCH_SIZE), desc="处理批次"):
            batch = data.iloc[i:i + BATCH_SIZE]['desc_processed'].tolist()
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
            with torch.no_grad():
                emb = model(**inputs).last_hidden_state.mean(dim=1).numpy()
                embeddings.extend(emb)
        return np.array(embeddings)

    else:
        vectorizer = TfidfVectorizer(max_features=2000) if model_name == "TF-IDF" else CountVectorizer()
        print(f"\n=== {model_name} 特征提取 ===")
        return vectorizer.fit_transform(data['desc_processed']).toarray()


# ====================== 3. 肘部法则自动确定最佳簇数 ======================
def find_elbow_point(k_values, inertias):
    """通过二阶差分法自动检测肘部点"""
    # 计算一阶差分（斜率变化）
    diff1 = np.diff(inertias)
    # 计算二阶差分（曲率）
    diff2 = np.diff(diff1)
    # 找到二阶差分最大值的索引（拐点）
    elbow_idx = np.argmax(diff2) + 1  # 索引调整
    return k_values[elbow_idx]


def determine_optimal_clusters(features):
    """计算惯性并返回最佳k值"""
    inertias = []
    k_values = list(K_RANGE)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)

    # 绘制肘部曲线
    plt.figure(figsize=(10, 6))
    # 修复后：移除fmt字符串，单独使用关键字参数（避免冗余定义）
    plt.figure(figsize=(10, 6))
    plt.plot(
        k_values, inertias,
        marker='o',  # 显式设置标记
        color='#3498db',  # 显式设置颜色
        linestyle='-',  # 显式设置线型
        markersize=6,  # 增加标记大小提升可读性
        linewidth=2  # 增加线条宽度
    )
    plt.title("肘部法则确定最佳簇数")
    plt.xlabel("簇数 (k)")
    plt.ylabel("惯性 (Inertia)")
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/elbow_curve.png")
    plt.close()

    # 自动检测肘部点
    try:
        optimal_k = find_elbow_point(k_values, inertias)
        print(f"自动检测到肘部点簇数: {optimal_k}")
        return optimal_k
    except:
        print("自动检测失败，使用默认簇数3")
        return 3  # fallback值


# ====================== 4. 聚类与可视化 ======================
def cluster_and_visualize(model_name, features, optimal_k):
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    score = silhouette_score(features, labels)

    # TSNE可视化
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    emb_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 6))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', alpha=0.7, edgecolors='white')
    plt.title(f"{model_name} 聚类分布\n(簇数={optimal_k}, 轮廓系数={score:.4f})", fontsize=12)
    plt.xlabel("TSNE 维度1")
    plt.ylabel("TSNE 维度2")
    plt.savefig(f"{OUTPUT_DIR}/tsne_{model_name.lower()}.png")
    plt.close()

    return labels, score, optimal_k


# ====================== 5. 结果保存函数 ======================
def save_cluster_results(model_name, labels, optimal_k):
    """整合文本文件、Excel标签、信息表格保存"""
    # 1. 保存簇文本文件
    model_dir = f"{OUTPUT_DIR}/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    for c in range(optimal_k):
        cluster_data = data[labels == c][['title', 'description', 'desc_processed']]
        file_path = f"{model_dir}/cluster_{c}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"=== {model_name} 簇{c} 样本数: {len(cluster_data)} ===\n\n")
            for idx, row in cluster_data.iterrows():
                f.write(f"【标题】{row['title']}\n")
                f.write(f"【描述】{row['description']}\n")
                f.write(f"【预处理文本】{row['desc_processed']}\n\n")
        print(f"已保存 {model_name} 簇{c} 到 {file_path}")

    # 2. 保存聚类标签到Excel（分sheet）
    excel_path = f"{OUTPUT_DIR}/cluster_labels.xlsx"
    data[f"{model_name}_cluster"] = labels
    if not os.path.exists(excel_path):
        data.to_excel(excel_path, index=False, engine="xlsxwriter")
    else:
        with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="new") as writer:
            data[f"{model_name}_cluster"].to_excel(writer, sheet_name=model_name, index=False)

    # 3. 保存含Title/Description的新表格
    save_cluster_info_table(model_name, labels, optimal_k)


def save_cluster_info_table(model_name, labels, optimal_k):
    """保存包含标题、描述、聚类标签的表格"""
    df = data[['title', 'description', f'{model_name}_cluster']].copy()
    df.columns = ['岗位标题', '岗位描述', '聚类标签']
    table_path = f"{OUTPUT_DIR}/{model_name}_详细信息表.xlsx"
    df.to_excel(table_path, index=False, engine="xlsxwriter")
    print(f"已生成详细信息表: {table_path}")


# ====================== 6. 主流程 ======================
if __name__ == "__main__":
    model_scores = []  # 存储各模型轮廓系数和簇数

    for model in MODELS:
        features = get_features(model)

        # 1. 确定最佳簇数
        print(f"\n=== 开始分析 {model} ===")
        optimal_k = determine_optimal_clusters(features)

        # 2. 聚类与可视化
        labels, score, optimal_k = cluster_and_visualize(model, features, optimal_k)
        save_cluster_results(model, labels, optimal_k)
        model_scores.append((model, score, optimal_k))  # 收集分数和簇数

    # ====================== 绘制轮廓系数对比柱状图 ======================
    plt.figure(figsize=(8, 5))
    models, scores, ks = zip(*model_scores)
    bars = plt.bar(models, scores, color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='black')

    plt.title("不同模型轮廓系数对比")
    plt.xlabel("模型")
    plt.ylabel("轮廓系数（值越高聚类质量越好）")
    plt.ylim(0, 1)

    # 添加数据标签（包含簇数信息）
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}\n(k={ks[i]})',
                 ha='center', va='bottom', fontsize=9, linespacing=1.2)

    plt.savefig(f"{OUTPUT_DIR}/silhouette_comparison.png")
    plt.close()

    print("\n=== 所有操作完成 ===")
    print("1. 肘部曲线已保存为 cluster_results/elbow_curve.png")
    print("2. 各模型TSNE图和详细信息表已按自动检测的簇数生成")
    print("3. 轮廓系数对比图包含各模型使用的簇数（k值）")