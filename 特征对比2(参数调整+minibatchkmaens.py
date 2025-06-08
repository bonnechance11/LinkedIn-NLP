import os
import os
import pandas as pd
import numpy as np
import re
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from transformers import BertTokenizer, BertModel
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

# ====================== 0. 全局配置 ======================
os.environ['OMP_NUM_THREADS'] = '8'
BATCH_SIZE = {'bert': 32, 'fast': 128}  # 差异化批处理配置
MODELS_RUN_ORDER = ["tfidf", "count", "bert"]  # 先跑轻量模型再跑BERT
PREPROCESS_SAVE_PATH = "preprocessed_data.pkl"
OUTPUT_DIR = "step_by_step_results"
K_RANGE = range(3, 7)


# ====================== 1. 智能预处理（带缓存） ======================
def load_or_preprocess_data():
    if os.path.exists(PREPROCESS_SAVE_PATH):
        print("[INFO] 加载预处理缓存...")
        return pd.read_pickle(PREPROCESS_SAVE_PATH)

    print("[STEP 1] 完整数据预处理（带词性还原）")
    data = pd.read_excel('linkedin_jobs.xlsx', dtype={'title': 'string', 'description': 'string'})
    data[['title', 'description']] = data[['title', 'description']].fillna("empty")

    stop_words = set(stopwords.words('english')) | {
        'using', 'use', 'used', 'also', 'would', 'could', 'get', 'need',
        'required', 'require', 'role', 'amp', 'nbsp', 'including', 'within',
        'across', 'over', 'into', 'onto', 'via', 'without', 'among', 'along',
        'during', 'towards', 'job', 'post', 'position', 'apply', 'application',
        'team', 'work', 'year', 'month', 'day', 'hour', 'minute'
    }
    lemmatizer = WordNetLemmatizer()
    CLEAN_REGEX = re.compile(r'[^a-zA-Z0-9\s]', re.IGNORECASE)

    def preprocess(text):
        if len(text) < 50: return "empty"
        text = text.lower().strip()
        text = CLEAN_REGEX.sub(' ', text)
        text = re.sub(r'\b\d+\b', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        tokens = [
            lemmatizer.lemmatize(token, pos='v') if token.startswith(('un', 're', 'dis'))
            else lemmatizer.lemmatize(token)
            for token in text.split()
            if token not in stop_words and len(token) > 2
        ]
        return ' '.join(tokens) if len(tokens) >= 5 else "empty"

    data['desc_processed'] = data['description'].apply(preprocess)
    valid_count = data['desc_processed'].ne("empty").sum()
    print(f"[INFO] 有效文本数: {valid_count}/{len(data)} ({valid_count / len(data):.1%})")

    # 保存预处理结果
    pd.to_pickle(data, PREPROCESS_SAVE_PATH)
    return data


data = load_or_preprocess_data()


# ====================== 2. 特征提取模块（分模型实现） ======================
def get_tfidf_features():
    print("\n=== TF-IDF特征提取 ===")
    return TfidfVectorizer(
        max_features=8000, ngram_range=(1, 2),
        stop_words=list(stopwords.words('english'))
    ).fit_transform(data['desc_processed'])


def get_count_features():
    print("\n=== CountVectorizer特征提取 ===")
    return CountVectorizer(
        max_features=10000, ngram_range=(1, 1),
        stop_words=list(stopwords.words('english'))
    ).fit_transform(data['desc_processed'])


def get_bert_features():
    print("\n=== BERT本体特征提取 ===")
    model_name_or_path = 'bert-base-uncased'  # 需本地已存在
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    model = BertModel.from_pretrained(model_name_or_path).eval()

    embeddings = []
    for i in tqdm(range(0, len(data), BATCH_SIZE['bert']), desc="BERT处理"):
        batch = data.iloc[i:i + BATCH_SIZE['bert']]['desc_processed'].tolist()
        if not batch: continue

        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=384)
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            embeddings.extend(normalize(emb, axis=1))
    return embeddings


# ====================== 3. 通用聚类与可视化函数 ======================
def cluster_and_visualize(features, model_name):
    print(f"\n=== {model_name.upper()} 聚类处理 ===")
    if model_name in ["tfidf", "count"]:
        features = TruncatedSVD(n_components=500).fit_transform(features)

    mbk = MiniBatchKMeans(
        batch_size=BATCH_SIZE['fast'], n_init=10, max_iter=300,
        random_state=42, verbose=0
    )

    best_score, best_k = -1, 3
    for k in K_RANGE:
        mbk.n_clusters = k
        labels = mbk.fit_predict(features)
        metric = 'cosine' if model_name == 'bert' else 'euclidean'
        score = silhouette_score(features, labels, metric=metric)

        if score > best_score:
            best_score, best_k = score, k
            best_labels = labels

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(*TruncatedSVD(n_components=2).fit_transform(features).T,
                c=best_labels, cmap='tab10', alpha=0.7)
    plt.title(f"{model_name.upper()} 聚类结果 (k={best_k}, 轮廓系数={best_score:.4f})")
    plt.savefig(f"{OUTPUT_DIR}/{model_name}_cluster.png", dpi=200)
    return best_k, best_score


# ====================== 4. 主流程控制（按顺序执行） ======================
os.makedirs(OUTPUT_DIR, exist_ok=True)

for model in MODELS_RUN_ORDER:
    if model == 'bert' and not os.path.exists(PREPROCESS_SAVE_PATH):
        raise FileNotFoundError("请先完成预处理并保存缓存")

    feature_func = {
        'tfidf': get_tfidf_features,
        'count': get_count_features,
        'bert': get_bert_features
    }[model]

    features = feature_func()
    cluster_and_visualize(features, model)

print("\n[FINISH] 所有模型处理完成，结果按运行顺序保存在", OUTPUT_DIR)