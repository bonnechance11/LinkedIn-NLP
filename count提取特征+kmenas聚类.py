import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ====================== 0. 基础配置 ======================
os.environ['OMP_NUM_THREADS'] = '1'  # 解决Windows内存泄漏
BATCH_SIZE = 8  # 保持小批量确保稳定性
MODEL = "CountVectorizer"  # 仅处理该模型
OUTPUT_DIR = "cluster_results"
K = 4  # 固定簇数（如需自动选择可恢复肘部法则，此处为快速模式设为4）

os.makedirs(f"{OUTPUT_DIR}/{MODEL}", exist_ok=True)

# ====================== 1. 数据预处理 ======================
print("=== 数据预处理 ===")
data = pd.read_excel('linkedin_jobs.xlsx')
data[['title', 'description']] = data[['title', 'description']].astype(str).fillna("empty")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) | {'role', 'required', 'responsibility'}

# 增强版停用词表（包含用户新增词汇）
stop_words = set(stopwords.words('english'))
extra_stopwords = {
    'work', 'job', 'career', 'opportunity', 'employment', 'status', 'may', 'equal', 'using',
    'verizon', 'united', 'paid', 'status,,', 'law.',
    'search', 'offer', 'center', 'team', 'part', 'join', 'time', 'year', 'make', 'please', 'must'
}
ALL_STOPWORDS = stop_words.union(extra_stopwords)

def preprocess(text):
    tokens = [lemmatizer.lemmatize(token.lower()) for token in text.split()
              if token.lower() not in ALL_STOPWORDS and len(token) > 2]
    return ' '.join(tokens) if tokens else "empty"


data['desc_processed'] = data['description'].apply(preprocess)
print(f"有效文本数: {len(data)}")

# ====================== 2. CountVectorizer特征提取 ======================
print(f"\n=== {MODEL} 特征提取 ===")
vectorizer = CountVectorizer(max_features=2000)
features = vectorizer.fit_transform(data['desc_processed']).toarray()

# ====================== 3. 快速聚类（固定簇数K=4，根据之前的肘部法则结果） ======================
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = kmeans.fit_predict(features)
data[f"{MODEL}_cluster"] = labels


# ====================== 4. 保存簇文本文件（关键部分） ======================
def save_cluster_texts(model_name, labels, k):
    model_dir = f"{OUTPUT_DIR}/{model_name}"
    for c in range(k):
        cluster_data = data[labels == c][['title', 'description', 'desc_processed']]
        file_path = f"{model_dir}/cluster_{c}.txt"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"=== {model_name} 簇 {c} 样本数: {len(cluster_data)} ===\n\n")
            for idx, row in cluster_data.iterrows():
                f.write(f"【岗位标题】{row['title']}\n")
                f.write(f"【岗位描述】{row['description']}\n")
                f.write(f"【预处理文本】{row['desc_processed']}\n\n")
        print(f"已保存簇 {c} 到 {file_path}")


save_cluster_texts(MODEL, labels, K)
print(f"\n=== {MODEL} 聚类完成，共生成{K}个簇文本文件 ===")