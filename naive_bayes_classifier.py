# -*- coding: utf-8 -*-
import os, re, warnings
from collections import Counter

# ตั้งค่าโฟลเดอร์ข้อมูล PyThaiNLP ก่อน import pythainlp
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHAINLP_DATA_DIR"] = os.path.join(BASE_DIR, ".pythainlp-data")

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords

warnings.filterwarnings("ignore")
plt.rcParams["font.family"], FONT_PATH = "Tahoma", r"C:\Windows\Fonts\tahoma.ttf"
STOP_WORDS = set(thai_stopwords()) | {"this", "the", "is", "a", "an", "ครับ", "ค่ะ", "คะ", "นะ", "ๆ"}


def clean_and_tokenize(text: str) -> list[str]:
    """ตัดคำ ลบสัญลักษณ์/Stop words และลดรูปคำ (Stemming)"""
    text = re.sub(r"[^ก-ฮะ-์a-zA-Z\s]", " ", str(text)).lower()
    tokens = [re.sub(r"s$", "", w.strip()) for w in word_tokenize(text, engine="newmm", keep_whitespace=False)]
    return [w for w in tokens if len(w) > 1 and w not in STOP_WORDS]


# 0. โหลดข้อมูล (10,000 แถวเพื่อความรวดเร็ว) และตัดคำ
df = pd.read_csv(os.path.join(BASE_DIR, "w_review_train.csv"), sep=";", header=None, names=["review", "rating"]).dropna()
df, _ = train_test_split(df, train_size=10000, stratify=df["rating"], random_state=42)
df["tokens"] = df["review"].apply(clean_and_tokenize)
stars = sorted(df["rating"].unique())

# ==============================================================================
# 1. นับจำนวนคำ (Word Count) แยกตามระดับดาว
# ==============================================================================
print("=== 1. จำนวนคำ (Word Count) แยกตามระดับดาว ===")
star_counts = {s: Counter(w for toks in df[df["rating"] == s]["tokens"] for w in toks) for s in stars}
for s, c in star_counts.items():
    top_10 = ", ".join(f"{w}({n})" for w, n in c.most_common(10))
    print(f"★ {s} ดาว | รวม: {sum(c.values()):,} คำ (ไม่ซ้ำ {len(c):,} คำ) | Top 10: {top_10}")

# ==============================================================================
# 2. ภาพ Word Cloud 1 หน้าต่าง (เรียงตามจำนวนดาว 1-5)
# ==============================================================================
fig1, axes = plt.subplots(1, 5, figsize=(20, 4.5), num="Window 1 - Word Cloud (1-5 Stars)")
for ax, s, cmap in zip(axes, stars, ["Reds", "Oranges", "YlOrBr", "GnBu", "Blues"]):
    wc = WordCloud(font_path=FONT_PATH, width=400, height=400, background_color="white", colormap=cmap, regexp=r"[\u0E00-\u0E7Fa-zA-Z']+").generate_from_frequencies(star_counts[s])
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(f"ระดับ {s} ดาว ({sum(star_counts[s].values()):,} คำ)")
    ax.axis("off")
plt.tight_layout()
fig1.savefig(os.path.join(BASE_DIR, "wordcloud_by_stars.png"), dpi=150)

# ==============================================================================
# 3. คำนวณ Naive Bayes & Laplace Smoothing จากสมการโดยตรง (ไม่ใช้โมเดลสำเร็จรูป)
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(df["tokens"], df["rating"], test_size=0.2, stratify=df["rating"], random_state=42)
y_train_arr, y_test_arr = y_train.values, y_test.values

vec = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
X_train_tf = vec.fit_transform(X_train)
X_test_tf = vec.transform(X_test)
vocab, V_size, stars_arr = np.array(vec.get_feature_names_out()), len(vec.vocabulary_), np.array(stars)

# คำนวณ Prior P(c) และความถี่คำในแต่ละคลาส Count(w, c)
log_prior = np.log([(y_train_arr == c).sum() / len(y_train_arr) for c in stars_arr])
count_wc = np.array([X_train_tf[y_train_arr == c].sum(axis=0).A1 for c in stars_arr]) # (n_classes, |V|)
total_words_c = count_wc.sum(axis=1, keepdims=True)                                  # Total words per class

# ฟังก์ชันคำนวณ Posterior และทำนาย: y_pred = argmax [ log P(c) + sum( TF(w) * log P(w|c) ) ]
predict_nb = lambda log_p: stars_arr[np.argmax(X_test_tf.dot(log_p.T) + log_prior, axis=1)]

# 3.1 สมการ Smoothing = 0: P(w|c) = Count(w, c) / Total_Words_c
prob_0 = count_wc / total_words_c
log_prob_0 = np.where(prob_0 > 0, np.log(np.maximum(prob_0, 1e-12)), -1e9)  # ค่า 0 ให้ log(0) = -inf
acc_0 = accuracy_score(y_test_arr, predict_nb(log_prob_0))

# 3.2 สมการ Laplace Smoothing (+1): P(w|c) = (Count(w, c) + 1) / (Total_Words_c + |V|)
prob_1 = (count_wc + 1.0) / (total_words_c + 1.0 * V_size)
y_pred = predict_nb(np.log(prob_1))
acc_1 = accuracy_score(y_test_arr, y_pred)

# แสดงตารางตัวอย่างคำที่มีค่า 0 ในบางคลาส vs หลังบวก 1 ด้วย Laplace Smoothing
zero_cols = np.where((count_wc == 0).any(axis=0))[0]
sample_idx = zero_cols[np.argsort(count_wc[:, zero_cols].sum(axis=0))[::-1][:8]]
df_compare = pd.concat([
    pd.DataFrame(count_wc[:, sample_idx].T, index=vocab[sample_idx], columns=[f"TF_{s}ดาว" for s in stars]),
    pd.DataFrame((count_wc + 1)[:, sample_idx].T, index=vocab[sample_idx], columns=[f"Laplace(+1)_{s}ดาว" for s in stars])
], axis=1)

print("\n=== 2. ตัวอย่างการคำนวณ Laplace Smoothing จากสมการ (แก้ปัญหาความถี่เป็น 0) ===")
print(df_compare.to_string())

print(f"\n=== 3. ผลการคำนวณจากสมการ (เปรียบเทียบ Smoothing = 0 vs Laplace Smoothing = +1) ===")
print(f"• สมการ Smoothing = 0    : P(w|c) = Count(w,c) / Total_Words_c           -> Accuracy: {acc_0 * 100:.2f}% (ติดปัญหาค่า 0)")
print(f"• สมการ Laplace Smoothing : P(w|c) = (Count(w,c) + 1) / (Total_Words_c + |V|) -> Accuracy: {acc_1 * 100:.2f}% (แม่นยำขึ้น)\n")
print(classification_report(y_test_arr, y_pred, digits=4))

# ==============================================================================
# 4. Visualization Heat Map (ความน่าจะเป็น P(word|star) จากสมการ & Confusion Matrix)
# ==============================================================================
top_20 = np.argsort(X_train_tf.sum(axis=0).A1)[::-1][:20]
prob_df = pd.DataFrame(prob_1[:, top_20].T, index=vocab[top_20], columns=[f"{s} ดาว" for s in stars])

fig2, (ax_prob, ax_cm) = plt.subplots(1, 2, figsize=(15, 6.5), num="Window 2 - Naive Bayes Heat Map Visualizations")
sns.heatmap(prob_df, annot=True, fmt=".4f", cmap="Blues", linewidths=0.5, cbar_kws={"label": "P(word | Star)"}, ax=ax_prob)
ax_prob.set_title("Heat Map ความน่าจะเป็นคำศัพท์ P(word | Star) [Laplace +1]")
ax_prob.set_xlabel("ระดับดาว (Class)")
ax_prob.set_ylabel("คำศัพท์ (Word)")

sns.heatmap(confusion_matrix(y_test_arr, y_pred, labels=stars), annot=True, fmt="d", cmap="Blues",
            xticklabels=[f"{s} ดาว" for s in stars], yticklabels=[f"{s} ดาว" for s in stars],
            linewidths=0.5, cbar_kws={"label": "จำนวนรีวิว (Count)"}, ax=ax_cm)
ax_cm.set_title(f"Confusion Matrix Heat Map (Test Accuracy: {acc_1 * 100:.2f}%)")
ax_cm.set_xlabel("Predicted Rating (ระดับดาวที่ทำนาย)")
ax_cm.set_ylabel("Actual Rating (ระดับดาวจริง)")

plt.tight_layout()
fig2.savefig(os.path.join(BASE_DIR, "confusion_matrix_heatmap.png"), dpi=150)
plt.show()
