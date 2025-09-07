
from pathlib import Path
import json, numpy as np, matplotlib.pyplot as plt, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"; RESULTS.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(RESULTS / "prepared_core.csv")
X_train, X_test, y_train, y_test = train_test_split(df["text_clean"], df["label"], test_size=0.25, random_state=42, stratify=df["label"])

nb = Pipeline([("vec", CountVectorizer(ngram_range=(1,1), max_df=0.8, min_df=6)),
               ("tfidf", TfidfTransformer()),
               ("clf", MultinomialNB(alpha=1.2))])
svc = Pipeline([("vec", CountVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=3)),
                ("tfidf", TfidfTransformer()),
                ("clf", LinearSVC(C=1.2, random_state=42))])

nb.fit(X_train, y_train); svc.fit(X_train, y_train)
y_nb = nb.predict(X_test); y_svc = svc.predict(X_test)

metrics = {"NB":{"accuracy": float(accuracy_score(y_test, y_nb)),
                 "f1_macro": float(f1_score(y_test, y_nb, average="macro"))},
           "BERT_Proxy":{"accuracy": float(accuracy_score(y_test, y_svc)),
                         "f1_macro": float(f1_score(y_test, y_svc, average="macro"))}}
(RESULTS / "metrics_core.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

labels = sorted(df["label"].unique())
def cmplot(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6.6,5.6))
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    ax.set_title(title); plt.tight_layout(); plt.savefig(RESULTS / fname, dpi=170); plt.close()

def bar_per_class(y_true, y_pred, title, fname):
    from sklearn.metrics import f1_score
    scores = f1_score(y_true, y_pred, average=None, labels=labels)
    x = np.arange(len(labels))
    plt.figure(figsize=(7.8,4.2))
    plt.bar(x, scores); plt.xticks(x, labels, rotation=15)
    plt.ylim(0,1.0)
    for i,v in enumerate(scores): plt.text(i, v+0.015, f"{v:.2f}", ha="center")
    plt.ylabel("F1"); plt.title(title); plt.tight_layout()
    plt.savefig(RESULTS / fname, dpi=170); plt.close()

cmplot(y_test, y_nb, "Konfusionsmatrix – NB (core)", "confusion_nb.png")
cmplot(y_test, y_svc, "Konfusionsmatrix – BERT‑Proxy (core)", "confusion_svc.png")
bar_per_class(y_test, y_nb, "F1 je Klasse – NB (core)", "per_class_f1_nb.png")
bar_per_class(y_test, y_svc, "F1 je Klasse – BERT‑Proxy (core)", "per_class_f1_svc.png")

# Overlay vs. Zielwerte (Studie)
thesis_f1 = {"NB": 0.63, "BERT": 0.71}
thesis_acc = {"NB": 0.57, "BERT": 0.71}
demo_f1 = {"NB": metrics["NB"]["f1_macro"], "BERT": metrics["BERT_Proxy"]["f1_macro"]}
demo_acc = {"NB": metrics["NB"]["accuracy"], "BERT": metrics["BERT_Proxy"]["accuracy"]}

def overlay_plot(demo, target, ylabel, title, fname):
    labs = ["NB","BERT"]; x = np.arange(len(labs)); w=0.35
    plt.figure(figsize=(7.2,4.8))
    plt.bar(x - w/2, [demo[l] for l in labs], width=w, label="Demo (core)")
    plt.bar(x + w/2, [target[l] for l in labs], width=w, label="Thesis (Ziel)")
    plt.ylim(0,1.0); plt.xticks(x, labs); plt.ylabel(ylabel); plt.title(title)
    for i,l in enumerate(labs):
        plt.text(i - w/2, demo[l]+0.015, f"{demo[l]:.2f}", ha="center")
        plt.text(i + w/2, target[l]+0.015, f"{target[l]:.2f}", ha="center")
    plt.legend(); plt.tight_layout(); plt.savefig(RESULTS / fname, dpi=170); plt.close()

overlay_plot(demo_f1, thesis_f1, "F1‑macro", "F1: Demo (core) vs. Thesis", "overlay_f1.png")
overlay_plot(demo_acc, thesis_acc, "Accuracy", "Accuracy: Demo (core) vs. Thesis", "overlay_acc.png")

print("core metrics & figures ready")
