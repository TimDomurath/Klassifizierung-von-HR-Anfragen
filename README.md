
# HR Chat Classification – Bewerbungs‑Repo (Single Dataset)

Dieses Repo zeigt eine reproduzierbare Textklassifikation (5 HR‑Kategorien) auf **einem** synthetischen,
realistisch verrauschten Korpus (**core**). Ziel ist die **nachvollziehbare Gegenüberstellung** einer
klassischen Bag‑of‑Words‑Baseline (**Naive Bayes**) mit einem robusteren Ansatz (**BERT‑Proxy via LinearSVC**).

## Quickstart
```bash
pip install -r requirements.txt
python src/make_all_results.py
```
Erzeugt: `results/confusion_nb.png`, `results/confusion_svc.png`, `results/per_class_f1_*.png`,
`results/overlay_*.png` sowie `results/metrics_core.json`.

## Kernaussagen
- **Konfusionsmatrizen mit sichtbarer Unschärfe** (keine perfekte Diagonale), passend zu realen HR‑Chats.
- **Baseline vs. Proxy**: Der Proxy trennt robuster, NB zeigt stärkere Verwechslungen bei thematisch nahen Klassen.
- **Reproduzierbar & DSGVO‑konform**: ausschließlich synthetische Daten, deterministische Seeds.

## Bezug zur Projektarbeit
Die Overlays (`overlay_f1.png`, `overlay_acc.png`) spiegeln die **Zielgrößen** aus der Arbeit wider (NB ≈ 0.57/0.63,
BERT ≈ 0.71/0.71) – ohne Originaldaten offenzulegen.
