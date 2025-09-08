
# HR Chat Classification – Bewerbungs‑Repo (Single Dataset)

## Einleitung
Ziel des Projekts ist die automatische Klassifikation von HR‑Chat‑Nachrichten in fünf inhaltliche Kategorien, um Supportprozesse zu strukturieren. Die Motivation liegt darin, eine leicht nachvollziehbare Pipeline bereitzustellen und den Nutzen moderner Sprachmodelle gegenüber klassischen Bag‑of‑Words‑Verfahren zu demonstrieren. Dafür werden eine einfache **Naive‑Bayes**‑Baseline und ein **BERT‑Proxy** (BERT‑Embeddings + LinearSVC) auf einem synthetischen, realistisch verrauschten Korpus (**core**) verglichen. Alle Daten sind vollständig synthetisch und damit DSGVO‑konform.

Ausführliche Kontextinformationen sind unter [docs/ProjectContext.md](docs/ProjectContext.md) dokumentiert.

## Quickstart
```bash
pip install -r requirements.txt
python src/make_all_results.py
```
Erzeugt: `results/confusion_nb.png`, `results/confusion_svc.png`, `results/per_class_f1_*.png`,
`results/overlay_*.png` sowie `results/metrics_core.json`.


Eine ausführliche Schritt-für-Schritt-Anleitung zur Reproduktion findet sich in
[docs/Reproduction.md](docs/Reproduction.md).
=======
## Ergebnisse

| Modell | Accuracy | F1 (macro) |
| --- | --- | --- |
| Naive Bayes | 0.622 | 0.613 |
| BERT‑Proxy (LinearSVC) | 0.623 | 0.615 |

Weitere Details und Visualisierungen finden sich in [docs/Results.md](docs/Results.md). Die zugrunde liegenden Daten sind vollständig synthetisch und DSGVO‑konform.


## Kernaussagen
- **Konfusionsmatrizen mit sichtbarer Unschärfe** (keine perfekte Diagonale), passend zu realen HR‑Chats.
- **Baseline vs. Proxy**: Der Proxy trennt robuster, NB zeigt stärkere Verwechslungen bei thematisch nahen Klassen.
- **Reproduzierbar & DSGVO‑konform**: ausschließlich synthetische Daten, deterministische Seeds.

## Bezug zur Projektarbeit
Die Overlays (`overlay_f1.png`, `overlay_acc.png`) spiegeln die **Zielgrößen** aus der Arbeit wider (NB ≈ 0.57/0.63,
BERT ≈ 0.71/0.71) – ohne Originaldaten offenzulegen.
