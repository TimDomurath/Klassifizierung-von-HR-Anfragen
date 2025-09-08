# Reproduktion

Diese Anleitung beschreibt, wie sich die Ergebnisse dieses Repos vollständig neu erzeugen lassen.

## Voraussetzungen
- Python 3.10 oder neuer (getestet mit 3.12)
- Keine spezielle Hardware notwendig; CPU genügt.

## Installation
```bash
pip install -r requirements.txt
```

## Daten generieren & Modelle ausführen
Das Skript `src/make_all_results.py` erzeugt den synthetischen Korpus, trainiert Naive Bayes und LinearSVC und legt Kennzahlen sowie Plots ab.
```bash
python src/make_all_results.py
```

## Artefakte
- `results/`: CSVs, Metriken und Konfusionsmatrizen.
- `docs/figures/`: Kopien der relevanten Grafiken für die Dokumentation.

Die Seeds sind fest gesetzt; erneute Durchläufe liefern identische Resultate.
