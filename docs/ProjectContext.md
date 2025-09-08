# Projektkontext

## Ziel und Anwendungsszenario
Dieses Demo-Repo zeigt die Klassifikation typischer HR-Chatnachrichten in fünf Kategorien (Leave, Payroll, Admin, Time, Benefits). Es dient als nachvollziehbarer Vergleich zwischen einer einfachen Bag-of-Words-Baseline (Naive Bayes) und einem robusterem Ansatz (BERT-Proxy via LinearSVC) und soll als Vorlage für Projekte rund um HR-Tickets oder Chatbots dienen.

## Synthetischer Datensatz
Die Daten werden vollständig synthetisch erzeugt. Das Skript [src/data_core_generate.py](../src/data_core_generate.py) nutzt Vorlagen und zufällige Perturbationen, um realistische, verrauschte Chatzeilen mit Labeln zu generieren. Zur Reproduzierbarkeit sind die Seeds fest auf `random.seed(73)` und `np.random.seed(73)` gesetzt; ebenso arbeiten das Train/Test-Split und die Modelle mit festen Seeds.

## Reproduzierbarkeit
Alle Resultate können mit einem einzigen Befehl neu erstellt werden:
```bash
python src/make_all_results.py
```
Dabei werden Datensatz, Modelle und Visualisierungen deterministisch neu aufgebaut.

## Lizenz und Datenschutz
Der gesamte Code steht unter der MIT-Lizenz. Da ausschließlich synthetische Daten verwendet werden, sind keine personenbezogenen Daten betroffen und es bestehen keine Datenschutzbedenken.
