
from pathlib import Path
import random, numpy as np, pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"; DATA.mkdir(exist_ok=True, parents=True)
RESULTS = BASE / "results"; RESULTS.mkdir(exist_ok=True, parents=True)

random.seed(73); np.random.seed(73)

classes = ["Leave","Payroll","Admin","Time","Benefits"]
size_plan = {"Leave":1200,"Payroll":1100,"Admin":1100,"Time":900,"Benefits":800}
shared = ["antrag","formular","zugang","system","info","hilfe","pruefen","problem","konto","bestaetigung","hinweis","link","ticket","vorgang","unklar"]
cross = {"urlaub":["urlaub","frei","abwesenheit"],
         "gehalt":["gehalt","lohn","abrechnung"],
         "vertrag":["vertrag","dokument","unterlagen"],
         "zeit":["zeit","stunden","erfassung"],
         "leistung":["leistung","benefit","vorteil"]}
templates = {
    "Leave":["{urlaub} beantragen","rest{urlaub} uebertragen","{urlaub} tage abfragen","{urlaub} nachtragen","{urlaub} abgelehnt"],
    "Payroll":["{gehalt} fehlt {monat}","{gehalt} stimmt nicht","bank fuer {gehalt} aendern","zulagen fehlen","steuerklasse aendern"],
    "Admin":["{vertrag} kopie","adresse aendern","bescheinigung fuer bank","probezeit regelung","vertrag verlaengern"],
    "Time":["ueberstunden abbuchen","kernarbeitszeiten","homeoffice {zeit} erfassen","{zeit} fehler","pause richtig buchen"],
    "Benefits":["{leistung} beantragen","welche {leistung} stehen mir zu","jobrad","altersvorsorge erklaeren","zuschuss fitness"]
}
months = ["januar","februar","maerz","april","mai","juni","juli","august","september","oktober","november","dezember"]
polite = [""," bitte"," danke"," hallo"," guten tag"," vielen dank"]
filler = [""," dringend"," zeitnah"," optional"," info benoetigt"," unklar"]

def perturb(base, drop_key_p=0.55, amb_p=0.95, typos=0.4):
    t = base.replace("{monat}", random.choice(months))
    for k,alts in cross.items():
        if "{"+k+"}" in t:
            t = t.replace("{"+k+"}", random.choice(alts))
    if random.random() < drop_key_p:
        parts = t.split()
        if len(parts)>1: t = " ".join(parts[1:])
    if random.random() < amb_p: t += " " + random.choice(shared)
    if random.random() < amb_p*0.9: t += " " + random.choice(shared)
    t += random.choice(polite) + random.choice(filler)
    if random.random() < typos and len(t)>6:
        i = random.randrange(1, len(t)-2); t = t[:i] + t[i] + t[i:]
    return t.strip()

def make_df(label_noise=0.34):
    rows = []
    for lbl, n in size_plan.items():
        for _ in range(n):
            base = random.choice(templates[lbl])
            txt = perturb(base)
            y = lbl if random.random() > label_noise else random.choice([c for c in classes if c!=lbl])
            rows.append({"text": txt, "label": y})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=73).reset_index(drop=True)
    return df

def clean(s):
    s = s.lower()
    for ch in "!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~0123456789":
        s = s.replace(ch, " ")
    while "  " in s: s = s.replace("  "," ")
    return s.strip()

if __name__ == "__main__":
    df = make_df()
    (DATA / "demo_synth_core.csv").write_text(df.to_csv(index=False))
    df["text_clean"] = df["text"].apply(clean)
    df.to_csv(RESULTS / "prepared_core.csv", index=False)
    print("core dataset ready")
