from flask import Flask, request, jsonify
import joblib
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)
app.json.ensure_ascii = False


model = joblib.load("model.pkl")
word_vectorizer = joblib.load("word_vectorizer.pkl")
char_vectorizer = joblib.load("char_vectorizer.pkl")
def clean_text(t):
    if t is None:
        return ""
    t = str(t).lower()
    t = t.replace('0', 'o').replace('1', 'i').replace('3', 'e').replace('4', 'a')
    t = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

INCIDENT_KEYWORDS = [
    "erreur", "panne", "bug", "bloqué", "bloque", "crash", "lent",
    "impossible", "ne fonctionne", "ne marche", "ne s'ouvre",
    "ne répond", "ne démarre", "écran noir", "écran bleu",
    "plantage", "dysfonctionnement", "défaillance", "coupé",
    "interrompu", "perdu", "corrompu", "endommagé", "détecté",
    "vulnérabilité", "violation", "incident", "problème",
    "signaler", "urgence", "urgent", "critique", "sévère",
    "tombé", "tombe", "redémarre", "freeze", "gel", "gelé",
    "ralenti", "surchauffe", "bruit", "casse", "cassé",
    "infecté", "virus", "malware", "attaque", "fuite",
    "échec", "échoué", "refusé", "timeout", "expir",
    "inaccessible", "indisponible", "hors service",
    "ne charge pas", "ne se connecte", "déconnecté",
    "incohérence", "retard", "irrégulier",
]

DEMANDE_KEYWORDS = [
    "demande", "besoin", "installer", "installation", "configurer",
    "configuration", "créer", "création", "ajouter", "ajout",
    "nouveau", "nouvelle", "accès", "autorisation", "permission",
    "licence", "abonnement", "mise à jour", "mise à niveau",
    "upgrade", "migration", "déployer", "déploiement",
    "commander", "commande", "fournir", "obtenir",
    "renseigne", "information", "renseignement",
    "souhaite", "souhaiterais", "voudrais", "voudrait",
    "pourriez", "pouvez", "possible", "envisager",
    "recommandation", "suggestion", "conseil",
    "optimiser", "améliorer", "amélioration", "extension",
    "intégration", "intégrer", "automatiser",
    "former", "formation", "tutoriel",
    "planifier", "prévoir", "budget",
    "proposer", "proposition", "devis",
    "remplacer", "remplacement", "changer", "changement",
    "ajuster", "adapter", "personnaliser",
]
def extract_keyword_features(texts):
    """Extract keyword-based features for each text."""
    features = []
    for t in texts:
        t_lower = t.lower()
        incident_count = sum(1 for kw in INCIDENT_KEYWORDS if kw in t_lower)
        demande_count = sum(1 for kw in DEMANDE_KEYWORDS if kw in t_lower)

        total = incident_count + demande_count + 1  # +1 to avoid div by zero
        incident_ratio = incident_count / total
        demande_ratio = demande_count / total


        negation = len(re.findall(r'ne\s+\w+\s+pas|impossible|aucun|pas\s+de|plus\s+de', t_lower))


        question = t.count('?') + len(re.findall(r'pourriez|pouvez|serait|possible|svp|merci', t_lower))


        urgency = len(re.findall(r'urgent|critique|immédiat|asap|rapidement|toute urgence|priorit', t_lower))


        length = len(t)

        features.append([
            incident_count, demande_count,
            incident_ratio, demande_ratio,
            negation, question, urgency,
            length,
        ])

    return np.array(features)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    message = data.get("email", "")
    subject = data.get("subject", "")

    clean_msg = clean_text(message)
    clean_sub = clean_text(subject)

    word_feat = word_vectorizer.transform([clean_msg])
    char_feat = char_vectorizer.transform([clean_msg])
    kw_feat = extract_keyword_features([clean_msg + " " + clean_sub])

    combined = hstack([word_feat, char_feat, csr_matrix(kw_feat)])

    pred = model.predict(combined)[0]
    proba = model.predict_proba(combined)[0]
    score = float(max(proba))

    return jsonify({
        "Sujet": subject,
        "Message": message,
        "Type": pred,
        "Score": score
    })

if __name__ == '__main__':
    app.run(debug=True)