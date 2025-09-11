# Anonymizer 

Un projet Python pour l'anonymisation de documents XML juridiques en français, utilisant des modèles de langue pour identifier et remplacer les informations personnellement identifiables (PII).

## 🔧 Prérequis

- Python 3.8 ou supérieur
- CUDA (optionnel, pour l'accélération GPU)
- Accès à une API OpenAI ou compatible

## 📦 Installation

1. Clonez le dépôt :
```bash
git clone <url-du-depot>
cd anony
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

1. Copiez le fichier de configuration exemple :
```bash
cp .env.example .env
```

2. Éditez le fichier `.env` avec vos paramètres :
```bash
# API OpenAI ou compatible
OPENAI_API_KEY=votre-clé-api
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# Modèle d'enseignant (optionnel)
TEACHER_API_KEY=votre-clé-api
TEACHER_BASE_URL=https://api.openai.com/v1
TEACHER_MODEL=gpt-4
```

## 🚀 Utilisation

### Anonymisation de base

```bash
python anonymize.py --input document.xml --output document_anonymized.xml
```

### Avec modèle personnalisé

```bash
python anonymize.py --input document.xml --output document_anonymized.xml --model eternisai/Anonymizer-4B
```

### Avec règles personnalisées

```bash
python anonymize.py --input document.xml --output document_anonymized.xml --rules mes_regles.txt
```

## 📋 Paramètres disponibles

- `--input` : Fichier d'entrée (XML)
- `--output` : Fichier de sortie (XML anonymisé)
- `--model` : Modèle HuggingFace à utiliser (défaut: eternisai/Anonymizer-4B)
- `--rules` : Fichier contenant les règles d'anonymisation
- `--timeout` : Timeout pour les appels API (défaut: 60s)

## 🧠 Fonctionnement

Le système utilise deux approches :

1. **Modèle local** : Utilise un modèle HuggingFace (eternisai/Anonymizer-4B) pour l'anonymisation
2. **API externe** : Peut utiliser une API OpenAI/compatible comme modèle "enseignant"

### Règles d'anonymisation

Le système applique des règles spécifiques pour :
- **Noms personnels** : Remplace par des équivalents fictifs
- **Organisations** : Remplace les entités privées/niche
- **Lieux** : Remplace les adresses et petites villes
- **Dates** : Décale légèrement en conservant l'année
- **Identifiants** : Remplace par des formats valides fictifs
- **Valeurs monétaires** : Multiplie par un facteur 0.8-1.25

## 📁 Structure du projet

```
anony/
├── anonymize.py          # Script principal d'anonymisation
├── requirements.txt      # Dépendances Python
├── README.md            # Documentation
├── .env.example         # Template de configuration
├── .gitignore          # Fichiers à ignorer par Git
├── data/               # Dossier pour les données d'entrée
└── out_hf/            # Dossier pour les sorties
```

## 🔒 Sécurité

- Ne commitez jamais vos clés API dans le dépôt
- Utilisez le fichier `.env` pour vos configurations sensibles
- Le fichier `.env` est automatiquement ignoré par Git

## 🤝 Contribution

Les contributions sont les bienvenues ! Veuillez :

1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Committer vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence libre. Voir le fichier `LICENSE` pour plus de détails.