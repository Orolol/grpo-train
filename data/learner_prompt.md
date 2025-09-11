Tâche: anonymisation de texte juridique (XML)

Consignes
- Retourne UNIQUEMENT un JSON valide respectant strictement ce schéma:
  {{
    "diff": [
      {{ "search": ["..."], "replace": "...", "motif": "..." }}
    ]
  }}
- Aucune explication, aucun commentaire, aucun bloc de code, aucun texte hors JSON.
- Les chaînes de "search" DOIVENT exister dans le TEXTE (contenu textuel) et viser le contenu, pas les balises XML.
- "replace" et "motif" doivent respecter les règles.
- N’ajoute aucune clé supplémentaire; respecte la casse JSON standard.

Règles
<<RULES>>
{rules}
<<END_RULES>>

Texte à anonymiser
<<TEXT>>
{text}
<<END_TEXT>>

Sortie attendue
- Uniquement le JSON décrit ci-dessus.

