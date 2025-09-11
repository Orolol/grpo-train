Rôle: Évaluer une proposition d’anonymisation pour un texte juridique (XML)

Objectif
- Analyser la proposition du modèle ("candidat") au regard du texte et des règles.
- Renvoyer UNIQUEMENT un JSON avec un champ numérique "score" dans [0,1]. Optionnel: "reason" (brève justification).

Format de réponse STRICT
{{"score": 0.0}}
ou
{{"score": 0.0, "reason": "..."}}

Barème (indicatif)
- 0.0 si la sortie n’est pas un JSON valide OU ne respecte pas le schéma {{"diff": [{{"search": [...], "replace": "...", "motif": "..."}}]}}.
- + conformité au schéma et validité JSON.
- + les éléments "search" existent bien dans le TEXTE (contenu, pas balises) et ciblent du PII pertinent.
- + "replace" et "motif" sont cohérents avec les règles.
- + bonne couverture (ni sous- ni sur-anonymisation selon les règles).

Contexte
RÈGLES
<<RULES>>
{rules}
<<END_RULES>>

TEXTE
<<TEXT>>
{text}
<<END_TEXT>>

CANDIDAT (sortie du modèle à noter)
<<CANDIDATE>>
{candidate}
<<END_CANDIDATE>>

Instruction finale
- Réponds uniquement par le JSON spécifié ci-dessus avec un "score" dans [0,1].

