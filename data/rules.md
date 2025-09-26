Principe général : l’anonymisation concerne les personnes physiques

Pas d’anonymisation pour la composition de la cour, pas d’anonymisation non plus pour les noms de société
Reprendre la première lettre pourrait poser soucis / noms qui auraient les mêmes initiales (exemple Monsieur et Madame ont un prénom commençant par la même lettre)
Une même personne doit restée identifiable tout au long de l’arrêt et les noms de famille doivent être traduit par une même lettre
- Reprendre les initiales des prénoms/noms
- Si deux personnes ont des initiales identiques alors ajouter une lettre de plus sur le prénom.
- Si seul le nom est présent, alors remplacer par la première lettre du nom.
- Attention au nom et aux dénominations en ancien français, les formulations type "sieur" et les noms a particules précèdent le nom, qu'il faut anonymiser

Coordonnées 
- Adresse : remplacer par [Adresse] 
- Nom de ville : [Localité]
- Numéro de téléphone : [Téléphone]
- Adresse mail : [Courriel]

Date de naissance, de mariage, de décès (et uniquement ces dates là. Pas de date de début de procès, de jugement, de sentence, etc.)
- [Date] année

 	Lieu de naissance 
- [Localité]

Autres données personnelles 
- Plaque d’immatriculation : [Immatriculation]      
- Numéro sécurité sociale : [N° SS]
- Numéros de documents officiels : [nbre de caractères remplacés par X]
- Compte bancaire : [IBAN]

A noter : s’il existe plusieurs éléments à anonymiser du même type, il faut numéroter le terme générique de manière à conserver la bonne compréhension tout au long du texte.
Ex :  [Adresse 1], [Adresse 2] 


Peux-tu me fournir un diff au format JSON suivant pour anonymiser le texte :
{
    "diff": [
        {
            "search": [
                "<texte original à rechercher>",
                "<texte original à rechercher>",
                ... // Ajouter autant de textes à rechercher que nécessaire
            ],
            "replace": "<texte anonymisé>",
            "motif": "<motif de l'anonymisation>"
        },
        ... // Ajouter autant de diff que de texte à anonymiser
    ]
}

Exemples d'anonymisation


{
    "diff": [
        {
            "search": [
                "Antoine Marchand"
            ],
            "replace": "A M",
            "motif": "initiales personne physique"
        },
        {
            "search": [
                "Claire Rousseau"
            ],
            "replace": "C R",
            "motif": "initiales personne physique"
        },
        {
            "search": [
                "05/05/1969"
            ],
            "replace": "[Date 1] ann\u00e9e",
            "motif": "date de naissance"
        },
        {
            "search": [
                "18/12/1975"
            ],
            "replace": "[Date 2] ann\u00e9e",
            "motif": "date de naissance"
        },
        {
            "search": [
                "103 rue de Solferino, 59160 Lille"
            ],
            "replace": "[Adresse 1]",
            "motif": "adresse"
        },
        {
            "search": [
                "55 rue Nationale, 59000 Lille"
            ],
            "replace": "[Adresse 2]",
            "motif": "adresse"
        },
        {
            "search": [
                "06 82 50 37 93"
            ],
            "replace": "[T\u00e9l\u00e9phone 1]",
            "motif": "num\u00e9ro de t\u00e9l\u00e9phone"
        },
        {
            "search": [
                "06 73 60 92 68"
            ],
            "replace": "[T\u00e9l\u00e9phone 2]",
            "motif": "num\u00e9ro de t\u00e9l\u00e9phone"
        },
        {
            "search": [
                "antoine.marchand@exemple.fr"
            ],
            "replace": "[Courriel 1]",
            "motif": "adresse mail"
        },
        {
            "search": [
                "claire.rousseau@exemple.fr"
            ],
            "replace": "[Courriel 2]",
            "motif": "adresse mail"
        }
    ]
}