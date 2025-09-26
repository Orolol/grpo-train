Voici les règles d'anonymisation :

Principe général : l'anonymisation concerne les personnes physiques

Pas d'anonymisation pour :
- la composition de la cour,
- les noms de société ou personnes morales
- le greffier, le président
- les noms des institutions publiques tel que les hopitaux, centres d'administration publiques, écoles/collèges/lycées/universitéss etc.
- les sommes et montants

Noms et prénoms des personnes physiques
- L'anonymisation des personnes physiques concerne toute personne physique identifiée dans le texte à l'exception des personnes citées ci-dessus.
- De règle générale tous les noms et prénoms doivent être anonymisés à l'exception des personnes citées ci-dessus.
- Les règles d'anonymisation des noms et prénoms des personnes physique sont les suivantes :
  -- De règle générale, les noms et prénoms des personnes physique doivent être anonymisés par les initiales des noms et prénoms, par exemple "Jean DUPONT" devient "J D", "Laure BUFFET" devient "L B"
  -- Les prénoms seuls, par exemple "Arthur" devient "A", "Clothilde" devient "C"
  -- Les noms de famille seuls, par exemple "Monsieur DUPONT" devient "Monsieur D", "Madame MARTIN" devient "Mme M"
  -- Les noms seuls sans civilité, par exemple "DUPONT" devient "D", "MARTIN" devient "M"
  -- Les noms composés, par exemple "Paul-Marie NGUYEN" devient "P-M N".
  -- Les noms à particules, par exemple "Sigmund de la Fontaine" devient "J d l F". Les particules sont : "de", "du", "d'", "des", "la", "le", "l'", "les", "au", "aux", "à".
  -- Les deuxième et troisième prénoms etc, par exemple "Philippe Henri MARCADET" devient "P H M". Ou "Sophie, Anne-Claire, Élisabeth Martin" devient "S, A-C, E M".
  -- Les noms de jeune fille, par exemple "Esther WILLIAMS née HUGUES" devient "E W née H".
  -- Les noms d'épouse, par exemple "Madame Mélodie JACQUARD épouse BIGART" devient "Mme M J épouse B".
  -- Les groupes des personnes avec le même nom de famille, par exemple "Olivier et Julien QUENON" deviennent "O et J Q".
- Une même personne physique peut être citée plusieurs fois dans le texte, il faut donc anonymiser toutes les occurrences de cette personne physique.
- Une même personne physique doit restée identifiable tout au long du texte et une même personne doit être anonymisée de la même manière.
- Les noms et prénoms d'une même personne physique peuvent être écrits de différentes manières, il faut donc faire attention à anonymiser les différentes formes de la même manière.
- Faire très attention aux cas suivants :
  -- les prénoms et noms peuvent être inversés, par exemple "Jean DUPONT" et "DUPONT Jean" réfèrent à la même personne et doivent être anonymisée de la même manière : "J D".
  -- attention aux orthographes qui peuvent manquer d'accents, par exemple "Hélène LÉVÊQUE", "Hélène LEVEQUE" et "Helene LEVEQUE" réfèrent à la même personne et doivent être anonymisés de la même manière : "H L".
  -- attention aux différents varients d'orthographe des noms d'une même personne, par exemple :
    --- "Jean-Pierre RIBOULAIX" et "Jean Pierre RIBOULAIX" réfèrent à la même personne et doivent être anonymisés de la même manière : "J-P R".
    --- "Linda ZANTOLI" et "Lynda ZANTOLI" réfèrent à la même personne et doivent être anonymisés de la même manière : "L Z".
    --- "Remi LESECHE" et "Rémy LESECHE" réfèrent à la même personne et doivent être anonymisés de la même manière : "R L".
- La civilité et d'autres dénominations anciennes (par exemple "sieur") ne doit pas être anonymisée, par exemple "Monsieur Albert OCCATIA" devient "Monsieur A O". "Mme. Laëtitia ROUX" devient "Mme. L R".
- Parfois les noms et prénoms sont inversés, par exemple "Walter TREMOINS" devient "W T" ou "TREMOINS Walter" devient "T W".

Avant de produire le JSON, il est impératif de suivre cette démarche :
1. **Identifier toutes les personnes physiques** mentionnées dans le texte.
2. **Lister tous les noms et alias** pour chaque personne (ex: nom complet, nom de famille seul, nom d'épouse, etc.).
3. **Assurer la cohérence** : une même personne, même si elle est désignée par des noms différents (ex: "Jessica LEBLOND" et "Jessica CUFFEL"), doit être anonymisée de manière cohérente. La règle est de conserver les initiales de chaque partie du nom tel qu'il apparait dans le texte. Par exemple, "Madame Jessica LEBLOND épouse CUFFEL" devient "Mme J L épouse C". Si plus loin dans le texte on trouve "Madame CUFFEL", on anonymisera par "Mme C".

**Attention, cette règle de cohérence est cruciale.** L'objectif est qu'une personne reste identifiable sous une même forme anonymisée *lorsque le nom utilisé pour la désigner est le même*.

Coordonnées des personnes identifiées
- Il faut anonymiser les coordonnées personnelles des personnes identifiées ci-dessus:
- L'anonymisation des coordonnées concerne :
  -- Les adresses des demeures, des résidences principales et secondaires, des locations, les lieux de travail
  -- Les adresses partielles
  -- Les codes postaux, les villes, les communes, les lieux dites et les départements
  -- Les numéros de téléphone fixe, mobile et portable
  -- Les adresses mail
  -- Les nom d'utilisateur et identifiants des médias sociaux et des sites internet
- Les règles d'anonymisation des coordonnées des personnes physiques sont les suivantes :
  -- les adresses doivent être anonymisées par "[Adresse]", par exemple :
    --- "37 rue de la Général Leclerc, 75016 Paris" devient "[Adresse]"
    --- "Les petits écuries, rue des Charmes, 52260 Chanoy" devient "[Adresse]"
  -- Les codes postaux, les villes, les communes, les lieux dites et les départements seuls doivent être anonymisés par "[Localité]", par exemple :
    --- "14700 FALAISE" devient "[Localité]"
    --- "CHAILLOUE (61)" devient "[Localité]"
    --- "La Ferme Hirondelle" devient "[Localité]"
  -- Les numéros de téléphone fixe, mobile et portable doivent être anonymisés par "[Téléphone]", par exemple :
    --- "01 23 45 67 89" devient "[Téléphone]"
    --- "+33 6 98 76 54 32" devient "[Téléphone]"
    --- "+44 (0)118 123 4567" devient "[Téléphone]"
  -- Les adresses mail doivent être anonymisées par "[Courriel]", par exemple :
    --- "fred.zaid@exemple.com" devient "[Courriel]"
  -- Les nom d'utilisateur et identifiants des médias sociaux et des sites internet doivent être anonymisés par "[Identifiant]", par exemple :
    --- "damien.tarieux" devient "[Identifiant]"
    --- "@maelstom569" devient "[Identifiant]"

Dates liées aux personnes identifiées
- L'anonymisation des dates concerne les dates de naissance, de mariage, de divorce et de décès des personnes identifiées ci-dessus.
- L'anonymisation des dates ne concerne pas les dates de début de procès, de jugement, de sentence, etc.
- Les règles d'anonymisation des dates sont les suivantes :
  -- Les dates doivent être remplacées par "[Date]", par exemple :
    --- "26 mai 1980" devient "[Date]"
    --- "1er janvier 2000" devient "[Date]"
    --- "15/08/1945" devient "[Date]"

Lieux liés aux personnes identifiées
- L'anonymisation des lieux concerne les lieux de naissance, de mariage, de divorce et de décès des personnes identifiées ci-dessus.
- Les règles d'anonymisation des lieux sont les suivantes :
  -- Les lieux doivent être remplacés par "[Localité]", par exemple :

Autres données personnelles des personnes identifiées
- Plaque d'immatriculation : [Immatriculation]
- Numéro sécurité sociale : [N° SS]
- Numéros de documents officiels : [Numéro]
- Numéros d'aide juridictionnelle Totale : [Numéro]
- Compte bancaire : [IBAN]

A noter : s'il existe plusieurs éléments à anonymiser du même type, il faut numéroter le terme générique de manière à conserver la bonne compréhension tout au long du texte.
Ex : [Adresse 1], [Adresse 2], [Localité 1], [Localité 2] etc.

Si aucune anonymisation n'est possible ou si aucune n'est a faire, ne pas ajouter de diff.

Ne pas produire de diff identique, un seul par texte a remplacer, éviter les doublons qui ne sont pas utile.

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

Le texte original à rechercher doit contenir que le texte à remplacer : il sera remplacé en entier.
Le texte original à rechercher doit obligatoirement exister dans le texte : n'inventez pas de texte à anonymiser qui n'existe pas dans le texte.
Le texte anonymisé doit contenir uniquement le texte anonymisé.
Pour le motif, fournir le motif d'anonymisation

**Gestion des ambiguïtés**
Si le texte original est susceptible d'être ambigu (par exemple, un nom de personne qui est aussi un nom de lieu comme "Dijon"), il est **obligatoire** d'ajouter du contexte dans le champ `search` pour que la recherche soit précise et ne provoque pas d'effets de bord. Dans ce cas, il faut créer autant d'entrées `diff` que nécessaire pour traiter tous les cas sans erreur.

Le diff produit doit appliquer strictement les règles d'anonymisation.
