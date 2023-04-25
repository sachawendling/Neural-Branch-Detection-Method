# Neural Branch Detection Method

## Usage
```
python main.py
```

## Requirements
```
Python
OpenCV
Matplotlib
Numpy
Scipy
NetworkX
```
## Classes et fichiers :
```
skeleton .py :
La classe Skeleton a pour but de représenter le squelette d'une image binaire et de permettre d'effectuer des traitements dessus. 
La méthode __init__ est appelée lors de la création d'un nouvel objet de la classe Skeleton. Elle prend en argument une image binaire matrix et une instance de la classe Soma soma, puis initialise les différents attributs de l'objet.
La méthode plot permet d'afficher les points du squelette sur une fenêtre Matplotlib.
La méthode simplify permet de simplifier les lignes du squelette pour éviter d'avoir un surplus de points par endroits ce qui occasionne parfois des points de ramifications qui ont 4 voisins.
La méthode get_neighbors permet de récupérer les voisins d'un point donné sur une case adjacente, qui ne sont pas diagonaux si l'option excludeDiag est à True et qui ne sont pas dans la liste à exclure excludeThose.
La méthode get_branching_points permet de détecter les points de ramification du squelette, c'est-à-dire les points en lesquels une branche vient se séparer en deux. Cette méthode parcourt tous les points du squelette.

Branch.py :
-La classe Branch représente une branche du squelette. 
"init" : le constructeur de la classe qui initialise les propriétés de la branche.
    • "is_branching_out" : retourne True si le dernier point de la branche est un point de ramification du squelette.
    • "relier_centre" : prend en entrée les coordonnées du centre et trace un segment de ligne droite entre le premier point de la branche et le centre sur la carte de squelette.
    • "least_square_approximation" : calcule une fonction d'approximation des points de la branche avec une méthode des moindres carrés.
    • "plot_approximation" : calcule les points de la courbe paramétrique de l'approximation polynomiale avec une discrétisation de [0,1] et les coefficients de la fonction. Affiche la courbe.
    • "measure_average_thickness" : calcule l'épaisseur moyenne de la branche.
Lsq.py :
ce fichier définit trois fonctions: ConstructionMXB, LeastSquaresConstraintsMonomes et compute_parametric_curve.
La fonction ConstructionMXB prend en entrée une matrice A, une matrice F, un vecteur b et un vecteur c. Elle construit la matrice M et le vecteur B selon le schéma spécifié dans la docstring de la fonction et renvoie M et B.
La fonction LeastSquaresConstraintsMonomes prend en entrée deux vecteurs xi et yi, deux vecteurs xic et yic, un entier degree et un vecteur t. Elle calcule le polynôme d'approximation aux moindres carrés p(t) de degré degree approchant les données (xi, yi) sous la contrainte de passer par les points (xic, yic) en utilisant la base des monômes. Elle renvoie le vecteur des coefficients du polynôme.
La fonction compute_parametric_curve prend en entrée le vecteur des coefficients cf et le vecteur t. Elle calcule la courbe paramétrique correspondante au polynôme représenté par cf évalué sur t en utilisant la méthode de Horner et renvoie le vecteur pt.
```
## Etude de complexité
```   
Lsq.py :

Le code a deux fonctions principales: ConstructionMXB et LeastSquaresConstraintsMonomes, qui sont appelées l'une par l'autre.
La complexité de la fonction ConstructionMXB dépend principalement de 
la complexité de l'opération matricielle np.dot(A.T, A). 
Cette opération est de complexité O(n^2p), où n est le nombre de lignes dans A 
et p le nombre de colonnes. 
La complexité totale de ConstructionMXB est donc de O(n^2p).
La complexité de la fonction LeastSquaresConstraintsMonomes dépend principalement
de deux opérations matricielles : la construction de la matrice A(O(n^2) dans le pire des cas,)
(de taille nblin1 x nbcol) et la construction de la matrice F (O(n^2) dans le pire des cas,)
(de taille nblin2 x nbcol).
La fonction compute_parametric_curve est de complexité O(degree n) car elle effectue une évaluation d'un polynôme de degré "degree" en n points de t.
L'appel a la fonction ConstructionMXB est également de complexité O(n^2p)
donc la complexite du code de ce fichier est de O(n^2p)


Branch.py :

parametric_linear_interpolation(points):la complexité totale de cette fonction O(n).
Branch.__init__(self, points, branching_points):ne contient que des opérations
  en temps constant, donc sa complexité est O(1).
Branch.is_branching_out():elle ne parcourt qu'un seul point, sa complexité est O(1).
Branch.least_square_approximation(): cette fonction effectue une 
  approximation des moindres carrés d'une courbe polynomiale à partir d'une liste
  de points. Le degré est fixé à 8, ce qui signifie que la complexité est
  de l'ordre de O(n^3) mais, comme le nombre de points de contrainte est faible (2 point: depart et arrivee ),
  la complexité totale reste relativement faible.
 

skeleton .py :

get_neighbors(): a une simple complexite de O(1) car elle fait une simple recherche dans une liste
get_branching_points(): fait appel a get_neighbors() pour chaque point du squelette
dans une boucle 'for' avec une complexité de O(n^2). la boucle 'for k in range(10)' a une complexité de O(1).
globalement cette methode a une complexité de O(n^2)
Les autres méthodes, comme  __init__() et simplify(), ont une complexité de O(n) 
ou moins, car elles parcourent chaque élément de la matrice une fois.
le reste du code consiste a des appels simple avec une complexité de O(1) donc négligeable.

main.py:

'find_noyau': contient une boucle for qui parcourt tous les contours trouvés dans l'image et calcule leur centroïde. À chaque contour, la fonction exécute une série d'opérations, notamment la création d'un masque et le calcul des pixels adjacents. La complexité de cette fonction dépend donc du nombre de contours détectés dans l'image et de leur taille. Si l'image contient n contours et que le contour le plus grand a m pixels, alors la complexité de cette fonction est O(n * m).

'afficher_image': charge l'image et l'affiche à l'aide de la bibliothèque Tkinter. La complexité de cette fonction dépend de la taille de l'image, mais dans l'ensemble, elle est relativement faible.

'code': charge l'image et la redimensionne, puis applique un filtre de flou. La complexité de cette fonction dépend donc de la taille de l'image et de la taille du filtre de flou. En général, la complexité de cette fonction est O(n^2), où n est la largeur ou la hauteur de l'image redimensionnée.
```
