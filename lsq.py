import numpy as np
import matplotlib.pyplot as plt

def ConstructionMXB(A, F, b, c):
    """
        Construction selon le schéma suivant : 
            ( A^T A | F^T)  ( x )   (A^T b) 
            ( -----------)  (---) = (-----) 
            (   F   | 0  )  ( L )   (  c  ) 
        noté:       M         X   =    B
        A est une matrice (n, p) avec n > p
        F est une matrice (m, p) avec m < p
        b est un vecteur de R^n
        c est un vecteur de R^m
        Output : 
            M matrice (p+m, p+m)
            B vecteur de R^(p+m)
    """
    # matrice M 
    # à gauche en haut :
    M1 = np.dot(A.T, A)

    # à gauche :
    Mleft = np.vstack((M1,F))

    # à droite :
    f1,f2 = np.shape(F)
    Mright = np.vstack((F.T, np.zeros((f1, f1))))

    # M total :
    M = np.hstack((Mleft, Mright))

    # second membre
    B1 = np.dot(A.T, b)
    B = np.vstack((B1, c))
    return M, B

def LeastSquaresConstraintsMonomes(xi, yi, xic, yic, degree, t):
    """ Determination du polynome d'approximation aux Moindres carrés p(t) 
        de degré "degree" (dans la base des monomes) 
        approchant les données (xi, yi) 
        sous la contrainte de passer par les points (xic, yic)
        Returns :
            pt = p(t) : image of vector t
    """

    nblin1 = np.size(xi)
    nbcol = degree + 1

    # matrix A d'approximation (monomial basis)
    A = np.ones((nblin1, nbcol))
    for k in range(1, nbcol):
        A[:, k] = A[:, k-1] * xi

    # matrix F des contraintes (monomial basis)
    nblin2 = np.size(xic)
    F = np.ones((nblin2, nbcol))
    for k in range(1, nbcol):
        F[:, k] = F[:, k-1] * xic

    # Systeme des equations normales sous contraintes
    yi  = np.array([yi]).T
    yic = np.array([yic]).T
    M, B = ConstructionMXB(A, F, yi, yic)

    # Resolution
    cf = np.linalg.solve(M, B)

    # Retourner les coefficients du polynome
    return cf

def compute_parametric_curve(cf, t):
    degree = len(cf) - 3

    # Horner evaluation of vector p(t) for display
    pt = cf[degree] * np.ones(np.size(t))
    for k in range(degree-1, -1, -1):
        pt *= t
        pt += cf[k]
    return pt