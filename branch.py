import numpy as np
import cv2
import matplotlib.pyplot as plt

import lsq

def parametric_linear_interpolation(points):
    x=points[0:-2:2]
    y=points[1:-1:2]

    m, b = np.polyfit(x, y, 1)

    # Generate x values to plot the line
    line_x = np.linspace(x[0], x[-1], 100)

    # Calculate the corresponding y values on the line
    line_y = m * line_x + b

    return line_x,line_y

class Branch:
    """
        Classe pour représenter une branche de maniere abstraite et travailler dessus avec
        des méthodes de plus haut niveau
    """

    def __init__(self, points, branching_points):
        """
            Constructeur de la classe Branch
            points: liste des points (i,j) de la branche
            nb_points: nombre de points de la branche
            branching_points: liste des points (i,j) de ramification du squelette
            start: premier point de la branche
            end: dernier point de la branche
            lsqcfx et lsqcfy: coefficients des polynomes d'approximation aux moindres
                carrés qui approxime les points de la branche en x et y
            thickness: epaisseur moyenne de la branche
            length: taille de la branche (calculée à partir de son approximation polynomiale)
        """
        self.points = points
        self.nb_points = len(points)
        self.branching_points = branching_points
        self.start = points[0]
        self.end = points[-1]
        self.lsqcfx = None
        self.lsqcfy = None
        self.thickness = 0
        self.length = 0

        self.centre = -1
        self.line = np.array([])
        self.fonction_centre = None

    def is_branching_out(self):
        """
            Retourne True ssi le dernier point de la branche est un point de
            ramification du squelette
        """
        return self.end in self.branching_points

    def relier_centre(self,adjacent,centre,image):
        """
            Retourne True ssi le premier point de la branche est le centre
        """
        if self.start in adjacent : 

            start_point = self.start
            end_point = centre
            # Tracer un segment de ligne droite entre les deux points sur la carte de squelette
            path_img = np.zeros_like(image)
            line=cv2.line(path_img, start_point, end_point, 255, 1)
            for i in range(0,len(line)):
                for j in range(0,len(line[i])):
                    if line[i][j]==255 :
                        self.line=np.append(self.line,np.array([j,i]))
            self.centre=1

    def least_square_approximation(self):
        """
            Calcule une fonction d'approximation des points de la branche 
            avec une méthode des moindres carrés
        """
        # Recuperer la liste des coordonnées x et y des points de la branche
        N = self.nb_points
        xi = np.array(self.points)[:, 0]
        yi = np.array(self.points)[:, 1]

        # Definir les contraintes: la spline doit passer par le premier
        # et le dernier point de la branche
        #middle = int(np.floor(N/2))
        xic = [self.start[0],  self.end[0]]
        yic = [self.start[1],  self.end[1]]
        kc = [0, N-1]

        # A FAIRE: Rajouter des points de contraintes (ex: tous les 20 points)
        # pour avoir une courbe la plus lisse possible

        # Paramétrisation chordale
        tc = np.zeros(N)
        for i in range(1,N):
            di = np.sqrt((xi[i] - xi[i-1])**2 + (yi[i] - yi[i-1])**2)
            tc[i] = tc[i-1] + di
        tc = tc / tc[N-1]

        # Discrétisation de [0,1] pour afficher la courbe
        t = np.linspace(0, 1, 500)

        # On détermine les parametres associés aux points (xic, yic) à interpoler
        tcc = []
        for i in kc:
            tcc.append(tc[i])

        # Degré de la spline
        degree = 8

        # Calculer les coefficients de l'approximation en x et en y
        self.lsqcfx = lsq.LeastSquaresConstraintsMonomes(tc, xi, tcc, xic, degree, t)
        self.lsqcfy = lsq.LeastSquaresConstraintsMonomes(tc, yi, tcc, yic, degree, t)

    def plot_approximation(self):
        """
            Calcule les points de la courbe paramétrique de l'approximation polynomiale 
            avec une discretisation de [0,1] et les coefficients de la fonction.
            Affiche la courbe. Doit être appelée après least_square_approximation().
        """
        t = np.linspace(0, 1, 1000)
        fonction=[[],[]]
        if self.centre==1 : 
            fonction=parametric_linear_interpolation(self.line)
            self.fonction_centre=fonction

        ptx = lsq.compute_parametric_curve(self.lsqcfx, t)
        ptx=np.concatenate((fonction[0],ptx))
        pty = lsq.compute_parametric_curve(self.lsqcfy, t)
        pty=np.concatenate((fonction[1],pty))
        plt.plot(ptx, pty, color="blue")
        
    def measure_average_thickness(self, image):
        """
            Calcule l'épaisseur moyenne de la branche, en prenant en chaque point de
            l'approximation, pris à intervalle régulier, le nombre de pixels blancs que l'on
            compte sur la direction perpendiculaire de la tengente en ce point.
            Doit être appelée apres l'approximation aux moindres carrés.
            Prend l'image binaire en entrée
        """
        # On decoupe la courbe en n points
        n = 10
        t = np.linspace(0, 1, n)
        ptx = lsq.compute_parametric_curve(self.lsqcfx, t)
        pty = lsq.compute_parametric_curve(self.lsqcfy, t)

        # On récupère la liste des points sous forme d'une liste de couples, 
        # et on retire le premier point qui est le point de ramification car les 
        # mesures effectuees a cet endroit ne seront pas pertinentes
        points = list(zip(ptx, pty))
        points.pop(0) # point de ramification
        # Si la branche se termine par un point de ramification, on enlève le dernier point
        if self.is_branching_out():
            points.pop(-1)

        # Calcul des dérivées x(t) et y(t)
        dxdt = np.gradient(ptx)
        dydt = np.gradient(pty)

        # On compte l'indice des points
        point_indice = 1

        # On stocke les mesures des differentes epaisseurs
        measurements = []

        # Pour chaque point sur la courbe
        for p in points:
            plt.scatter(p[0], p[1], color="red")

            # Calculer les tangeantes de l'approximation à intervalle régulier
            tangent = np.array([dxdt[point_indice], dydt[point_indice]])
            tangent = tangent / np.linalg.norm(tangent)
            plt.quiver(p[0], p[1], tangent[0], tangent[1], angles='xy', scale_units='xy', scale=1)

            # Pour chaque tangeante, calculer la direction perpendiculaire associée
            u = np.array([-tangent[1], tangent[0]])
            u = u / np.linalg.norm(u)
            plt.quiver(p[0], p[1], u[0], u[1], angles='xy', scale_units='xy', scale=1)

            # Compter les pixels blancs de l'image dans la direction de cette perpendiculaire
            # On compte dans le sens de u
            pk, k, measure = p, 0, 0
            while image[round(pk[1])][round(pk[0])] == 255:
                measure += 1
                k += 1
                plt.scatter(pk[0], pk[1], s=1.5, color="red")
                pk = p + k*u

            # Et on compte dans le sens de -u pour faire toute l'epaisseur
            pk, k = p, 0
            while image[round(pk[1])][round(pk[0])] == 255:
                measure += 1
                k += 1
                plt.scatter(pk[0], pk[1], s=1.5, color="red")
                pk = p - k*u

            # On enregistre la mesure de l'epaisseur au point p
            measurements.append(measure)

            # On passe au point suivant
            point_indice += 1
        
        # Calculer la moyenne de tous les comptages => epaisseur moyenne de la branche
        moyenne = np.around(np.mean(np.array(measurements)), decimals=2)
        ecart_type = np.std(np.array(measurements))
        seuil =  ecart_type
        nouvelle_liste = []
        if seuil != 0:
            for point in measurements:
                if abs(point - moyenne) < seuil:
                    nouvelle_liste.append(point)
        if len(nouvelle_liste) != 0:
            # Calcul de la moyenne de la nouvelle liste
            self.thickness = np.around(np.mean(nouvelle_liste), decimals=2)
            print("Epaisseur:", self.thickness)

    def measure_length(self):
        """
            Calcule la longueur d'une branche en calculant la longueur de la courbe de
            l'approximation polynomiale de la branche.
        """
        # Calcul des points de la courbe sur [0,1]
        t = np.linspace(0, 1, 300)
        ptx = lsq.compute_parametric_curve(self.lsqcfx, t)
        pty = lsq.compute_parametric_curve(self.lsqcfy, t)

        # Dérivées de x(t) et y(t)
        dxdt, dydt = np.gradient(ptx), np.gradient(pty)

        # Longueur de la courbe
        self.length = np.around(np.sum(np.sqrt(dxdt**2 + dydt**2)), decimals=2)
        print("Longueur:", self.length)
