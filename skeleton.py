import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx # pip install networkx
from scipy.spatial.distance import cosine

from branch import Branch

class Skeleton:
    """
        Classe pour représenter le squelette de maniere abstraite et travailler dessus avec
        des méthodes de plus haut niveau
    """

    def __init__(self, matrix, soma):
        """
            Constructeur de la classe Skeleton
            matrix: np.array de dimension 2 qui contient une image binaire du squelette
        """
        # Recuperer la liste des points blancs de la matrice, ROI du squelette
        self.points = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[j][i] == 255:
                    self.points.append((i, j))
        
        # Contient les points de ramification du squelette
        self.branching_points = []

        # Contient des instances de la classe Branch qui sont les branches du squelette
        self.branches = []
        self.main_branch = []
        self.soma = soma
        self.G = None

    def plot(self):
        """ 
            Afficher les points du squelette sur une fenetre Matplotlib 
        """
        # Parcourir les points du squelette et les afficher
        for p in self.points:
            plt.scatter(p[0], p[1], s=1.5, linewidths=0.5, color="black")

    def simplify(self):
        """
            Simplifie les lignes du squelette pour éviter d'avoir un surplus de points par endroits
            ce qui occasionne parfois des points de ramifications qui ont 4 voisins
        """
        new_points = []
        for point in self.points:
            i, j = point[0], point[1]
            if not ((i-1, j) in self.points and (i, j-1) in self.points):
                new_points.append(point)
        self.points = new_points

    def get_neighbors(self, p, excludeDiag=False, excludeThose=[]):
        """
            Retourner les voisins d'un point p = (i,j) qui se trouvent sur une case adjacente, qui ne sont pas 
            diagonaux ssi excludeDiag=True et qui ne sont pas dans la liste à exclure
        """
        i, j = p[0], p[1]
        if excludeDiag:
            return [
                (i+di, j+dj) for di in [-1,0,1] for dj in [-1,0,1] 
                if (di, dj) != (0, 0) and abs(di) != abs(dj) and (i+di, j+dj) in self.points
                and (i+di, j+dj) not in excludeThose
            ]
        else:
            return [
                (i+di, j+dj) for di in [-1,0,1] for dj in [-1,0,1] 
                if (di, dj) != (0,0) and (i+di, j+dj) in self.points
                and (i+di, j+dj) not in excludeThose
            ]

    def get_branching_points(self, point_adja):
        """
            Detecter les points de ramification du squelette, c'est-à-dire les points en
            lesquels une branche vient se séparer en deux. On parcourt tous les points du squelette.
        """
        # Liste qui contient les points pouvant être des ramifications du squelette
        branching_points_candidates = []

        # Liste a retourner qui contient les points exacts de ramification
        branching_points = []

        for p in point_adja : 
            branching_points.append(p)

        # Pour chaque point du squelette
        for point in self.points:

            # Si le point a exactement 3 voisins
            nb_neighbors = len(self.get_neighbors(point))
            if nb_neighbors == 3:

                # Stocker dans une liste les points d'arrivees
                # de chaque trajectoire apres 10 iterations
                arrivals = []

                # Recuperer les voisins du point de depart
                neighbors = self.get_neighbors(point)

                # Pour chacun de ses voisins
                for n in neighbors:

                    # On définit le point courant
                    p_current = n

                    # On effectue 10 itérations pour s'eloigner du point de depart
                    # en se deplacant a chaque fois au point le plus eloigné du point de depart
                    for k in range(10):

                        # Recuperer les voisins du voisin du point de depart
                        neighbors_2nd = self.get_neighbors(p_current)

                        # Parmis ses voisins, choisir celui qui est le plus éloigné du point
                        # de départ pour recommencer jusqu'à avoir avancé 10 fois
                        distances = np.linalg.norm(np.array(neighbors_2nd) - np.array(point), axis=1)
                        index_max = np.argmax(distances)
                        p_current = neighbors_2nd[index_max]

                    # Ajouter dans la liste prévue à cet effet le point d'arrivée
                    arrivals.append(p_current)

                # Verifier que dans la liste des points d'arrivees on ait pas deux points qui sont
                # tres proches. Si c'est le cas ils sont sur la meme branche et donc le point que l'on
                # regarde ne peut pas etre un point de ramification
                dist_max = 3
                new_arrivals = []
                for k, x in enumerate(arrivals):
                    close_points = [
                        y for l, y in enumerate(arrivals) if k != l 
                        and np.linalg.norm(np.array(x) - np.array(y)) < dist_max
                    ]
                    if not close_points:
                        new_arrivals.append(x)

                # Donc si la taille de la nouvelle liste d'arrivée n'est plus égale à 3
                # des points trop proches ont été supprimés, le point de départ ne peut pas etre un point 
                # de ramification. Si la taille est toujours égale à 3 alors on enregistre le point
                if len(new_arrivals) == 3:
                    branching_points_candidates.append(point)

        # Pour chaque candidat pouvant être un point de ramification
        for p in branching_points_candidates:

            # On recupere ses voisins
            neighbors = self.get_neighbors(p)

            # On calcule les 3 vecteurs que le candidat forme avec ses 3 voisins
            u0 = np.array(neighbors[0]) - np.array(p)
            u1 = np.array(neighbors[1]) - np.array(p)
            u2 = np.array(neighbors[2]) - np.array(p)

            # Calculer les 3 angles formés par u0, u1 et u2
            angle_u0u1 = np.arccos(1 - cosine(u0, u1))
            angle_u1u2 = np.arccos(1 - cosine(u1, u2))
            angle_u2u0 = np.arccos(1 - cosine(u2, u0))

            # On verifie si parmi ces 3 vecteurs, on en a 2 qui forment un angle de même mesure,
            # si oui alors p est un point de ramification et on l'ajoute a la liste
            if len(set([angle_u0u1, angle_u1u2, angle_u2u0])) < 3:
                branching_points.append(p)

        self.branching_points = branching_points


    def segmentation(self):
        """
            Parcourir les branches du squelette en partant des points de ramification.
            Les branches sont stockées dans une liste d'instances de la classe Branch.
            Doit être appelée apres la detection des points de ramification.
        """
        # Pour chaque point de ramification
        for branching_point in self.branching_points:

            # Recuperer les voisins du point de ramification (1 voisin = 1 branche)
            neighbors = self.get_neighbors(branching_point)

            # Pour chaque voisin (chaque branche)
            for n in neighbors:

                # On regarde si ce voisin est déjà dans une des autres branches du squelette
                # Si oui on passe à la branche suivante pour ne pas repasser sur une branche
                # qu'on a déjà parcouru
                already_visited = False
                for branch in self.branches:
                    if n in branch.points:
                        already_visited = True
                if already_visited:
                    continue

                # On stocke les points de la branche dans cette liste
                # Le premier point de la branche est le point de ramification courant
                # Le deuxieme est le voisin qu'on regarde
                br_points = [branching_point]
                
                # On enregistre le point courant
                p_current = n

                # On enregistre le point precedent
                p_previous = branching_point

                # Variable passee a true si le nombre de voisins est nul, le seul voisin
                # du point courant a deja été visité
                end_branch = False

                # On compte le nombre d'iterations pour changer le point qu'on choisit par rapport 
                # auquel on regarde la distance du voisin le plus éloigné 
                iteration_counter = 1

                # Tant qu'on est pas arrivé à la fin d'une branche
                while end_branch == False:

                    # Si on est arrivé à un point de ramification on s'arrete et on l'ajoute a la liste
                    if p_current in self.branching_points:
                        br_points.append(p_current)
                        break
                    
                    # Ajouter le point courant à la liste des points de la branche
                    br_points.append(p_current)
                    
                    # On recupere les voisins du point_courant en excluant ceux qu'on a déjà visité
                    neighbors_2nd = self.get_neighbors(p_current, excludeThose=br_points)

                    # Si on ne trouve plus de voisins, on est arrivé à la fin de la branche
                    if len(neighbors_2nd) == 0:
                        end_branch = True
                    else:
                        # Sinon on verifie si un des voisins est un point de ramification, 
                        # si c'est le cas il devient le prochain point courant et la boucle s'arretera
                        neighbor_found = False
                        for p in neighbors_2nd:
                            if p in self.branching_points:
                                p_previous = p_current
                                p_current = p
                                neighbor_found = True

                        # Si on a pas de voisin qui est un point de ramification
                        if neighbor_found == False:

                            # Alors, si on a fait:
                            # - moins de 5 itérations on prend le voisin le plus éloigné 
                            # du point de ramification d'où l'on est parti
                            # + de 5 itérations on prend le voisin le plus éloigné du point précédent
                            dist = 0
                            if iteration_counter < 5:
                                dist = np.linalg.norm(
                                    np.array(neighbors_2nd) - np.array(branching_point), axis=1)
                            else:
                                dist = np.linalg.norm(
                                    np.array(neighbors_2nd) - np.array(p_previous), axis=1)

                            index_max = np.argmax(dist)
                            farthest = neighbors_2nd[index_max]
                            p_previous = p_current
                            p_current = farthest

                    iteration_counter += 1
                
                # On ajoute la branche a la liste des branches du squelette
                # branch = Branch(br_points, self.branching_points)
                # self.branches.append(branch)
                if (len(br_points) > 5):
                    branch = Branch(br_points, self.branching_points)
                    self.branches.append(branch)

    def remplacement_des_points(self,point_adjacent): 
        i=0 
        while i<len(self.branching_points) : 
            if self.branching_points[i] in point_adjacent :
                self.branching_points.remove(self.branching_points[i])
            else : 
                i=i+1

        self.branching_points.append(tuple(self.soma))
            
    def to_graph(self):
        """
            Retourne un graphe représentant le neurone dont les sommets sont les ramifications
            et les arêtes sont les branches
        """
        # Créer une instance de graphe
        G = nx.Graph()

        # Recuperer les points de terminaison des branches qui ne sont
        # pas des points de ramification
        ending_points = []
        for branch in self.branches:
            if not branch.is_branching_out():
                ending_points.append(branch.end)

        # Ajouter les sommets dans le graphe c'est-à-dire
        # (points de ramifications + points de terminaison)
        G.add_nodes_from(self.branching_points + ending_points)

        # Pour chaque branche on récupère le point de départ et d'arrivée
        # de manière à construire la liste des arêtes
        edges_list = []
        for b in self.branches:
            if b.centre == 1 : 
                edges_list.append(
                    (tuple(self.soma), b.end, {'thickness': b.thickness, 'length': b.length})
                )
            else :
                edges_list.append(
                    (b.start, b.end, {'thickness': b.thickness, 'length': b.length})
                )

        # Ajouter les arêtes dans le graphe
        G.add_edges_from(edges_list)
        self.G = G
    
    def get_main_branch(self):
        """
            Retourne la liste des branches qui forment la branche
            principale, c'est-à-dire le chemin le plus long de G
        """
        # Creer un graphe H = G tel que H est pondéré uniquement par length
        H = nx.Graph()
        new_edges = []
        for (u, v, l) in self.G.edges.data('length'):
            new_edges.append((u, v, l))
            
        # Ajout des arêtes pondérées au graphe
        H.add_weighted_edges_from(new_edges)

        edges_list = []
        for edge in H.edges.data():
            edges_list.append((edge[0], edge[1], edge[2]['weight']))
        # print(edges_list)

        # Recuperer tous les chemins partant du soma vers les autres sommets
        paths = []
        for node in list(H.nodes):
            print(type(tuple(self.soma)))
            paths.append(nx.bellman_ford_path(self.G, tuple(self.soma), node, weight='weight'))
        # print(paths)

        # Pour chaque chemin calculer la somme des poids des branches
        weighted_paths = []
        for path in paths:

            # Somme totale des poids
            sum_weight = 0

            # On récupère le point de départ et d'arrivée du chemin
            a, b = path[0], path[-1]

            # Pour chaque noeud du chemin
            for node_indice in range(0, len(path)-1):
                x = path[node_indice]
                y = path[node_indice+1]
                weight = H.get_edge_data(x, y)['weight']
                sum_weight += weight
            weighted_paths.append((a, b, sum_weight))

        # Recuperer la branche principale
        max_path = max(weighted_paths, key=lambda path: path[2])
        main_branch_edges = nx.dijkstra_path(H, max_path[0], max_path[1])

        # Stocker la branche principale
        for i in range(0, len(main_branch_edges)-1):
            self.main_branch.append((main_branch_edges[i], main_branch_edges[i+1]))


    
