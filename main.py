import cv2, time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx # pip install networkx

from skeleton import Skeleton

# Point d'entree
if __name__ == '__main__':

    # Ouvrir l'image sur laquelle travailler
    image = cv2.imread('test1.jpeg', 0)
    image = cv2.resize(image, (500, 500))

    # Filtrage bilatéral ou gaussien pour réduire le bruit
    # image = cv2.bilateralFilter(image, 5, 20, 100, borderType=cv2.BORDER_CONSTANT)
    kernel_size = 11 # taille impaire
    image = cv2.GaussianBlur(image, (kernel_size,kernel_size), 0)

    # Segmentation de l'image
    threshold = 70
    _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Appliquer un filtre de seuillage pour binariser l'image
    # _, binary_image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

    # Squelettisation: Recuperer le squelette du neuronne et l'afficher
    thinned_image = cv2.ximgproc.thinning(image)
    # skeleton = Skeleton(thinned_image, (73, 166))
    skeleton = Skeleton(thinned_image)

    # Enlever quelques points inutiles
    skeleton.simplify()

    # Afficher le squelette
    # skeleton.plot()

    # Detecter les ramifications du squelette en parcourant tous les points
    skeleton.get_branching_points()
    for p in skeleton.branching_points:
        plt.scatter(p[0], p[1], color="red")

    # Segmenter les branches en partant des points de ramification
    skeleton.segmentation()

    # Pour chaque branche
    for branch in skeleton.branches:

        # Calculer une approximation polynomiale de la branche
        branch.least_square_approximation()

        # Tracer l'approximations de la branche
        # plt.plot(branch.lsqx, branch.lsqy, color="blue")
        branch.plot_approximation()

        # Calculer l'épaisseur moyenne de la branche
        branch.measure_average_thickness(image)

        # Calculer la longueur de la branche
        branch.measure_length()
        print("---")

    # Ranger toutes les branches dans une structure de graphe avec 
    # les points de ramifications comme sommet et les branches comme arêtes
    skeleton.to_graph()

    # Trouver la branche principale (chemin le plus long du graphe)
    # skeleton.get_main_branch()

    # Afficher la fenetre Matplotlib
    # plt.grid()
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()

    # Afficher le graphe correspondant au neurone
    pos = nx.spring_layout(skeleton.G)
    nx.draw_networkx_nodes(skeleton.G, pos)
    nx.draw_networkx_edges(skeleton.G, pos)
    nx.draw_networkx_edge_labels(skeleton.G, pos, font_size=6,
        edge_labels = {
            (u, v): f"thickness:{d['thickness']}\nlength:{d['length']}" 
            for u, v, d in skeleton.G.edges(data=True)
        }
    )
    nx.draw_networkx_labels(skeleton.G, pos)
    plt.show()