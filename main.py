import cv2, time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx # pip install networkx

from skeleton import Skeleton

def find_noyau(img, skel):

    # Erosion et dilatation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.erode(img, kernel, iterations=9)

    # Appliquer un filtre gaussien
    thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
    opening = cv2.dilate(thresh, kernel, iterations=9)

    # Trouver les contours
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Calculer le centroïde
    for contour in contours : 
        M = cv2.moments(contour)
        centroid_x = int(M['m10'] / M['m00'])
        centroid_y = int(M['m01'] / M['m00'])
        centre=np.array((centroid_x,centroid_y))
        mask=np.zeros_like(skel)

        cv2.drawContours(mask, [contour], 0, 255, -1)
        inverse_mask=cv2.bitwise_not(mask)  
        output = cv2.bitwise_and(skel, inverse_mask)
        adjacent_pixels = []
        for cnt in contour : 
            x, y = cnt[0]
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    if output[y + dy, x + dx] == 255:
                        adjacent_pixels.append((x + dx, y + dy))
                        
        unique_adjacent_pixel=list(set(adjacent_pixels))
        for p in unique_adjacent_pixel : 
            for op in unique_adjacent_pixel :
                if p==op :
                    continue 
                if np.abs(p[1]-op[1])<=2 and np.abs(p[0]-op[0])<=2 :
                    unique_adjacent_pixel.remove(op)

        return centre, output, unique_adjacent_pixel 

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
    threshold = 60
    _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Squelettisation: Recuperer le squelette du neuronne et l'afficher
    thinned_image = cv2.ximgproc.thinning(image)
    centre, thinned_image, point_adja = find_noyau(image, thinned_image)
    skeleton = Skeleton(thinned_image, centre)

    # Enlever quelques points inutiles
    skeleton.simplify()

    # Afficher le squelette
    # skeleton.plot()

    # Detecter les ramifications du squelette en parcourant tous les points
    skeleton.get_branching_points(point_adja)
    for p in skeleton.branching_points:
        plt.scatter(p[0], p[1], color="red")
    plt.scatter(skeleton.soma[0], skeleton.soma[1], color="orange")

    # Segmenter les branches en partant des points de ramification
    skeleton.segmentation()

    # Pour chaque branche
    for branch in skeleton.branches:

        # Calculer une approximation polynomiale de la branche
        branch.least_square_approximation()

        # Tracer l'approximations de la branche
        # branch.plot_approximation()

        # Calculer l'épaisseur moyenne de la branche
        branch.measure_average_thickness(image)

        branch.relier_centre(point_adja, centre, image)
        branch.least_square_approximation()
        branch.plot_approximation()

        # Calculer la longueur de la branche
        branch.measure_length()
        print("---")

    skeleton.remplacement_des_points(point_adja)

    # Ranger toutes les branches dans une structure de graphe avec 
    # les points de ramifications comme sommet et les branches comme arêtes
    skeleton.to_graph()

    # Trouver la branche principale (chemin le plus long du graphe)
    skeleton.get_main_branch()

    # Afficher la fenetre Matplotlib
    # plt.grid()
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    # Afficher le graphe correspondant au neurone
    plt.figure()
    pos = nx.spring_layout(skeleton.G)
    colors = [
        "purple" if (node == tuple(centre)) else "blue"
        for node in skeleton.G.nodes()
    ]
    nx.draw_networkx_nodes(skeleton.G, pos, node_color=colors)

    nx.draw_networkx_edges(skeleton.G, pos)
    nx.draw_networkx_edges(skeleton.G, pos, edgelist=skeleton.main_branch, edge_color='r')
    nx.draw_networkx_edge_labels(skeleton.G, pos, font_size=6,
        edge_labels = {
            (u, v): f"thickness:{d['thickness']}\nlength:{d['length']}" 
            for u, v, d in skeleton.G.edges(data=True)
        }
    )
    nx.draw_networkx_labels(skeleton.G, pos)
    plt.show()