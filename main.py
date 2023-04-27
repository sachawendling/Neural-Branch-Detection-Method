import cv2, time, csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx # pip install networkx
import tkinter as tk

from skeleton import Skeleton
import gui

def find_noyau(img, skel):
    """
    A partir d'une image binarisée et de son squelette : trouve son noyau, retourne son centre, les points adjacents au noyau
     du squelette et le squelette sans les points à l'intérieur du noyau
    """

    # Erosion avec une ellipse
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.erode(img, kernel, iterations=9)

    # Appliquer un filtre gaussien et une dilatation
    thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
    opening = cv2.dilate(thresh, kernel, iterations=9)

    # Trouver les contours
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour=contours[0]
    # Calculer le centroïde
    M = cv2.moments(contour)
    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])
    centre=np.array((centroid_x,centroid_y))

    #Retire les points du squelette qui sont dans le noyau

    #Applique le masque du noyau sur le squelette
    mask=np.zeros_like(skel)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    inverse_mask=cv2.bitwise_not(mask)  
    output = cv2.bitwise_and(skel, inverse_mask)

    #Cherche les points du squelette collés au contour du noyau
    adjacent_pixels = []
    for cnt in contour : #pour chaque morceau du contour
        x, y = cnt[0]
        #On cherche les points voisins de dx et dy (les points à l'intérieur du noyau ont deja été enlevé)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if output[y + dy, x + dx] == 255:
                    adjacent_pixels.append((x + dx, y + dy))

    #On retire les doublons de la liste                 
    unique_adjacent_pixel=list(set(adjacent_pixels))

    #On supprime les points adjacents qui sont voisins
    for p in unique_adjacent_pixel : 
        for op in unique_adjacent_pixel :
            if p==op :
                continue 
            if np.abs(p[1]-op[1])<=2 and np.abs(p[0]-op[0])<=2 :
                unique_adjacent_pixel.remove(op)

    return centre, output, unique_adjacent_pixel 

def traitement():

    fig1 = plt.figure()

    # Ouvrir l'image sur laquelle travailler
    image = cv2.imread("images/" + gui.imgsrc_input.get(), 0)
    image = cv2.resize(image, (500, 500))

    # Filtrage bilatéral ou gaussien pour réduire le bruit
    # image = cv2.bilateralFilter(image, 5, 20, 100, borderType=cv2.BORDER_CONSTANT)
    kernel_size = 11 # taille impaire
    image = cv2.GaussianBlur(image, (kernel_size,kernel_size), 0)

    # Segmentation de l'image
    threshold = float(gui.thresh_input.get())
    _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Squelettisation: Recuperer le squelette du neuronne et l'afficher
    thinned_image = cv2.ximgproc.thinning(image)
    try:
        centre, thinned_image, point_adja = find_noyau(image, thinned_image)
    except IndexError:
        print("Seuil trop élevé: le noyau n'a pas été détecté")
        return
    skeleton = Skeleton(thinned_image, centre)

    # Enlever quelques points inutiles
    skeleton.simplify()

    # Afficher le squelette
    if gui.plot_skeleton.get() == True:
        skeleton.plot()

    #Afficher le centre
    plt.scatter(skeleton.soma[0], skeleton.soma[1], color="orange")

    # Detecter les ramifications du squelette en parcourant tous les points
    skeleton.get_branching_points(point_adja)
    if gui.plot_brpts.get() == True:
        for p in skeleton.branching_points:
            plt.scatter(p[0], p[1], color="red")
    
    # Segmenter les branches en partant des points de ramification
    skeleton.segmentation()

    # Pour chaque branche
    for branch in skeleton.branches:

        # Calculer une approximation polynomiale de la branche
        branch.least_square_approximation()

        # Calculer l'épaisseur moyenne de la branche
        branch.measure_average_thickness(image)

        # Relier la branche au centre du neurone si son point de depart est 
        # un point adjacent du soma
        branch.relier_centre(point_adja, centre, image)

        branch.least_square_approximation()

        # Tracer l'approximations de la branche
        if gui.plot_trace_lsq.get() == True:
            branch.plot_approximation()

        # Calculer la longueur de la branche
        branch.measure_length()
        print("---")

    #Les branches qui partent d'un point adjacent partent du centre désormais
    skeleton.remplacement_des_points(point_adja)

    # Ranger toutes les branches dans une structure de graphe avec 
    # les points de ramifications comme sommet et les branches comme arêtes
    if gui.plot_graph.get() == True:
        skeleton.to_graph()

        # # Trouver la branche principale (chemin le plus long du graphe)
        skeleton.get_main_branch()

        # Exporter le graphe sous forme d'un fichier csv
        skeleton.save_as_csv(gui.imgsrc_input.get())

    # Afficher la fenetre Matplotlib
    # plt.grid()
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)

    # Afficher le graphe correspondant au neurone
    fig2 = plt.figure()
    if gui.plot_graph.get() == True:
        pos = nx.spring_layout(skeleton.G)
        colors = ["purple" if (node == tuple(centre)) else "blue" for node in skeleton.G.nodes()]
        nx.draw_networkx_nodes(skeleton.G, pos, node_color=colors)
        nx.draw_networkx_edges(skeleton.G, pos)
        nx.draw_networkx_edges(skeleton.G, pos, edgelist=skeleton.main_branch, edge_color='r')
        nx.draw_networkx_edge_labels(skeleton.G, pos, font_size=6, edge_labels={
            (u, v): f"thickness:{d['thickness']}\nlength:{d['length']}\ndepth:{d['depth']}" 
            for u, v, d in skeleton.G.edges(data=True)
        })
        nx.draw_networkx_labels(skeleton.G, pos)

    # Afficher les 2 figures sur l'interface graphique
    gui.afficher_plots(fig1, fig2)

# Point d'entree
if __name__ == '__main__':
    
    # Ajouter un bouton pour lancer le traitement
    button = tk.Button(gui.fen, text='Traitement',width=20, height=3, command=traitement)
    button.place(x=550, y=90)

    gui.fen.mainloop()