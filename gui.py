import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from PIL import ImageTk, Image

fen = tk.Tk()

def on_closing():
    fen.destroy()
fen.protocol("WM_DELETE_WINDOW", on_closing)

def afficher_image():
    # Ouvrir l'image avec PIL
    img = Image.open(imgsrc_input.get())
    
    # Redimensionner l'image si besoin
    # img = img.resize((largeur, hauteur))
    
    # Convertir l'image en format compatible avec Tkinter
    img_tk = ImageTk.PhotoImage(img)
    
    # Créer un widget Label pour afficher l'image
    label_img = tk.Label(fen, image=img_tk)
    label_img.pack()

def afficher_plots(fig1, fig2):
    canvas = FigureCanvasTkAgg(fig1, master=fen)
    canvas1 = FigureCanvasTkAgg(fig2, master=fen)
    canvas.draw()
    canvas1.draw()
    
    toolbar_frame = tk.Frame(fen)
    toolbar_frame.pack(side=tk.TOP, fill=tk.BOTH)
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()
    
    toolbar1_frame = tk.Frame(fen)
    toolbar1_frame.pack(side=tk.TOP, fill=tk.BOTH)
    toolbar1 = NavigationToolbar2Tk(canvas1, toolbar1_frame)
    toolbar1.update()
    
    canvas.get_tk_widget().config(width=500, height=500)
    canvas1.get_tk_widget().config(width=500, height=500)

    canvas.get_tk_widget().pack(side=tk.LEFT)
    canvas1.get_tk_widget().pack(side=tk.RIGHT)

# Afficher les logos sur la fenetre
img = Image.open("assets/uga_logo.jpeg")
img = img.resize((150, 80))
img_tk = ImageTk.PhotoImage(img)
label_img = tk.Label(fen, image=img_tk)
label_img.pack(side=tk.LEFT,anchor=tk.NW)
img1 = Image.open("assets/MoreHisto_logo.jpeg")
img1 = img1.resize((180, 90))
img_tk1 = ImageTk.PhotoImage(img1)
label_img1 = tk.Label(fen, image=img_tk1)
label_img1.pack(side=tk.RIGHT,anchor=tk.NE)

# Définir les labels pour chaque champs de texte
text = tk.Label(fen, text="entrer l'image", font=("Helvetica", 16, "bold"), fg="blue")
text.place(x=50, y=120)
text1 = tk.Label(fen,text="entrer le seuil", font=("Helvetica", 16, "bold"), fg="blue")
text1.place(x=50, y=170)

# Définir les champs de texte pour entrer l'image source
# et le seuil à appliquer pour la segmentation de l'image
imgsrc_input = tk.Entry(fen)
imgsrc_input.place(x=50, y=150)
thresh_input = tk.Entry(fen)
thresh_input.place(x=50, y=200)

bouton = tk.Button(fen, text="Afficher l'image", command=afficher_image)
bouton.pack()

text = tk.Label(fen, text="EXECUTER", font=("Arial", 20))
text.place(x=550, y=50)
fen.title("Traitement des neurones")
fen.geometry("2000x1000")
fen.resizable(width=True, height=True)