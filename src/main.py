"""
Punto de entrada de la aplicación.
Interfaz gráfica para detección de neumonía.
"""

from tkinter import Tk, ttk, font, filedialog, Text, StringVar
from tkinter.messagebox import showinfo, askokcancel, WARNING
from PIL import ImageTk, Image

from src.integrator import run_inference


MODEL_PATH = "models/conv_MLP_84.h5"


class App:
    """
    Aplicación gráfica principal.
    """

    def __init__(self):
        self.root = Tk()
        self.root.title("Detección de Neumonía")

        bold_font = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(False, False)

        # Labels
        ttk.Label(
            self.root,
            text="SOFTWARE PARA APOYO AL DIAGNÓSTICO DE NEUMONÍA",
            font=bold_font,
        ).place(x=120, y=20)

        ttk.Label(self.root, text="Imagen Radiográfica", font=bold_font).place(
            x=110, y=65
        )
        ttk.Label(self.root, text="Imagen con Heatmap", font=bold_font).place(
            x=545, y=65
        )
        ttk.Label(self.root, text="Resultado:", font=bold_font).place(x=500, y=350)
        ttk.Label(self.root, text="Probabilidad:", font=bold_font).place(x=500, y=400)

        # Variables
        self.result = StringVar()
        self.probability = StringVar()

        # Text areas
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)

        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        self.text_result = Text(self.root, width=12, height=1)
        self.text_prob = Text(self.root, width=12, height=1)

        self.text_result.place(x=610, y=350)
        self.text_prob.place(x=610, y=400)

        # Buttons
        ttk.Button(self.root, text="Cargar Imagen", command=self.load_image).place(
            x=70, y=460
        )
        ttk.Button(self.root, text="Predecir", command=self.predict).place(
            x=220, y=460
        )
        ttk.Button(self.root, text="Borrar", command=self.clear).place(
            x=670, y=460
        )

        self.image_path = None
        self.root.mainloop()

    def load_image(self):
        """
        Carga una imagen desde el sistema de archivos.
        """
        self.image_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("Imagen", "*.jpg *.jpeg *.png"),
            ),
        )

        if not self.image_path:
            return

        img = Image.open(self.image_path).resize((250, 250))
        self.img1 = ImageTk.PhotoImage(img)
        self.text_img1.image_create("end", image=self.img1)

    def predict(self):
        """
        Ejecuta la inferencia usando el integrador.
        """
        if not self.image_path:
            showinfo("Error", "Debe cargar una imagen primero.")
            return

        label, prob, heatmap = run_inference(
            self.image_path, MODEL_PATH
        )

        self.text_result.delete("1.0", "end")
        self.text_prob.delete("1.0", "end")

        self.text_result.insert("end", label)
        self.text_prob.insert("end", f"{prob:.2f}%")

        heatmap_img = Image.fromarray(heatmap).resize((250, 250))
        self.img2 = ImageTk.PhotoImage(heatmap_img)
        self.text_img2.image_create("end", image=self.img2)

    def clear(self):
        """
        Limpia la interfaz.
        """
        if askokcancel(
            title="Confirmación",
            message="¿Desea borrar los datos?",
            icon=WARNING,
        ):
            self.text_img1.delete("1.0", "end")
            self.text_img2.delete("1.0", "end")
            self.text_result.delete("1.0", "end")
            self.text_prob.delete("1.0", "end")
            self.image_path = None


def main():
    App()


if __name__ == "__main__":
    main()
