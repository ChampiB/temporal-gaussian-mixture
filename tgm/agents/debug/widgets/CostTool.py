import tkinter as tk
from PIL import Image
from PIL.ImageTk import PhotoImage
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tgm.agents.models.display.MatPlotLib import MatPlotLib


class CostTool(ctk.CTkFrame):

    def __init__(self, parent, debugger, font=("Helvetica", 30, "normal"), img_width=1920, img_height=1080):

        super().__init__(parent)

        # Retrieve the selected entry.
        selected = debugger.nav_tree.selected()

        # Configure the column and row weights.
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Ask the user to select an entry in the navigation tree.
        self.selection_label = ctk.CTkLabel(self, text="Please select an entry in the navigation tree.", font=font)
        if selected is None:
            self.selection_label.grid(row=0, column=0, sticky="nsew")

        # Create the image and label.
        empty_image = Image.new(mode="RGB", size=(img_width, img_height), color=(0, 0, 0))
        self.vfe_image = PhotoImage(empty_image)
        self.label = tk.Label(self, image=self.vfe_image)
        if selected is not None:
            prev_gm, next_gm, fit_id = selected
            self.update_content(
                debugger.gms, debugger.xs, debugger.init_params, int(prev_gm), int(next_gm), int(fit_id)
            )

    def update_content(self, gms, xs, init_params, prev_gm, next_gm, fit_id):

        # Remove the initial text, and display the images instead.
        if self.selection_label.winfo_viewable():
            self.selection_label.grid_forget()
        if not self.label.winfo_viewable():
            self.label.grid(row=0, column=0)

        # Update the VFE image.
        vfe = [gms[i].vfe for i in range(prev_gm, next_gm + 1)]
        x = list(range(prev_gm, next_gm + 1))
        image = MatPlotLib.draw_graph(x, vfe, "Variational Free Energy.")
        self.vfe_image = FigureCanvasTkAgg(image, master=self)
        self.label = self.vfe_image.get_tk_widget()
        self.label.grid(row=0, column=0)
