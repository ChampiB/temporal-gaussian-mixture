import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
from PIL.ImageTk import PhotoImage

import customtkinter as ctk
import tkinter as tk

from tgm.agents.debug.widgets.Shell import Shell
from tgm.agents.models.display.MatPlotLib import MatPlotLib


class WsTool(ctk.CTkFrame):

    def __init__(self, parent, debugger, font=("Helvetica", 30, "normal"), img_width=720, img_height=480):

        super().__init__(parent)

        # Store the debugger, retrieve the selected entry and the background color.
        self.debugger = debugger
        selected = debugger.nav_tree.selected()
        self.root = parent
        self.bg_color = parent._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
        self.text_color = parent._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"])

        # Configure the column and row weights.
        self.columnconfigure(0, weight=1)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        # Ask the user to select an entry in the navigation tree.
        self.selection_label = ctk.CTkLabel(self, text="Please select an entry in the navigation tree.", font=font)
        if selected is None:
            self.selection_label.grid(row=0, column=0, sticky="nsew")

        # Create the combo box to allow the user to select a distribution.
        self.k_frame = ctk.CTkFrame(self, bg_color=self.bg_color, fg_color=self.bg_color)
        self.k_label = ctk.CTkLabel(self.k_frame, text="k:", font=font)
        self.k_combobox = ctk.CTkComboBox(
            self.k_frame, values=["0"],
            font=font, dropdown_font=font, command=self.update_tool
        )
        self.k_label.grid(row=0, column=0, sticky="nse", padx=(15, 5))
        self.k_combobox.grid(row=0, column=1, sticky="nsew", padx=(5, 15))

        # Create the equations.
        empty_image = Image.new(mode="RGB", size=(img_width, img_height), color=(0, 0, 0))
        self.eq_frame = ctk.CTkFrame(self, bg_color=self.bg_color, fg_color=self.bg_color)
        self.eq_images = [[PhotoImage(empty_image), PhotoImage(empty_image)] for _ in range(7)]
        self.eq_labels = [[tk.Label(self.eq_frame, image=image) for image in images] for images in self.eq_images]

        # Update the tool content, if an entry is selected in the navigation tree.
        if selected is not None:
            self.update_content(debugger.checkpoints, selected)

    def update_tool(self, _):
        self.update_content(self.debugger.checkpoints, None)

    def update_content(self, checkpoints, tags):

        # Ensure that the tags are correct.
        if tags is None:
            tags = self.debugger.nav_tree.selected()
        self.k_combobox.configure(values=[str(i) for i in range(len(checkpoints[tags[0]]["gm"].m))])

        # Remove the initial text, and display the images instead.
        if self.selection_label.winfo_viewable():
            self.selection_label.grid_forget()
        if not self.k_label.winfo_viewable():
            self.k_frame.grid(row=1, column=1, sticky="nsew", padx=40, pady=20)
            self.eq_frame.grid(row=2, column=1)

        # Update the equations figures.
        k = int(self.k_combobox.get())
        gm_id = tags[0]
        gm = checkpoints[gm_id]["gm"]

        matrices = {
            r"W_k^{-1}": torch.inverse(gm.W[k]),
            r"W_k": gm.W[k],
            r"N_k^{'}": "None" if gm.N_prime is None else gm.N_prime[k],
            r"S_k^{'}": "None" if gm.S_prime is None else gm.S_prime[k],
            r"\frac{\beta_kN_k^{'}}{\beta_k+ N_k^{'}}": "None" if gm.N_prime is None or gm.β is None else (gm.β[k] * gm.N_prime[k]) / (gm.β[k] + gm.N_prime[k]),
            r"(\bar{x}^{'}_k - m_k)(\bar{x}^{'}_k - m_k)^\top": "None" if gm.x_prime is None or gm.m is None else torch.outer(gm.x_prime[k] - gm.m[k], gm.x_prime[k] - gm.m[k]),
            r"\bar{W}_k^{-1}": torch.inverse(gm.W_bar[k]),
            r"\bar{W}_k": gm.W_bar[k],
            r"N_k^{''}": "None" if gm.N_second is None else gm.N_second[k],
            r"S_k^{''}": "None" if gm.S_second is None else gm.S_second[k],
            r"\frac{\bar{\beta}_kN_k^{''}}{\bar{\beta}_k+ N_k^{''}}": "None" if gm.N_second is None or gm.β_bar is None else (gm.β_bar[k] * gm.N_second[k]) / (gm.β_bar[k] + gm.N_second[k]),
            r"(\bar{x}^{''}_k - \bar{m}_k)(\bar{x}^{''}_k - \bar{m}_k)^\top": "None" if gm.x_second is None or gm.m_bar is None else torch.outer(gm.x_second[k] - gm.m_bar[k], gm.x_second[k] - gm.m_bar[k]),
            r"\hat{W}_k^{-1}": torch.inverse(gm.W_hat[k]),
            r"\hat{W}_k": gm.W_hat[k],
        }
        i = 0
        for y in range(len(self.eq_labels)):
            for x in range(len(self.eq_labels[y])):
                name, matrix = list(matrices.items())[i]
                image = MatPlotLib.draw_equation(
                    f"{name} = " + Shell.to_latex_format(matrix),
                    self.to_rgb(self.bg_color), self.to_rgb(self.text_color), 50, 10
                )
                image = FigureCanvasTkAgg(image, master=self.eq_frame)
                self.eq_images[y][x] = image
                self.eq_labels[y][x].grid_forget()
                self.eq_labels[y][x] = image.get_tk_widget()
                self.eq_labels[y][x].grid(row=y, column=x, sticky="w", padx=15, pady=15)
                i += 1

    def to_rgb(self, color):
        return tuple(((c // 256) / 255 for c in self.root.winfo_rgb(color)))
