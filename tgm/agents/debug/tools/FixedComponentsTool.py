import tkinter as tk

import torch
from PIL import Image
from PIL.ImageTk import PhotoImage
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tgm.agents.models.display.MatPlotLib import MatPlotLib
from tgm.agents.models.inference.GaussianStability import GaussianStability


class FixedComponentsTool(ctk.CTkFrame):

    def __init__(self, parent, debugger, font=("Helvetica", 30, "normal"), img_width=720, img_height=480):

        super().__init__(parent)

        # Retrieve the selected entry and the background color.
        selected = debugger.nav_tree.selected()
        self.bg_color = parent._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])

        # Configure the column and row weights.
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        # Ask the user to select an entry in the navigation tree.
        self.selection_label = ctk.CTkLabel(self, text="Please select an entry in the navigation tree.", font=font)
        if selected is None:
            self.selection_label.grid(row=0, column=0, rowspan=3, columnspan=2, sticky="nsew")

        # Variables storing the indices of the previous and next Gaussian mixture displayed.
        self.prev_gm = -1
        self.next_gm = -1

        # Create all the images and labels.
        empty_image = Image.new(mode="RGB", size=(img_width, img_height), color=(0, 0, 0))

        self._cache = {
            "posterior": {},
            "fixed_components": {},
            "matrices": {},
            "threshold_matrices": {},
        }

        self.images = [
            [PhotoImage(empty_image), PhotoImage(empty_image)],
            [PhotoImage(empty_image), PhotoImage(empty_image)],
        ]
        self.matrix_images = [PhotoImage(empty_image), PhotoImage(empty_image)]

        self.labels = [
            [tk.Label(self, image=image) for image in images]
            for images in self.images
        ]
        self.matrix_labels = [tk.Label(self, image=matrix_image) for matrix_image in self.matrix_images]

        if selected is not None:
            self.update_content(debugger.checkpoints, selected)

    def update_content(self, checkpoints, tags):

        # Retrieve the previous and next Gaussian mixture indices.
        prev_gm = tags[0]
        next_gm = tags[-1]

        # Remove the initial text, and display the images instead.
        if self.selection_label.winfo_viewable():
            self.selection_label.grid_forget()
        if not self.labels[0][0].winfo_viewable():
            for y, labels in enumerate(self.labels):
                for x, label in enumerate(labels):
                    label.grid(row=y, column=x)
                    self.labels[y][x].configure(background=self.bg_color)

        # Update the labels corresponding to the previous Gaussian mixture.
        if prev_gm != self.prev_gm:
            self.update_image(0, 0, self.cache(checkpoints, "posterior", prev_gm))
            self.update_image(1, 0, self.cache(checkpoints, "fixed_components", prev_gm))

        # Update the labels corresponding to the next Gaussian mixture.
        if next_gm != self.next_gm:
            self.update_image(0, 1, self.cache(checkpoints, "posterior", next_gm))
            self.update_image(1, 1, self.cache(checkpoints, "fixed_components", next_gm))

        # Update the matrix of KL-divergences.
        self.update_kl_matrices(checkpoints, prev_gm, next_gm)

        # Update the indices of the previous and next Gaussian mixture.
        self.prev_gm = prev_gm
        self.next_gm = next_gm

    def update_kl_matrices(self, checkpoints, prev_gm, next_gm):

        # Load the matrix images in the cache.
        if prev_gm not in self._cache["matrices"]:
            gm0 = checkpoints[prev_gm]["gm"]
            gm1 = checkpoints[next_gm]["gm"]
            matrix, mask = self.compute_kl_matrix(gm0, gm1)
            self._cache["matrices"][prev_gm] = MatPlotLib.draw_matrix(
                matrix, "KL-divergence between components.", log_scale=True, mask=mask
            )
            matrix = torch.where(matrix < gm0.fixed_gaussian.kl_threshold, 1, 0)
            self._cache["threshold_matrices"][prev_gm] = MatPlotLib.draw_matrix(
                matrix, "KL-divergence below threshold.", mask=mask
            )

        # Display the matrix images in the GUI.
        self.matrix_images[0] = FigureCanvasTkAgg(self._cache["matrices"][prev_gm], master=self)
        self.matrix_labels[0] = self.matrix_images[0].get_tk_widget()
        self.matrix_labels[0].grid(row=2, column=0)
        self.matrix_images[1] = FigureCanvasTkAgg(self._cache["threshold_matrices"][prev_gm], master=self)
        self.matrix_labels[1] = self.matrix_images[1].get_tk_widget()
        self.matrix_labels[1].grid(row=2, column=1)

    @staticmethod
    def compute_kl_matrix(gm0, gm1):
        matrix = torch.zeros([len(gm0.m_hat), len(gm1.m_hat)])
        mask = torch.zeros([len(gm0.m_hat), len(gm1.m_hat)])
        ks0 = gm0.active_components
        ks1 = gm1.active_components
        for i0 in range(len(gm0.m_hat)):
            precision0 = gm0.W_hat[i0] * gm0.v_hat[i0]
            for i1 in range(len(gm1.m_hat)):
                precision1 = gm1.W_hat[i1] * gm1.v_hat[i1]
                matrix[i0][i1] = GaussianStability.kl_gaussian(
                    gm0.m_hat[i0], precision0, gm1.m_hat[i1], precision1
                )
                if i0 not in ks0 or i1 not in ks1:
                    mask[i0][i1] = 1
        return matrix, mask

    def cache(self, checkpoints, distribution_type, gm_id):

        # If image not in cache, compute and cache all the images corresponding to the Gaussian mixture index.
        if gm_id not in self._cache[distribution_type].keys():
            gm = checkpoints[gm_id]["gm"]
            x = checkpoints[gm_id]["gm_data"].get()
            r = gm.compute_responsibilities(x, "posterior")
            self._cache["posterior"][gm_id] = gm.draw_distribution(x, r, "posterior")
            self._cache["fixed_components"][gm_id] = gm.draw_fixed_components(x, r, "posterior")

        # Return the cached image.
        return self._cache[distribution_type][gm_id]

    def update_image(self, y, x, image):
        self.images[y][x] = FigureCanvasTkAgg(image, master=self)
        self.labels[y][x] = self.images[y][x].get_tk_widget()
        self.labels[y][x].grid(row=y, column=x if x < 2 else x + 1)
