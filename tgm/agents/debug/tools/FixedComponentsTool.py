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

        # Update cache.
        self.cache(checkpoints, prev_gm, next_gm)

        # Update the labels corresponding to the previous Gaussian mixture.
        if prev_gm != self.prev_gm:
            self.update_image(0, 0, self._cache["posterior"][prev_gm])
            self.update_image(1, 0, self._cache["fixed_components"][prev_gm])

        # Update the labels corresponding to the next Gaussian mixture.
        if next_gm != self.next_gm:
            self.update_image(0, 1, self._cache["posterior"][next_gm])
            self.update_image(1, 1, self._cache["fixed_components"][next_gm])

        # Update the matrix of KL-divergences.
        if prev_gm != self.prev_gm:
            self.matrix_images[0] = FigureCanvasTkAgg(self._cache["matrices"][prev_gm], master=self)
            self.matrix_labels[0] = self.matrix_images[0].get_tk_widget()
            self.matrix_labels[0].grid(row=2, column=0)
            self.matrix_images[1] = FigureCanvasTkAgg(self._cache["threshold_matrices"][prev_gm], master=self)
            self.matrix_labels[1] = self.matrix_images[1].get_tk_widget()
            self.matrix_labels[1].grid(row=2, column=1)

        # Update the indices of the previous and next Gaussian mixture.
        self.prev_gm = prev_gm
        self.next_gm = next_gm

    @staticmethod
    def compute_kl_matrix(gm0, r0, gm1, r1):
        if r0 is None or r1 is None:
            return None, None

        matrix = torch.zeros([len(gm0.m_hat), len(gm1.m_hat)])
        mask = torch.zeros([len(gm0.m_hat), len(gm1.m_hat)])
        ks0 = [i for i, N_k in enumerate(r0.sum(dim=0)) if N_k != 0.0]
        ks1 = [i for i, N_k in enumerate(r1.sum(dim=0)) if N_k != 0.0]
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

    def cache(self, checkpoints, prev_id, next_id):
        gm0 = checkpoints[prev_id]["gm"].fixed_gaussian
        gm1 = checkpoints[next_id]["gm"].fixed_gaussian
        x0 = checkpoints[prev_id]["gm_data"].get()
        x1 = checkpoints[next_id]["gm_data"].get()
        r0 = gm0.compute_responsibilities(x0)
        r1 = gm1.compute_responsibilities(x1)

        # If image not in cache, compute and cache all the images corresponding to the Gaussian mixture index.
        if prev_id not in self._cache["posterior"].keys():
            self._cache["posterior"][prev_id] = gm0.draw_distribution(x0, r0)
            self._cache["fixed_components"][prev_id] = gm0.draw_fixed_components(x0, r0)

        # If image not in cache, compute and cache all the images corresponding to the Gaussian mixture index.
        if next_id not in self._cache["posterior"].keys():
            self._cache["posterior"][next_id] = gm1.draw_distribution(x1, r1)
            self._cache["fixed_components"][next_id] = gm1.draw_fixed_components(x1, r1)

        # Load the matrix images in the cache.
        if prev_id not in self._cache["matrices"].keys():
            matrix, mask = self.compute_kl_matrix(gm0, r0, gm1, r1)
            if matrix is None or mask is None:
                time = "0" if r0 is None else "1"
                text = f"Matrix cannot be generated because \n responsibilities at time {time} are none."
                self._cache["matrices"][prev_id] = MatPlotLib.draw_text(text)
                self._cache["threshold_matrices"][prev_id] = MatPlotLib.draw_text(text)
            else:
                self._cache["matrices"][prev_id] = MatPlotLib.draw_matrix(
                    matrix, "KL-divergence between components.", log_scale=True, mask=mask
                )
                matrix = torch.where(matrix < gm0.kl_threshold, 1, 0)
                self._cache["threshold_matrices"][prev_id] = MatPlotLib.draw_matrix(
                    matrix, "KL-divergence below threshold.", mask=mask
                )

    def update_image(self, y, x, image):
        self.images[y][x] = FigureCanvasTkAgg(image, master=self)
        self.labels[y][x] = self.images[y][x].get_tk_widget()
        self.labels[y][x].grid(row=y, column=x if x < 2 else x + 1)
