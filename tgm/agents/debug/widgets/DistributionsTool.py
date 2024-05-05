from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import customtkinter as ctk
import tkinter as tk
from PIL import Image
from PIL.ImageTk import PhotoImage


class DistributionsTool(ctk.CTkFrame):

    def __init__(self, parent, debugger, font=("Helvetica", 30, "normal"), img_width=720, img_height=480):

        super().__init__(parent)

        # Retrieve the selected entry and the background color.
        selected = debugger.nav_tree.selected()
        self.bg_color = parent._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])

        # Configure the column and row weights.
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, weight=1)
        self.columnconfigure(4, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        # Ask the user to select an entry in the navigation tree.
        self.selection_label = ctk.CTkLabel(self, text="Please select an entry in the navigation tree.", font=font)
        if selected is None:
            self.selection_label.grid(row=0, column=0, rowspan=3, columnspan=5, sticky="nsew")

        # Variables storing the indices of the previous and next Gaussian mixture displayed.
        self.prev_gm = -1
        self.next_gm = -1

        # Create all the images and labels.
        empty_image = Image.new(mode="RGB", size=(img_width, img_height), color=(0, 0, 0))

        self._cache = {
            "distributions": {
                "prior": {},
                "empirical_prior": {},
                "posterior": {}
            },
            "responsibilities": {
                "prior": {},
                "empirical_prior": {},
                "posterior": {}
            }
        }

        self.images = [
            [PhotoImage(empty_image), PhotoImage(empty_image), PhotoImage(empty_image), PhotoImage(empty_image)],
            [PhotoImage(empty_image), PhotoImage(empty_image), PhotoImage(empty_image), PhotoImage(empty_image)],
            [PhotoImage(empty_image), PhotoImage(empty_image), PhotoImage(empty_image), PhotoImage(empty_image)]
        ]

        self.labels = [
            [tk.Label(self, image=image) for image in images]
            for images in self.images
        ]

        if selected is not None:
            prev_gm, next_gm, fit_id = selected
            self.update_content(
                debugger.gms, debugger.xs, debugger.init_params, int(prev_gm), int(next_gm), int(fit_id)
            )

    def update_content(self, gms, xs, init_params, prev_gm, next_gm, fit_id):

        # Remove the initial text, and display the images instead.
        if self.selection_label.winfo_viewable():
            self.selection_label.grid_forget()
        if not self.labels[0][0].winfo_viewable():
            for y, labels in enumerate(self.labels):
                for x, label in enumerate(labels):
                    label.grid(row=y, column=x if x < 2 else x + 1)
                    self.labels[y][x].configure(background=self.bg_color)

        # Update the labels corresponding to the previous Gaussian mixture.
        if prev_gm != self.prev_gm:
            self.update_image(0, 0, self.cache(gms, xs, "distributions", "prior", prev_gm))
            self.update_image(0, 1, self.cache(gms, xs, "responsibilities", "prior", prev_gm))
            self.update_image(1, 0, self.cache(gms, xs, "distributions", "empirical_prior", prev_gm))
            self.update_image(1, 1, self.cache(gms, xs, "responsibilities", "empirical_prior", prev_gm))
            self.update_image(2, 0, self.cache(gms, xs, "distributions", "posterior", prev_gm))
            self.update_image(2, 1, self.cache(gms, xs, "responsibilities", "posterior", prev_gm))

        # Update the labels corresponding to the next Gaussian mixture.
        if next_gm != self.next_gm:
            self.update_image(0, 2, self.cache(gms, xs, "distributions", "prior", next_gm))
            self.update_image(0, 3, self.cache(gms, xs, "responsibilities", "prior", next_gm))
            self.update_image(1, 2, self.cache(gms, xs, "distributions", "empirical_prior", next_gm))
            self.update_image(1, 3, self.cache(gms, xs, "responsibilities", "empirical_prior", next_gm))
            self.update_image(2, 2, self.cache(gms, xs, "distributions", "posterior", next_gm))
            self.update_image(2, 3, self.cache(gms, xs, "responsibilities", "posterior", next_gm))

        # Update the indices of the previous and next Gaussian mixture.
        self.prev_gm = prev_gm
        self.next_gm = next_gm

    def cache(self, gms, xs, graph_type, distribution_type, gm_id):

        # If image not in cache, compute and cache all the images corresponding to the Gaussian mixture index.
        if gm_id not in self._cache[graph_type][distribution_type].keys():
            gm = gms[gm_id]
            x = xs[gm_id]
            r = gm.compute_responsibilities(x, "prior")
            self._cache["distributions"]["prior"][gm_id] = gm.draw_distribution(x, r, "prior")
            self._cache["responsibilities"]["prior"][gm_id] = gm.draw_responsibilities(r, "prior")
            r = gm.compute_responsibilities(x, "empirical_prior")
            self._cache["distributions"]["empirical_prior"][gm_id] = gm.draw_distribution(x, r, "empirical_prior")
            self._cache["responsibilities"]["empirical_prior"][gm_id] = gm.draw_responsibilities(r, "empirical_prior")
            r = gm.compute_responsibilities(x, "posterior")
            self._cache["distributions"]["posterior"][gm_id] = gm.draw_distribution(x, r, "posterior")
            self._cache["responsibilities"]["posterior"][gm_id] = gm.draw_responsibilities(r, "posterior")

        # Return the cached image.
        return self._cache[graph_type][distribution_type][gm_id]

    def update_image(self, y, x, image):
        self.images[y][x] = FigureCanvasTkAgg(image, master=self)
        self.labels[y][x] = self.images[y][x].get_tk_widget()
        self.labels[y][x].grid(row=y, column=x if x < 2 else x + 1)
