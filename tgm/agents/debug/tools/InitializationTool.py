import tkinter as tk

import torch
from PIL import Image
from PIL.ImageTk import PhotoImage
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tgm.agents.models.display.MatPlotLib import MatPlotLib


class InitializationTool(ctk.CTkFrame):

    def __init__(self, parent, debugger, font=("Helvetica", 30, "normal"), img_width=720, img_height=480):

        super().__init__(parent)

        # Retrieve the selected entry and the background color.
        selected = debugger.nav_tree.selected()
        self.bg_color = parent._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])

        # Configure the column and row weights.
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(0, weight=1)

        # Ask the user to select an entry in the navigation tree.
        self.selection_label = ctk.CTkLabel(self, text="Please select an entry in the navigation tree.", font=font)
        if selected is None:
            self.selection_label.grid(row=0, column=0, rowspan=3, columnspan=2, sticky="nsew")

        # Variables storing the indices of the previous and next Gaussian mixture displayed.
        self.fit_id = -1

        # Create all the images and labels.
        empty_image = Image.new(mode="RGB", size=(img_width, img_height), color=(0, 0, 0))

        self._cache = {
            "fixed": {},
            "flexible": {},
            "combined": {},
        }

        self.images = [
            [PhotoImage(empty_image), PhotoImage(empty_image), PhotoImage(empty_image)],
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
                    label.grid(row=y, column=x)
                    self.labels[y][x].configure(background=self.bg_color)

        # Extract the parameters corresponding to the current fit.
        params = init_params[fit_id]

        # Update the labels corresponding to the previous Gaussian mixture.
        if fit_id != self.fit_id:
            self.update_image(0, self.cache(xs[next_gm], params, "fixed", fit_id))
            self.update_image(1, self.cache(xs[next_gm], params, "flexible", fit_id))
            self.update_image(2, self.cache(xs[next_gm], params, "combined", fit_id))

        # Update the indices of the previous and next Gaussian mixture.
        self.fit_id = fit_id

    def cache(self, x, params, distribution_type, fit_id):

        # If image not in cache, compute and cache all the images corresponding to the Gaussian mixture index.
        if fit_id not in self._cache[distribution_type].keys():

            # Cache the fixed component image.
            v, d, β, m, W = params["fixed"]
            title = "Fixed components"
            if v is None or d is None or β is None or m is None or W is None:
                self._cache["fixed"][fit_id] = MatPlotLib.draw_data_points(x, title)
                fixed_colors = []
            else:
                p = (m, β, v, W)
                fixed_colors = ["black" for _ in range(len(m))]
                active_components = [k for k in range(len(m))]
                self._cache["fixed"][fit_id] = MatPlotLib.draw_ellipses(
                    x, active_components=active_components, params=p, all_colors=fixed_colors, title=title
                )

            # Cache the flexible component image.
            v, d, β, m, W = params["flexible"]
            title = "Flexible components"
            if v is None or d is None or β is None or m is None or W is None:
                self._cache["flexible"][fit_id] = MatPlotLib.draw_data_points(x, title)
                flexible_colors = []
            else:
                p = (m, β, v, W)
                flexible_colors = ["red" for _ in range(len(m))]
                active_components = [k for k in range(len(m))]
                self._cache["flexible"][fit_id] = MatPlotLib.draw_ellipses(
                    x, active_components=active_components, params=p, all_colors=flexible_colors, title=title
                )

            # Cache the combined component image.
            v, d, β, m, W = params["combined"]
            title = "Combined components"
            if v is None or d is None or β is None or m is None or W is None:
                self._cache["combined"][fit_id] = MatPlotLib.draw_data_points(x, title)
            else:
                p = (m, β, v, W)
                all_colors = fixed_colors + flexible_colors
                active_components = [k for k in range(len(m))]
                self._cache["combined"][fit_id] = MatPlotLib.draw_ellipses(
                    x, active_components=active_components, params=p, all_colors=all_colors, title=title
                )

        # Return the cached image.
        return self._cache[distribution_type][fit_id]

    def update_image(self, x, image):
        self.images[0][x] = FigureCanvasTkAgg(image, master=self)
        self.labels[0][x] = self.images[0][x].get_tk_widget()
        self.labels[0][x].grid(row=0, column=x)
