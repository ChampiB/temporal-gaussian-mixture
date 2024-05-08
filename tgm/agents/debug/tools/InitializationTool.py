import tkinter as tk

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
            self.selection_label.grid(row=0, column=0, rowspan=1, columnspan=3, sticky="nsew")

        # Variables storing the indices of the previous and next Gaussian mixture displayed.
        self.fixed_gm_id = -1

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
            self.update_content(debugger.checkpoints, selected)

    def update_content(self, checkpoints, tags):

        # Remove the initial text, and display the images instead.
        if self.selection_label.winfo_viewable():
            self.selection_label.grid_forget()
        if not self.labels[0][0].winfo_viewable():
            for y, labels in enumerate(self.labels):
                for x, label in enumerate(labels):
                    label.grid(row=y, column=x)
                    self.labels[y][x].configure(background=self.bg_color)

        # Extract the parameters corresponding to the current fit.
        fixed_gm_id, flexible_gm_id, combined_gm_id = tags
        fixed_gm = checkpoints[fixed_gm_id]["gm"]
        flexible_gm = checkpoints[flexible_gm_id]["gm"]
        combined_gm = checkpoints[combined_gm_id]["gm"]
        params = {
            "fixed": (fixed_gm.v_fixed, fixed_gm.d_fixed, fixed_gm.β_fixed, fixed_gm.m_fixed, fixed_gm.W_fixed),
            "flexible": (flexible_gm.v, flexible_gm.d, flexible_gm.β, flexible_gm.m, flexible_gm.W),
            "combined": (combined_gm.v, combined_gm.d, combined_gm.β, combined_gm.m, combined_gm.W)
        }

        # Update the labels corresponding to the previous Gaussian mixture.
        if fixed_gm_id != self.fixed_gm_id:
            x = checkpoints[combined_gm_id]["gm_data"].get()
            self.update_image(0, self.cache(x, params, "fixed", fixed_gm_id))
            self.update_image(1, self.cache(x, params, "flexible", fixed_gm_id))
            self.update_image(2, self.cache(x, params, "combined", fixed_gm_id))

        # Update the indices of the previous and next Gaussian mixture.
        self.fixed_gm_id = fixed_gm_id

    def cache(self, x, params, distribution_type, fixed_gm_id):

        # If image not in cache, compute and cache all the images corresponding to the Gaussian mixture index.
        if fixed_gm_id not in self._cache[distribution_type].keys():

            # Cache the fixed component image.
            v, d, β, m, W = params["fixed"]
            title = "Fixed components"
            if v is None or d is None or β is None or m is None or W is None:
                self._cache["fixed"][fixed_gm_id] = MatPlotLib.draw_data_points(x, title)
                fixed_colors = []
            else:
                p = (m, β, W, v)
                fixed_colors = ["black" for _ in range(len(m))]
                active_components = [k for k in range(len(m))]
                self._cache["fixed"][fixed_gm_id] = MatPlotLib.draw_ellipses(
                    x, active_components=active_components, params=p, all_colors=fixed_colors, title=title
                )

            # Cache the flexible component image.
            v, d, β, m, W = params["flexible"]
            title = "Flexible components"
            if v is None or d is None or β is None or m is None or W is None:
                self._cache["flexible"][fixed_gm_id] = MatPlotLib.draw_data_points(x, title)
                flexible_colors = []
            else:
                p = (m, β, W, v)
                flexible_colors = ["red" for _ in range(len(m))]
                active_components = [k for k in range(len(m))]
                self._cache["flexible"][fixed_gm_id] = MatPlotLib.draw_ellipses(
                    x, active_components=active_components, params=p, all_colors=flexible_colors, title=title
                )

            # Cache the combined component image.
            v, d, β, m, W = params["combined"]
            title = "Combined components"
            if v is None or d is None or β is None or m is None or W is None:
                self._cache["combined"][fixed_gm_id] = MatPlotLib.draw_data_points(x, title)
            else:
                p = (m, β, W, v)
                all_colors = fixed_colors + flexible_colors
                active_components = [k for k in range(len(m))]
                self._cache["combined"][fixed_gm_id] = MatPlotLib.draw_ellipses(
                    x, active_components=active_components, params=p, all_colors=all_colors, title=title
                )

        # Return the cached image.
        return self._cache[distribution_type][fixed_gm_id]

    def update_image(self, x, image):
        self.images[0][x] = FigureCanvasTkAgg(image, master=self)
        self.labels[0][x] = self.images[0][x].get_tk_widget()
        self.labels[0][x].grid(row=0, column=x)
