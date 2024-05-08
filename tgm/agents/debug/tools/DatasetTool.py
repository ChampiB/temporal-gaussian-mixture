import tkinter as tk

from PIL import Image
from PIL.ImageTk import PhotoImage
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tgm.agents.models.display.MatPlotLib import MatPlotLib


class DatasetTool(ctk.CTkFrame):

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
            self.selection_label.grid(row=0, column=0, rowspan=1, columnspan=3, sticky="nsew")

        # Variables storing the indices of the previous and next Gaussian mixture displayed.
        self.prev_gm = -1
        self.next_gm = -1

        # Create all the images and labels.
        empty_image = Image.new(mode="RGB", size=(img_width, img_height), color=(0, 0, 0))

        self._cache = {
            "forget": {},
            "keep": {},
            "all": {},
        }

        self.images = [
            [PhotoImage(empty_image), PhotoImage(empty_image)],
            [PhotoImage(empty_image), PhotoImage(empty_image)],
            [PhotoImage(empty_image), PhotoImage(empty_image)],
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

        # Retrieve the previous and next Gaussian mixture indices.
        prev_gm = tags[0]
        next_gm = tags[-1]

        # Update the labels corresponding to the previous Gaussian mixture.
        if prev_gm != self.prev_gm:
            self.update_image(0, 0, self.cache(checkpoints[prev_gm]["gm_data"], "forget", prev_gm))
            self.update_image(0, 1, self.cache(checkpoints[prev_gm]["gm_data"], "keep", prev_gm))
            self.update_image(0, 2, self.cache(checkpoints[prev_gm]["gm_data"], "all", prev_gm))

        # Update the labels corresponding to the previous Gaussian mixture.
        if next_gm != self.next_gm:
            self.update_image(1, 0, self.cache(checkpoints[next_gm]["gm_data"], "forget", next_gm))
            self.update_image(1, 1, self.cache(checkpoints[next_gm]["gm_data"], "keep", next_gm))
            self.update_image(1, 2, self.cache(checkpoints[next_gm]["gm_data"], "all", next_gm))

        # Update the indices of the previous and next Gaussian mixture.
        self.prev_gm = prev_gm
        self.next_gm = next_gm

    def cache(self, gm_data, distribution_type, gm_id):

        # If image not in cache, compute and cache all the images corresponding to the Gaussian mixture index.
        if gm_id not in self._cache[distribution_type].keys():

            # Retrieve all the data points.
            x = gm_data.get()
            x_forget, x_keep = gm_data.get(split=True)

            # Cache the fixed component image.
            title = "Data points to forget"
            self._cache["forget"][gm_id] = MatPlotLib.draw_data_points(
                [x_keep, x_forget], title, data_colors=["white", "gray"]
            )

            # Cache the flexible component image.
            title = "Data points to keep"
            self._cache["keep"][gm_id] = MatPlotLib.draw_data_points(
                [x_forget, x_keep], title, data_colors=["white", "gray"]
            )

            # Cache the combined component image.
            title = "All the data points"
            self._cache["all"][gm_id] = MatPlotLib.draw_data_points(x, title)

        # Return the cached image.
        return self._cache[distribution_type][gm_id]

    def update_image(self, x, y, image):
        self.images[y][x] = FigureCanvasTkAgg(image, master=self)
        self.labels[y][x] = self.images[y][x].get_tk_widget()
        self.labels[y][x].grid(row=y, column=x)
