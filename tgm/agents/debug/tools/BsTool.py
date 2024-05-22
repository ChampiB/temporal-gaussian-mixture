from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
from PIL.ImageTk import PhotoImage

import customtkinter as ctk
import tkinter as tk


class BsTool(ctk.CTkFrame):

    def __init__(
        self, parent, debugger, font=("Helvetica", 30, "normal"), img_width=720, img_height=480, action_names=None
    ):

        super().__init__(parent)

        # Store the names of the action that the agent can take.
        self.action_names = action_names

        # Store the debugger, retrieve the selected entry and the background/text colors.
        self.debugger = debugger
        selected = debugger.nav_tree.selected()
        self.root = parent
        self.bg_color = parent._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
        self.text_color = parent._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"])

        # Configure the column and row weights.
        self.columnconfigure(0, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(4, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        # Ask the user to select an entry in the navigation tree.
        self.selection_label = ctk.CTkLabel(self, text="Please select an entry in the navigation tree.", font=font)
        if selected is None:
            self.selection_label.grid(row=0, column=0, sticky="nsew")

        # Create the combo box to allow the user to select a distribution.
        self.distribution0_frame = ctk.CTkFrame(self, bg_color=self.bg_color, fg_color=self.bg_color)
        self.distribution0_frame.columnconfigure(0, weight=1)
        self.distribution0_frame.columnconfigure(1, weight=1)
        self.distribution0_label = ctk.CTkLabel(self.distribution0_frame, text="distribution:", font=font)
        self.distribution0_combobox = ctk.CTkComboBox(
            self.distribution0_frame, values=["prior", "empirical prior", "posterior"],
            font=font, dropdown_font=font, command=self.update_tool
        )
        self.distribution0_label.grid(row=0, column=0, sticky="nse", padx=(15, 5))
        self.distribution0_combobox.grid(row=0, column=1, sticky="nsew", padx=(5, 15))

        self.distribution1_frame = ctk.CTkFrame(self, bg_color=self.bg_color, fg_color=self.bg_color)
        self.distribution1_frame.columnconfigure(0, weight=1)
        self.distribution1_frame.columnconfigure(1, weight=1)
        self.distribution1_label = ctk.CTkLabel(self.distribution1_frame, text="distribution:", font=font)
        self.distribution1_combobox = ctk.CTkComboBox(
            self.distribution1_frame, values=["prior", "empirical prior", "posterior"],
            font=font, dropdown_font=font, command=self.update_tool
        )
        self.distribution1_label.grid(row=0, column=0, sticky="nse", padx=(15, 5))
        self.distribution1_combobox.grid(row=0, column=1, sticky="nsew", padx=(5, 15))

        # Variables storing the indices of the previous and next Gaussian mixture displayed.
        self.prev_tm = -1
        self.next_tm = -1

        # Variables storing the type of distribution displayed on the right and left hand side.
        self.prev_distribution = "prior"
        self.next_distribution = "prior"

        # The cached of images.
        self._cache = {
            "prior": {},
            "empirical_prior": {},
            "posterior": {}
        }

        # Create the matrix images.
        empty_image = Image.new(mode="RGB", size=(img_width, img_height), color=(0, 0, 0))
        self.b0_frame = ctk.CTkFrame(self, bg_color=self.bg_color, fg_color=self.bg_color)
        self.b0_images = [
            [PhotoImage(empty_image), PhotoImage(empty_image)],
            [PhotoImage(empty_image), PhotoImage(empty_image)],
            [PhotoImage(empty_image), None]
        ]
        self.b0_labels = [
            [None if image is None else tk.Label(self.b0_frame, image=image) for image in images]
            for images in self.b0_images
        ]
        self.b1_frame = ctk.CTkFrame(self, bg_color=self.bg_color, fg_color=self.bg_color)
        self.b1_images = [
            [PhotoImage(empty_image), PhotoImage(empty_image)],
            [PhotoImage(empty_image), PhotoImage(empty_image)],
            [PhotoImage(empty_image), None]
        ]
        self.b1_labels = [
            [None if image is None else tk.Label(self.b1_frame, image=image) for image in images]
            for images in self.b1_images
        ]

        # Update the tool content, if an entry is selected in the navigation tree.
        if selected is not None:
            self.update_content(debugger.checkpoints, selected)

    def update_tool(self, _):
        self.update_content(self.debugger.checkpoints, None)

    def update_content(self, checkpoints, tags):

        # Ensure that the tags are correct.
        if tags is None:
            tags = self.debugger.nav_tree.selected()

        # Remove the initial text, and display the images instead.
        if self.selection_label.winfo_viewable():
            self.selection_label.grid_forget()
        if not self.b0_frame.winfo_viewable():
            self.b0_frame.grid(row=1, column=1)
            self.distribution0_frame.grid(row=2, column=1, sticky="nsew", padx=40, pady=20)
            self.b1_frame.grid(row=1, column=3)
            self.distribution1_frame.grid(row=2, column=3, sticky="nsew", padx=40, pady=20)

        # Update the B matrices displayed.
        distribution0 = self.distribution0_combobox.get().replace(" ", "_")
        distribution1 = self.distribution1_combobox.get().replace(" ", "_")
        prev_tm = tags[0]
        next_tm = tags[-1]

        a = 0
        for y in range(len(self.b0_labels)):
            for x in range(len(self.b0_labels[y])):

                # Update the labels corresponding to the previous Gaussian mixture.
                if prev_tm != self.prev_tm or distribution0 != self.prev_distribution:
                    self.update_image(y, x, 0, self.cache(checkpoints, distribution0, prev_tm, a))

                # Update the labels corresponding to the next Gaussian mixture.
                if next_tm != self.next_tm or distribution1 != self.next_distribution:
                    self.update_image(y, x, 1, self.cache(checkpoints, distribution1, next_tm, a))

                a += 1

        # Update the indices of the previous and next Gaussian mixture.
        self.prev_distribution = distribution0
        self.prev_tm = prev_tm
        self.next_distribution = distribution1
        self.next_tm = next_tm

    def to_rgb(self, color):
        return tuple(((c // 256) / 255 for c in self.root.winfo_rgb(color)))

    def cache(self, checkpoints, distribution, tm_id, action):

        # If image not in cache, compute and cache all the images corresponding to the Gaussian mixture index.
        need_cache = False
        if tm_id not in self._cache[distribution].keys():
            need_cache = True
            self._cache["prior"][tm_id] = {}
            self._cache["empirical_prior"][tm_id] = {}
            self._cache["posterior"][tm_id] = {}

        if need_cache or action not in self._cache[distribution][tm_id].keys():
            tm = checkpoints[tm_id]["tm"]
            self._cache["prior"][tm_id][action] = tm.draw_b_matrix(action, "prior", self.action_names)
            self._cache["empirical_prior"][tm_id][action] = tm.draw_b_matrix(action, "empirical_prior", self.action_names)
            self._cache["posterior"][tm_id][action] = tm.draw_b_matrix(action, "posterior", self.action_names)

        # Return the cached image.
        return self._cache[distribution][tm_id][action]

    def update_image(self, y, x, t, image):

        if t == 0:
            if self.b0_images[y][x] is None:
                return
            self.b0_images[y][x] = FigureCanvasTkAgg(image, master=self.b0_frame)
            self.b0_labels[y][x] = self.b0_images[y][x].get_tk_widget()
            self.b0_labels[y][x].grid(row=y, column=x)
        else:
            if self.b1_images[y][x] is None:
                return
            self.b1_images[y][x] = FigureCanvasTkAgg(image, master=self.b1_frame)
            self.b1_labels[y][x] = self.b1_images[y][x].get_tk_widget()
            self.b1_labels[y][x].grid(row=y, column=x)
