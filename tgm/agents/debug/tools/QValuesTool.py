from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
from PIL.ImageTk import PhotoImage

import customtkinter as ctk
import tkinter as tk


class QValuesTool(ctk.CTkFrame):

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
        self.rowconfigure(2, weight=1)

        # Ask the user to select an entry in the navigation tree.
        self.selection_label = ctk.CTkLabel(self, text="Please select an entry in the navigation tree.", font=font)
        if selected is None:
            self.selection_label.grid(row=0, column=0, sticky="nsew")

        # Variables storing the indices of the previous and next Gaussian mixture displayed.
        self.prev_planner = -1
        self.next_planner = -1

        # The cached of images.
        self._cache = {
            "q_values": {}
        }

        # Create the matrix images.
        empty_image = Image.new(mode="RGB", size=(img_width, img_height), color=(0, 0, 0))
        self.q_images = [PhotoImage(empty_image), PhotoImage(empty_image)]
        self.q_labels = [tk.Label(self, image=image) for image in self.q_images]

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

        # Update the equations figures.
        prev_planner = tags[0]
        next_planner = tags[-1]

        # Update the image corresponding to the previous planner.
        if prev_planner != self.prev_planner:
            self.update_image(0, self.cache(checkpoints, prev_planner))

        # Update the image corresponding to the next planner.
        if next_planner != self.next_planner:
            self.update_image(1, self.cache(checkpoints, next_planner))

        # Update the indices of the previous and next Gaussian mixture.
        self.prev_planner = prev_planner
        self.next_planner = next_planner

    def cache(self, checkpoints, planner_id):

        # If image not in cache, compute and cache all the images corresponding to the Gaussian mixture index.
        if planner_id not in self._cache["q_values"].keys():
            planner = checkpoints[planner_id]["planner"]
            self._cache["q_values"][planner_id] = planner.draw_q_values(self.action_names)

        # Return the cached image.
        return self._cache["q_values"][planner_id]

    def update_image(self, x, image):
        self.q_images[x] = FigureCanvasTkAgg(image, master=self)
        self.q_labels[x] = self.q_images[x].get_tk_widget()
        self.q_labels[x].grid(row=0, column=2 * x + 1)
