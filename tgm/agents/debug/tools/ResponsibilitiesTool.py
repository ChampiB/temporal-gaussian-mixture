import math

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
from PIL.ImageTk import PhotoImage
from torch import softmax, digamma, logdet, tensor

from tgm.agents.models.inference.GaussianMixture import GaussianMixture as GMix

import customtkinter as ctk
import tkinter as tk

from tgm.agents.debug.widgets.Shell import Shell
from tgm.agents.models.display.MatPlotLib import MatPlotLib


class ResponsibilitiesTool(ctk.CTkFrame):

    def __init__(self, parent, debugger, font=("Helvetica", 30, "normal"), img_width=720, img_height=480):

        super().__init__(parent)

        # Store the debugger, retrieve the selected entry and the background color.
        self.debugger = debugger
        selected = debugger.nav_tree.selected()
        self.root = parent
        self.bg_color = parent._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
        self.text_color = parent._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"])

        # Configure the column and row weights.
        for i in range(6):
            self.columnconfigure(i, weight=1)
        self.columnconfigure(6, weight=6)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(3, weight=1)

        # Ask the user to select an entry in the navigation tree.
        self.selection_label = ctk.CTkLabel(self, text="Please select an entry in the navigation tree.", font=font)
        if selected is None:
            self.selection_label.grid(row=0, column=0, sticky="nsew")

        # Create the labels and entries prompting the user to pick (x, y) coordinates.
        fg_color = parent._apply_appearance_mode(self.cget("fg_color"))
        valid_coordinate = parent.register(self.valid_coordinate)

        self.x_label = ctk.CTkLabel(self, text="x:", font=font)
        self.x_coord = tk.StringVar()
        self.x_coord.set("1")
        self.x_entry = ctk.CTkEntry(
            master=self, font=font, border_width=1, corner_radius=0, fg_color=fg_color,
            textvariable=self.x_coord
        )
        self.x_entry.configure(validate="key", validatecommand=(valid_coordinate, "%P"))
        self.x_entry.bind("<Return>", self.update_tool)

        self.y_label = ctk.CTkLabel(self, text="y:", font=font)
        self.y_coord = tk.StringVar()
        self.y_coord.set("1")
        self.y_entry = ctk.CTkEntry(
            master=self, font=font, border_width=1, corner_radius=0, fg_color=fg_color,
            textvariable=self.y_coord
        )
        self.y_entry.configure(validate="key", validatecommand=(valid_coordinate, "%P"))
        self.y_entry.bind("<Return>", self.update_tool)

        # Create the combo box to allow the user to select a distribution.
        self.distribution_label = ctk.CTkLabel(self, text="distribution:", font=font)
        distributions = ["prior", "empirical prior", "posterior"]
        self.distribution_combobox = ctk.CTkComboBox(
            self, values=distributions, font=font, dropdown_font=font, command=self.update_tool
        )

        # Create the figure.
        empty_image = Image.new(mode="RGB", size=(img_width, img_height), color=(0, 0, 0))
        self.image = PhotoImage(empty_image)
        self.label = tk.Label(self, image=self.image)

        # Create the equations.
        self.eq_frame = ctk.CTkFrame(self, bg_color=self.bg_color, fg_color=self.bg_color)
        self.eq_images = [PhotoImage(empty_image) for _ in range(7)]
        self.eq_labels = [tk.Label(self.eq_frame, image=image) for image in self.eq_images]

        # Update the tool content, if an entry is selected in the navigation tree.
        if selected is not None:
            self.update_content(debugger.checkpoints, selected)

    @staticmethod
    def valid_coordinate(text):
        try:
            if text == "":
                return True
            float(text)
            return True
        except ValueError:
            return False

    def update_tool(self, _):
        self.update_content(self.debugger.checkpoints, None)

    def update_content(self, checkpoints, tags):

        # Ensure that the tags are correct.
        if tags is None:
            tags = self.debugger.nav_tree.selected()

        # Remove the initial text, and display the images instead.
        if self.selection_label.winfo_viewable():
            self.selection_label.grid_forget()
        if not self.x_label.winfo_viewable():
            self.x_label.grid(row=2, column=0, sticky="nse", padx=(15, 5))
            self.x_entry.grid(row=2, column=1, sticky="nsew", padx=(5, 15))
            self.y_label.grid(row=2, column=2, sticky="nse", padx=(15, 5))
            self.y_entry.grid(row=2, column=3, sticky="nsew", padx=(5, 15))
            self.distribution_label.grid(row=2, column=4, sticky="nse", padx=(15, 5))
            self.distribution_combobox.grid(row=2, column=5, sticky="nsew", padx=(5, 15))
            self.label.grid(row=1, column=0, columnspan=6)
            self.label.configure(background=self.bg_color)
            self.eq_frame.grid(row=1, column=6)

        # Update the responsibility figure.
        distribution = self.distribution_combobox.get().replace(" ", "_")
        x_coord = self.x_coord.get()
        x_coord = 0 if x_coord == "" else float(x_coord)
        y_coord = self.y_coord.get()
        y_coord = 0 if y_coord == "" else float(y_coord)

        gm_id = tags[0]
        gm = checkpoints[gm_id]["gm"]
        x = checkpoints[gm_id]["gm_data"].get()
        r = gm.compute_responsibilities(x, distribution)

        datum = Shell.near(x, x, [x_coord, y_coord], 1)

        image = gm.draw_distribution(x, r, distribution, ellipses=False, datum=datum)
        self.image = FigureCanvasTkAgg(image, master=self)
        self.label = self.image.get_tk_widget()
        self.label.grid(row=1, column=0, columnspan=6)
        self.label.configure(background=self.bg_color)

        # Update the equations figures.
        if distribution == "prior":
            m, β, W, v, d = gm.m, gm.β, gm.W, gm.v, gm.d
        elif distribution == "empirical_prior":
            m, β, W, v, d = gm.m_bar, gm.β_bar, gm.W_bar, gm.v_bar, gm.d_bar
        elif distribution == "posterior":
            m, β, W, v, d = gm.m_hat, gm.β_hat, gm.W_hat, gm.v_hat, gm.d_hat
        else:
            raise RuntimeError(f"[Error]: distribution type not supported '{distribution}'.")

        n_states = len(W)
        log_det_W = []
        digamma_sum = []
        for k in range(n_states):
            digamma_sum.append(sum([digamma((v[k] + 1 - i) / 2) for i in range(n_states)]))
            log_det_W.append(logdet(W[k]))
        f_v = tensor(digamma_sum).unsqueeze(dim=0)
        g_W = tensor(log_det_W).unsqueeze(dim=0)
        log_det_Λ = GMix.expected_log_det_Λ(W, v).unsqueeze(dim=0)
        log_D = GMix.expected_log_D(d).unsqueeze(dim=0)
        quadratic_form = GMix.expected_quadratic_form(datum, m, β, W, v)
        log_ρ = log_D - 0.5 * (len(W) * math.log(2 * math.pi) - log_det_Λ + quadratic_form)
        r = softmax(log_ρ, dim=1)

        matrices = {
            r"\mathbb{E}[\ln \mathbb{D}]": log_D,
            r"f(v)": f_v,
            r"g(W)": g_W,
            r"\mathbb{E}[\ln |\boldsymbol{\Lambda}|] = f(v) + g(W) + C": log_det_Λ,
            r"\mathbb{E}[(x - \mathbb{\mu})^\top \boldsymbol{\Lambda} (x - \mathbb{\mu})]": quadratic_form,
            r"\ln \rho": log_ρ,
            r"\hat{r}": r
        }
        for i, (name, matrix) in enumerate(matrices.items()):
            image = MatPlotLib.draw_equation(
                f"{name} = " + Shell.to_latex_format(matrix),
                self.to_rgb(self.bg_color), self.to_rgb(self.text_color), 50, 10
            )
            self.display_image(image, i)

    def to_rgb(self, color):
        return tuple(((c // 256) / 255 for c in self.root.winfo_rgb(color)))

    def display_image(self, image, i):
        image = FigureCanvasTkAgg(image, master=self.eq_frame)
        self.eq_images[i] = image
        self.eq_labels[i].grid_forget()
        self.eq_labels[i] = image.get_tk_widget()
        self.eq_labels[i].grid(row=i, column=0, sticky="w", padx=15, pady=15)
