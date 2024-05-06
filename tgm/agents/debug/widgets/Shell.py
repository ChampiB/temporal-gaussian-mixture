import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import customtkinter as ctk

from tgm.agents.models.display.MatPlotLib import MatPlotLib
from tgm.agents.models.inference.KMeans import KMeans


class Shell(ctk.CTkFrame):

    def __init__(self, root, master, debugger, font=("Helvetica", 30, "normal")):

        super().__init__(master)

        # Store the debugger, root, and Gaussian mixture indices.
        self.debugger = debugger
        self.root = root
        self.prev_gm = -1
        self.next_gm = -1

        # Create the shell display options.
        self.font = font
        self.padding = 30
        self.error_color = "red4"
        self.warning_color = "DarkOrange2"
        self.bg_color = root._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
        self.text_color = root._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"])

        # Configure rows and columns.
        self.rowconfigure(0, weight=50)
        self.rowconfigure(1, weight=1)
        for i in range(10):
            self.columnconfigure(i, weight=1)

        # Create the shell area.
        self.shell_frame = ctk.CTkScrollableFrame(self, corner_radius=0)
        self.shell_frame.grid(row=0, column=0, columnspan=10, sticky="nsew", padx=25, pady=30)
        for i in range(10):
            self.shell_frame.columnconfigure(i, weight=1)
        self.shell_frame.bind_all("<Button-4>", lambda e: self.shell_frame._parent_canvas.yview("scroll", -1, "units"))
        self.shell_frame.bind_all("<Button-5>", lambda e: self.shell_frame._parent_canvas.yview("scroll", 1, "units"))

        # Create the shell content.
        self.valid_commands = {
            "display": self.compute,
            "compute": self.compute,
            "clear": self.clear,
            "update_of": self.update_of
        }
        self.widgets = []
        self.images = []
        self.current_row = 0

        # The update equations.
        self.updates = {
            "N_prime": (10, 0, r"N^{'}_k = \sum_{n=\mathbb{N}^{'}} \hat{r}_{nk}"),
            "x_prime": (10, 10, r"\bar{x}^{'}_k = \frac{1}{N^{'}_k}\sum_{n=\mathbb{N}^{'}} \hat{r}_{nk} x_n"),
            "S_prime": (10, 10, r"S^{'}_k = \frac{1}{N^{'}_k}\sum_{n=\mathbb{N}^{'}} \hat{r}_{nk} (x_n - \bar{x}^{'}_k)(x_n - \bar{x}^{'}_k)^\top"),
            "W_bar": (0, 10, r"\bar{W}_k^{-1} = W_k^{-1} + N^{'}_k S^{'}_k + \frac{\beta_k N^{'}_k}{\beta_k + N^{'}_k}(\bar{x}^{'}_k - m_k)(\bar{x}^{'}_k - m_k)^\top"),
            "m_bar": (0, 35, r"\bar{m}_k = \frac{\beta_k m_k + N^{'}_k \bar{x}^{'}_k}{\beta_k + N^{'}_k}"),
            "v_bar": (10, 0, r"\bar{v}_k = v_k + N^{'}_k"),
            "beta_bar": (10, 0, r"\bar{\beta}_k = \beta_k + N^{'}_k"),
            "d_bar": (10, 0, r"\bar{d}_k = d_k + N^{'}_k"),
            "r_bar": (-30, 50, r"\bar{r}_{nk} = \frac{\rho_{nk}}{\sum_{k=1}^K \rho_{nk}} \quad where: \quad \rho_{nk} = \mathbb{E}_{Q(\boldsymbol{D})}[\ln \boldsymbol{D}_k] - \frac{K}{2}\ln 2\pi + \frac{1}{2}\mathbb{E}_{Q(\boldsymbol{\Lambda}_k)}[\ln \mid \boldsymbol{\Lambda}_k \mid] - \frac{1}{2}\mathbb{E}_{Q(\boldsymbol{\mu}_k, \boldsymbol{\Lambda}_k)}[(x_n - \boldsymbol{\mu}_k)^\top \boldsymbol{\Lambda}_k(x_n - \boldsymbol{\mu}_k)]"),
            "N_second": (10, 0, r"N^{''}_k = \sum_{n=\mathbb{N}^{''}} \hat{r}_{nk}"),
            "x_second": (10, 10, r"\bar{x}^{''}_k = \frac{1}{N^{''}_k}\sum_{n=\mathbb{N}^{''}} \hat{r}_{nk} x_n"),
            "S_second": (10, 10, r"S^{''}_k = \frac{1}{N^{''}_k}\sum_{n=\mathbb{N}^{''}} \hat{r}_{nk} (x_n - \bar{x}^{''}_k)(x_n - \bar{x}^{''}_k)^\top"),
            "W_hat": (0, 10, r"\hat{W}_k^{-1} = \bar{W}_k^{-1} + N^{''}_k S^{''}_k + \frac{\bar{\beta}_k N^{''}_k}{\bar{\beta}_k + N^{''}_k}(\bar{x}^{''}_k - \bar{m}_k)(\bar{x}^{''}_k - \bar{m}_k)^\top"),
            "m_hat": (0, 25, r"\hat{m}_k = \frac{\bar{\beta}_k \bar{m}_k + N^{''}_k \bar{x}^{''}_k}{\bar{\beta}_k + N^{''}_k}"),
            "v_hat": (10, 0, r"\hat{v}_k = \bar{v}_k + N^{''}_k"),
            "beta_hat": (10, 0, r"\hat{\beta}_k = \bar{\beta}_k + N^{''}_k"),
            "d_hat": (10, 0, r"\hat{d}_k = \bar{d}_k + N^{''}_k"),
            "r_hat": (-30, 50, r"\hat{r}_{nk} = \frac{\rho_{nk}}{\sum_{k=1}^K \rho_{nk}} \quad where: \quad \rho_{nk} = \mathbb{E}_{Q(\boldsymbol{D})}[\ln \boldsymbol{D}_k] - \frac{K}{2}\ln 2\pi + \frac{1}{2}\mathbb{E}_{Q(\boldsymbol{\Lambda}_k)}[\ln \mid \boldsymbol{\Lambda}_k \mid] - \frac{1}{2}\mathbb{E}_{Q(\boldsymbol{\mu}_k, \boldsymbol{\Lambda}_k)}[(x_n - \boldsymbol{\mu}_k)^\top \boldsymbol{\Lambda}_k(x_n - \boldsymbol{\mu}_k)]")
        }

        # Create the entry.
        fg_color = root._apply_appearance_mode(self.cget("fg_color"))
        self.entry_frame = tk.Frame(self, borderwidth=5, relief=tk.GROOVE, bg=fg_color)
        self.entry_frame.rowconfigure(0, weight=1)
        self.entry_frame.columnconfigure(0, weight=1)
        self.entry_frame.grid(row=1, column=1, columnspan=8, sticky="nsew", pady=(0, 30))

        self.command = tk.StringVar()
        self.command.set("Enter a command...")
        self.entry = ctk.CTkEntry(
            master=self.entry_frame, font=font, border_width=0, corner_radius=0, fg_color=fg_color,
            textvariable=self.command
        )
        self.entry.bind("<FocusIn>", lambda x: self.entry.delete(0, "end"))
        self.entry.bind('<Return>', self.execute_command)
        self.entry.grid(row=0, column=0, sticky="nsew", padx=15)

        # History variables.
        self.last_commands = []
        self.history_index = None
        self.entry.bind("<Up>", self.up_key_pressed)
        self.entry.bind("<Down>", self.down_key_pressed)

    def up_key_pressed(self, _):
        if len(self.last_commands) == 0:
            return
        if self.history_index is None:
            self.history_index = len(self.last_commands) - 1
        elif self.history_index > 0:
            self.history_index -= 1
        self.update_command_using_history()

    def down_key_pressed(self, _):
        if len(self.last_commands) == 0 or self.history_index is None:
            return
        if self.history_index < len(self.last_commands) - 1:
            self.history_index += 1
        self.update_command_using_history()

    def update_command_using_history(self):
        new_command = self.last_commands[self.history_index]
        self.command.set(new_command)
        self.entry.icursor(len(new_command))

    def update_content(self, prev_gm, next_gm):
        self.prev_gm = prev_gm
        self.next_gm = next_gm
        self.clear("clear all")
        self.command.set("Enter a command...")

    def execute_command(self, _):

        # Reset the index history.
        self.history_index = None

        # If no command was provided, return.
        cmd = self.command.get()
        tokens = cmd.split(" ")
        if len(tokens) == 1 and tokens[0] == "":
            return

        # Display the executed command.
        self.display_command(f"> {cmd}")
        self.last_commands.append(cmd)
        self.command.set("")

        # Check that the command is valid.
        if tokens[0] not in self.valid_commands.keys():
            valid_commands = ", ".join([f"'{cmd}'" for cmd in list(self.valid_commands.keys())[:-1]])
            valid_commands = f"{valid_commands} and '{list(self.valid_commands.keys())[-1]}'"
            text = f"Invalid command '{tokens[0]}'. The valid commands are: {valid_commands}."
            self.display_error_message(text)
            return

        # Execute the command.
        self.valid_commands[tokens[0]](cmd)

        # Scroll down after executing a command.
        self.shell_frame.after(10, self.shell_frame._parent_canvas.yview_moveto, 1.0)

    def display_command(self, cmd):
        cmd_label = ctk.CTkLabel(self.shell_frame, text=cmd, font=self.font)
        cmd_label.grid(
            row=self.current_row, column=0, columnspan=10, sticky="w", padx=self.padding, pady=self.padding
        )
        self.add_widget(cmd_label)

    def display_error_message(self, text):
        label = ctk.CTkLabel(self.shell_frame, text=text, font=self.font, text_color=self.error_color)
        label.grid(
            row=self.current_row, column=0, columnspan=10, sticky="w", padx=self.padding, pady=self.padding
        )
        self.add_widget(label)

    def display_warning_message(self, text):
        label = ctk.CTkLabel(self.shell_frame, text=text, font=self.font, text_color=self.warning_color)
        label.grid(
            row=self.current_row, column=0, columnspan=10, sticky="w", padx=self.padding, pady=self.padding
        )
        self.add_widget(label)

    def display_tensor(self, x, tensor_name="result"):

        # Display 0d tensor, i.e., scalar.
        if isinstance(x, torch.Tensor) and len(x.shape) == 0:
            x = x.item()
        if isinstance(x, float) or isinstance(x, int):
            image = MatPlotLib.draw_equation(
                f"{tensor_name} = {x:0.3f}",
                self.to_rgb(self.bg_color), self.to_rgb(self.text_color), 50, 10
            )
            self.display_image(image)
            self.entry.focus_force()
            return

        # Convert list into tensor.
        if isinstance(x, list):
            if len(x) == 0 or (len(x) > 0 and not isinstance(x[0], torch.Tensor)):
                x = torch.tensor(x)

        # Check that the size of the tensor is correct.
        len_shape = 1 + len(x[0].shape) if isinstance(x, list) else len(x.shape)
        if 3 < len_shape < 1:
            msg = f"Invalid shape, the tensor '{tensor_name}' should have between one and three dimensions."
            self.display_error_message(msg)

        # Format the tensor as a list of matrices.
        display_index = True if isinstance(x, list) else False
        if not isinstance(x, list):
            x = [x]
        if len(x[0].shape) <= 1:
            for i in range(len(x)):
                x[i] = x[i].unsqueeze(dim=0)

        # Display the list of matrices.
        for index, matrix in enumerate(x):
            equation = f"{tensor_name}[{index}] = " if display_index is True else f"{tensor_name} = "
            equation += self.to_latex_format(matrix)
            image = MatPlotLib.draw_equation(equation, self.to_rgb(self.bg_color), self.to_rgb(self.text_color), 50, 10)
            self.display_image(image)
        self.entry.focus_force()

    @staticmethod
    def to_latex_format(matrix):
        """
        Format a matrix using LaTeX syntax
        :param matrix: the matrix to format
        :return: the formatted matrix
        """
        n_rows, n_cols = matrix.shape
        if n_rows == 0 or n_cols == 0:
            return "[]"
        rows = []
        for row_id in range(n_rows):
            row = " & ".join([f"{matrix[row_id, col_id].item():0.3f}" for col_id in range(n_cols)])
            rows.append(row)
        return r"\begin{bmatrix} " + r"\\".join(rows) + r" \end{bmatrix}"

    def display_image(self, image, pad_left=0):
        image = FigureCanvasTkAgg(image, master=self.shell_frame)
        self.images.append(image)
        widget = image.get_tk_widget()
        widget.grid(
            row=self.current_row, column=0, columnspan=10, sticky="w",
            padx=(self.padding + pad_left, self.padding), pady=self.padding
        )
        self.add_widget(widget)

    def add_widget(self, widget):
        if len(self.widgets) != 0:
            self.widgets[-1].grid(pady=(self.padding, 0))
        self.widgets.append(widget)
        self.current_row += 1

    def to_rgb(self, color):
        return tuple(((c // 256) / 255 for c in self.root.winfo_rgb(color)))

    def update_of(self, cmd):

        # Split the command into tokens.
        tokens = cmd.split(" ")

        # Check that the command parameters are valid.
        if len(tokens) != 2 or (tokens[1] not in self.updates.keys() and tokens[1] != "all"):
            self.display_error_message(f"Invalid command parameters '{cmd}'.")
            self.display_error_message("Usage: 'update_of name', where 'name' is:")
            valid_updates = ", ".join([f"'{cmd}'" for cmd in list(self.updates.keys())[:-1]])
            valid_updates = f"\t {valid_updates} or '{list(self.updates.keys())[-1]}'."
            self.display_error_message(valid_updates)
            return

        # focus on only the update of interest.
        if tokens[1] != "all":
            pad_left, pad_x, equation = self.updates[tokens[1]]
            updates = {tokens[1]: (pad_left, pad_x, equation)}
        else:
            updates = self.updates

        # Display the update(s).
        for pad_left, pad_x, equation in updates.values():
            image = MatPlotLib.draw_equation(equation, self.to_rgb(self.bg_color), self.to_rgb(self.text_color), pad_x)
            self.display_image(image, pad_left=pad_left)
        self.entry.focus_force()

    def compute(self, cmd):

        # The variables and functions accessible from the shell.
        variables = {
            "x0": self.debugger.xs[self.prev_gm],
            "W0": self.debugger.gms[self.prev_gm].W,
            "m0": self.debugger.gms[self.prev_gm].m,
            "v0": self.debugger.gms[self.prev_gm].v,
            "beta0": self.debugger.gms[self.prev_gm].β,
            "d0": self.debugger.gms[self.prev_gm].d,
            "W0_bar": self.debugger.gms[self.prev_gm].W_bar,
            "m0_bar": self.debugger.gms[self.prev_gm].m_bar,
            "v0_bar": self.debugger.gms[self.prev_gm].v_bar,
            "beta0_bar": self.debugger.gms[self.prev_gm].β_bar,
            "d0_bar": self.debugger.gms[self.prev_gm].d_bar,
            "x0_bar": self.debugger.gms[self.prev_gm].x_bar,
            "r0_bar": self.debugger.gms[self.prev_gm].r_bar,
            "W0_hat": self.debugger.gms[self.prev_gm].W_hat,
            "m0_hat": self.debugger.gms[self.prev_gm].m_hat,
            "v0_hat": self.debugger.gms[self.prev_gm].v_hat,
            "beta0_hat": self.debugger.gms[self.prev_gm].β_hat,
            "d0_hat": self.debugger.gms[self.prev_gm].d_hat,
            "r0_hat": self.debugger.gms[self.prev_gm].r_hat,
            "N0_prime": self.debugger.gms[self.prev_gm].N_prime,
            "x0_prime": self.debugger.gms[self.prev_gm].x_prime,
            "S0_prime": self.debugger.gms[self.prev_gm].S_prime,
            "N0_second": self.debugger.gms[self.prev_gm].N_second,
            "x0_second": self.debugger.gms[self.prev_gm].x_second,
            "S0_second": self.debugger.gms[self.prev_gm].S_second,
            "gm0": self.debugger.gms[self.prev_gm],
            "x1": self.debugger.xs[self.next_gm],
            "W1": self.debugger.gms[self.next_gm].W,
            "m1": self.debugger.gms[self.next_gm].m,
            "v1": self.debugger.gms[self.next_gm].v,
            "beta1": self.debugger.gms[self.next_gm].β,
            "d1": self.debugger.gms[self.next_gm].d,
            "W1_bar": self.debugger.gms[self.next_gm].W_bar,
            "m1_bar": self.debugger.gms[self.next_gm].m_bar,
            "v1_bar": self.debugger.gms[self.next_gm].v_bar,
            "beta1_bar": self.debugger.gms[self.next_gm].β_bar,
            "d1_bar": self.debugger.gms[self.next_gm].d_bar,
            "x1_bar": self.debugger.gms[self.next_gm].x_bar,
            "r1_bar": self.debugger.gms[self.next_gm].r_bar,
            "W1_hat": self.debugger.gms[self.next_gm].W_hat,
            "m1_hat": self.debugger.gms[self.next_gm].m_hat,
            "v1_hat": self.debugger.gms[self.next_gm].v_hat,
            "beta1_hat": self.debugger.gms[self.next_gm].β_hat,
            "d1_hat": self.debugger.gms[self.next_gm].d_hat,
            "r1_hat": self.debugger.gms[self.next_gm].r_hat,
            "N1_prime": self.debugger.gms[self.next_gm].N_prime,
            "x1_prime": self.debugger.gms[self.next_gm].x_prime,
            "S1_prime": self.debugger.gms[self.next_gm].S_prime,
            "N1_second": self.debugger.gms[self.next_gm].N_second,
            "x1_second": self.debugger.gms[self.next_gm].x_second,
            "S1_second": self.debugger.gms[self.next_gm].S_second,
            "gm1": self.debugger.gms[self.next_gm]
        }
        functions = {
            "near": self.near
        }

        # Execute the python command provided by the user.
        python_code = cmd[len("compute "):]
        try:
            x = eval(python_code, variables | functions)
            self.display_tensor(x)
        except Exception as e:
            self.display_error_message(e)

    @staticmethod
    def near(rs, xs, x, n):
        """
        Return the responsibilities corresponding to the 'n' nearest points
        :param rs: the responsibilities
        :param xs: the data points
        :param x: the data point of references
        :param n: the number of responsibilities to return
        :return: the responsibilities of the nearest points
        """

        # Retrieve the indices of the nearest points.
        distances = [KMeans.distance(x, xs_i) for xs_i in xs]
        shortest_distances = sorted(distances)[:n]
        indices = []
        for i, distance in enumerate(distances):
            if distance in shortest_distances:
                indices.append(i)
        indices = indices[:n]

        # Return the responsibilities of the nearest points.
        return rs[indices]

    def clear(self, cmd):

        # Split the command into tokens.
        tokens = cmd.split(" ")

        # Clear all commands, including clear itself.
        if len(tokens) == 2 and tokens[1] == "all":
            for widget in self.widgets:
                widget.grid_forget()
            self.widgets.clear()
            self.images.clear()
            self.last_commands.clear()
            self.history_index = None
            self.current_row = 0
            return

        # Clear all commands but clear.
        for widget in self.widgets[:-1]:
            widget.grid_forget()
        self.widgets = [self.widgets[-1]]
        self.widgets[-1].grid(row=0, column=0, columnspan=10, sticky="w")
        self.images.clear()
        self.last_commands = [self.last_commands[:-1]]
        self.history_index = None
        self.current_row = 1

        # Display warning, if command was misused.
        if len(tokens) != 1:
            text = "Usage: 'clear [all]'."
            self.display_warning_message(text)
