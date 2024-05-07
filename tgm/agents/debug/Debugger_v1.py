from tkinter import VERTICAL
import tkinter.ttk
import customtkinter as ctk
from screeninfo import get_monitors
import matplotlib as mpl

from tgm.agents.debug.tools.CostTool import CostTool
from tgm.agents.debug.tools.DistributionsTool import DistributionsTool
from tgm.agents.debug.tools.FixedComponentsTool import FixedComponentsTool
from tgm.agents.debug.tools.InitializationTool import InitializationTool
from tgm.agents.debug.tools.ParametersTool import ParametersTool
from tgm.agents.debug.widgets.ToolsBar import ToolsBar
from tgm.agents.debug.widgets.TreeView import TreeView


class Debugger:

    def __init__(self, model, env):

        self.model = model
        self.env = env

        # Indices keeping track of the current fit, vi-step, and vi-update.
        self.current_fit = 0
        self.current_step = 0
        self.current_update = 0

        # Indices keeping track of the previous Gaussian mixture for fit, vi-step and vi-update.
        self.gm_prev_fit = 0
        self.gm_prev_step = 0
        self.gm_prev_update = 0

        # List of Gaussian mixtures, and the data to display in the tree.
        self.gms = []
        self.xs = []
        self.data = []
        self.init_params = []

        # Create a variable that will contain the tkinter window and the navigation tree.
        self.root = None
        self.nav_tree = None

        # Set the mode and theme, as well as matplotlib backend.
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        mpl.use("TkAgg")

        # The available tools of the GUI.
        self.tools = {
            "distributions": DistributionsTool,
            "parameters": ParametersTool,
            "vfe": CostTool,
            "fixed_components": FixedComponentsTool,
            "initialization": InitializationTool
        }
        self.tool_instances = {
            "distributions": None,
            "parameters": None,
            "vfe": None,
            "fixed_components": None,
            "initialization": None
        }
        self.current_tool_name = "distributions"
        self.current_tool = None

    def run(self):

        # Collect the data.
        self.model.train(self.env, self)

        # Create the GUI.
        self.root = ctk.CTk()
        self.root.title("Gaussian Mixture Debugger")
        monitor = get_monitors()[0]
        self.root.geometry(f"{monitor.width}x{monitor.width}")
        self.root.columnconfigure(3, weight=3)
        self.root.rowconfigure(0, weight=1)

        # Create the tools bar.
        tools = ctk.CTkFrame(self.root)
        ToolsBar(tools, self.update_tool)
        tools.grid(row=0, column=0, sticky="nsew")

        # Create a separator between the tools and navigation bars.
        separator = tkinter.ttk.Separator(self.root, orient=VERTICAL)
        separator.grid(row=0, column=1, sticky="ns")

        # Create the navigation bar.
        navigation = ctk.CTkFrame(self.root)
        self.nav_tree = TreeView(self.root, navigation, self.update_current_tool, self.data)
        navigation.grid(row=0, column=2, sticky="ns")
        navigation.rowconfigure(0, weight=1)

        # Create the tool frame.
        self.update_tool(self.current_tool_name)

        # Launch the application.
        self.root.mainloop()

    def update_tool(self, name):

        # Create the new tool, if it does not exist.
        if self.tool_instances[name] is None:
            self.tool_instances[name] = self.tools[name](self.root, self)

        # Update the tool displayed.
        if self.current_tool is not None:
            self.current_tool.grid_forget()
        self.current_tool = self.tool_instances[name]
        self.current_tool.grid(row=0, column=3, sticky="nsew")

        # Update the current tool name.
        self.current_tool_name = name

    def update_current_tool(self, tags):
        prev_gm, next_gm, fit_id = tags
        self.current_tool.update_content(self.gms, self.xs, self.init_params, int(prev_gm), int(next_gm), int(fit_id))

    def after_fit(self):
        self.current_fit += 1
        self.current_step = 0
        data = (f"fit_{self.current_fit}", (self.gm_prev_fit, self.gm_prev_update, self.current_fit - 1))
        self.data.insert(self.current_fit - 1, data)
        self.gm_prev_fit = self.gm_prev_update

        print(f"[Debugger] After fit: {self.current_fit}")

    def after_initialize(self):

        if len(self.gms) != 0:
            return

        # Keep track of the new Gaussian mixture, and the corresponding dataset.
        self.gms.append(self.model.gm.clone())
        self.xs.append(self.model.gm_data.get())

        print(f"[Debugger] After initialize.")

    def after_update(self):

        # Keep track of the new Gaussian mixture, and the corresponding dataset.
        self.gms.append(self.model.gm.clone())
        self.xs.append(self.model.gm_data.get())

        # Update the vi-step index, if needed.
        if self.current_update % 3 == 0:
            self.current_step += 1

        # Add the vi-step to the data, if needed.
        step_iid = f"fit_{self.current_fit + 1}.vi_step_{self.current_step}"
        if self.current_update == 0:
            self.data.append((step_iid, (self.gm_prev_step, self.gm_prev_step + 3, self.current_fit - 1)))
            self.gm_prev_step += 3

        # Add the vi-update to the list.
        update_text = ["update_Z", "update_D", "update_μ_and_Λ"][self.current_update % 3]
        update_iid = f"{step_iid}.{update_text}"
        self.data.append((update_iid, (self.gm_prev_update, self.gm_prev_update + 1, self.current_fit - 1)))
        self.current_update = (self.current_update + 1) % 3
        self.gm_prev_update += 1

    def update_last_gm(self):
        self.gms[-1] = self.model.gm.clone()

    def initialization_fixed(self, v, d, β, m, W, new_entry=True):
        if new_entry is True:
            self.init_params.append({})
        self.init_params[-1]["fixed"] = (v, d, β, m, W)

    def initialization_flexible(self, v, d, β, m, W, new_entry=False):
        if new_entry is True:
            self.init_params.append({})
        self.init_params[-1]["flexible"] = (v, d, β, m, W)

    def initialization_combined(self, v, d, β, m, W, new_entry=False):
        if new_entry is True:
            self.init_params.append({})
        self.init_params[-1]["combined"] = (v, d, β, m, W)
