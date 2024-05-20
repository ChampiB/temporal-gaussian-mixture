from functools import partial

import customtkinter as ctk
from screeninfo import get_monitors
import matplotlib as mpl

from tgm.agents.debug.tools.BsTool import BsTool
from tgm.agents.debug.tools.CostTool import CostTool
from tgm.agents.debug.tools.DatasetTool import DatasetTool
from tgm.agents.debug.tools.DistributionsTool import DistributionsTool
from tgm.agents.debug.tools.FixedComponentsTool import FixedComponentsTool
from tgm.agents.debug.tools.InitializationTool import InitializationTool
from tgm.agents.debug.tools.ParametersTool import ParametersTool
from tgm.agents.debug.tools.ResponsibilitiesTool import ResponsibilitiesTool
from tgm.agents.debug.tools.WsTool import WsTool
from tgm.agents.debug.widgets.ToolsBar import ToolsBar
from tgm.agents.debug.widgets.TreeView import TreeView


class DebuggerGUI:

    def __init__(self, data, checkpoints, action_names):

        # Save the tree view data, associated checkpoints, and environment's action names.
        self.data = data
        self.checkpoints = checkpoints
        self.action_names = action_names

        # Set the mode and theme, as well as matplotlib backend.
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        mpl.use("TkAgg")

        # Create the GUI.
        self.root = ctk.CTk()
        self.root.title("Gaussian Mixture Debugger")
        monitor = get_monitors()[0]
        self.root.geometry(f"{monitor.width}x{monitor.width}")
        self.root.columnconfigure(1, weight=3)
        self.root.rowconfigure(1, weight=1)

        # Create the navigation tree.
        navigation = ctk.CTkFrame(self.root)
        self.nav_tree = TreeView(self.root, navigation, self.update_current_tool, self.data)
        navigation.grid(row=0, column=0, rowspan=2, sticky="ns")
        navigation.rowconfigure(0, weight=1)

        # Create the tools bar.
        self.tools_frame = ctk.CTkFrame(self.root, corner_radius=0)
        self.tools_bar = ToolsBar(self.tools_frame, self.change_tool)

        # The available tools of the GUI.
        self.tools = {
            "distributions": DistributionsTool,
            "parameters": ParametersTool,
            "data": DatasetTool,
            "vfe": CostTool,
            "fixed_components": FixedComponentsTool,
            "initialization": InitializationTool,
            "responsibilities": ResponsibilitiesTool,
            "ws": WsTool,
            "bs": partial(BsTool, action_names=self.action_names)
        }
        self.tool_instances = {
            "distributions": None,
            "parameters": None,
            "data": None,
            "vfe": None,
            "fixed_components": None,
            "initialization": None,
            "responsibilities": None,
            "ws": None,
            "bs": None
        }
        self.current_tool_name = "distributions"
        self.current_tool = None

    def run(self):

        # Launch the application.
        self.root.mainloop()

    def change_tool(self, name):

        # Create the new tool, if it does not exist.
        if self.tool_instances[name] is None:
            self.tool_instances[name] = self.tools[name](self.root, self)

        # Update the tool displayed.
        if self.current_tool is not None:
            self.current_tool.grid_forget()

        self.current_tool = self.tool_instances[name]
        self.current_tool.grid(row=1, column=1, sticky="nsew")

        # Update the current tool being displayed.
        tags = [int(tag) for tag in self.nav_tree.selected()]
        self.current_tool.update_content(self.checkpoints, tags)

        # Update the current tool name.
        self.current_tool_name = name

    def update_current_tool(self, text, tags):

        # Display the tools bar.
        if self.current_tool is None:
            self.tools_frame.grid(row=0, column=1, sticky="nsew")

        # Update the tools in the tools bar.
        tools_to_display = {
            "tm_fit": ["bs"],
            "gm_fit": ["distributions", "parameters", "data", "vfe", "responsibilities", "ws"],
            "vi_step": ["distributions", "parameters", "data", "vfe", "responsibilities", "ws"],
            "update_Z": ["distributions", "parameters", "data", "vfe", "responsibilities", "ws"],
            "update_D": ["distributions", "parameters", "data", "vfe", "responsibilities", "ws"],
            "update_μ_and_Λ": ["distributions", "parameters", "data", "vfe", "responsibilities", "ws"],
            "update_fixed_components": ["fixed_components"],
            "prior_initialization": ["initialization"]
        }
        new_tools = []
        for tool_name, tools in tools_to_display.items():
            if tool_name in text:
                new_tools = tools
                self.tools_bar.update_displayed_tools(tools)
                break

        # Let the user know if the toolbar was not updated.
        if len(new_tools) == 0:
            print(f"[Warning] Tools bar was not updated for {text}.")
            return

        # Change the current tool.
        tool_name = self.current_tool_name if self.current_tool_name in new_tools else new_tools[0]
        self.change_tool(tool_name)

        # Update the current tool being displayed.
        tags = [int(tag) for tag in tags]
        self.current_tool.update_content(self.checkpoints, tags)
