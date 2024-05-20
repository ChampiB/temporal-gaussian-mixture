import customtkinter as ctk
from functools import partial
from os.path import join
from os import environ as env_vars
from PIL import Image


class ToolsBar:

    def __init__(self, parent, update_tool, padding=10, font=("Helvetica", 30, "normal"), size=65):

        self.padding = padding

        self.distribution_icon = ctk.CTkImage(
            Image.open(join(env_vars["DATA_DIRECTORY"], "icons", "distribution.png")), size=(50, 50)
        )
        self.distributions_button = ctk.CTkButton(
            parent, text="", image=self.distribution_icon, width=size, height=size, corner_radius=0, font=font,
            command=partial(update_tool, name="distributions")
        )

        self.parameter_image = ctk.CTkImage(
            Image.open(join(env_vars["DATA_DIRECTORY"], "icons", "parameter.png")), size=(50, 50)
        )
        self.parameters_button = ctk.CTkButton(
            parent, text="", image=self.parameter_image, width=size, height=size, corner_radius=0, font=font,
            command=partial(update_tool, name="parameters")
        )

        self.vfe_button = ctk.CTkButton(
            parent, text="F", width=size, height=size, corner_radius=0, font=font,
            command=partial(update_tool, name="vfe")
        )

        self.responsibilities_button = ctk.CTkButton(
            parent, text="R", width=size, height=size, corner_radius=0, font=font,
            command=partial(update_tool, name="responsibilities")
        )

        self.components_button = ctk.CTkButton(
            parent, text="C", width=size, height=size, corner_radius=0, font=font,
            command=partial(update_tool, name="fixed_components")
        )

        self.initialization_button = ctk.CTkButton(
            parent, text="I", width=size, height=size, corner_radius=0, font=font,
            command=partial(update_tool, name="initialization")
        )

        self.data_image = ctk.CTkImage(
            Image.open(join(env_vars["DATA_DIRECTORY"], "icons", "database.png")), size=(50, 50)
        )
        self.data_button = ctk.CTkButton(
            parent, text="", image=self.data_image, width=size, height=size, corner_radius=0, font=font,
            command=partial(update_tool, name="data")
        )

        self.ws_button = ctk.CTkButton(
            parent, text="W", width=size, height=size, corner_radius=0, font=font,
            command=partial(update_tool, name="ws")
        )

        self.bs_button = ctk.CTkButton(
            parent, text="B", width=size, height=size, corner_radius=0, font=font,
            command=partial(update_tool, name="bs")
        )

        self.buttons = {
            "distributions": self.distributions_button,
            "parameters": self.parameters_button,
            "data": self.data_button,
            "vfe": self.vfe_button,
            "fixed_components": self.components_button,
            "initialization": self.initialization_button,
            "responsibilities": self.responsibilities_button,
            "ws": self.ws_button,
            "bs": self.bs_button
        }

    def update_displayed_tools(self, tool_names):
        i = 0
        for tool_name, button in self.buttons.items():
            button.grid_forget()
            if tool_name in tool_names:
                button.grid(row=0, column=i, padx=self.padding, pady=self.padding)
                i += 1
