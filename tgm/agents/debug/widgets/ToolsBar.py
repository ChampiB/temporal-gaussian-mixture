import customtkinter as ctk
from functools import partial


class ToolsBar:

    def __init__(self, parent, update_tool, padding=10, font=("Helvetica", 30, "normal"), size=65):

        distributions_button = ctk.CTkButton(
            parent, text="D", width=size, height=size, corner_radius=0, font=font,
            command=partial(update_tool, name="distributions")
        )
        distributions_button.grid(row=0, column=0, padx=padding, pady=(padding, 0))

        parameters_button = ctk.CTkButton(
            parent, text="θ", width=size, height=size, corner_radius=0, font=font,
            command=partial(update_tool, name="parameters")
        )
        parameters_button.grid(row=1, column=0, padx=padding, pady=(padding, 0))

        vfe_button = ctk.CTkButton(
            parent, text="F", width=size, height=size, corner_radius=0, font=font,
            command=partial(update_tool, name="vfe")
        )
        vfe_button.grid(row=2, column=0, padx=padding, pady=(padding, 0))

        components_button = ctk.CTkButton(
            parent, text="C", width=size, height=size, corner_radius=0, font=font,
            command=partial(update_tool, name="fixed_components")
        )
        components_button.grid(row=3, column=0, padx=padding, pady=(padding, 0))
