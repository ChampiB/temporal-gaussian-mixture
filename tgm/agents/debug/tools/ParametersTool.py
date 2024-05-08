import customtkinter as ctk

from tgm.agents.debug.widgets.Shell import Shell


class ParametersTool(ctk.CTkFrame):

    def __init__(self, parent, debugger, font=("Helvetica", 30, "normal")):

        super().__init__(parent)

        # Retrieve the selected entry.
        selected = debugger.nav_tree.selected()

        # Configure the column and row weights.
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Ask the user to select an entry in the navigation tree.
        self.selection_label = ctk.CTkLabel(self, text="Please select an entry in the navigation tree.", font=font)
        if selected is None:
            self.selection_label.grid(row=0, column=0, sticky="nsew")

        # Create the shell.
        self.shell = Shell(parent, self, debugger)
        if selected is not None:
            self.update_content(debugger.checkpoints, selected)

    def update_content(self, checkpoints, tags):

        # Remove the initial text, and display the images instead.
        if self.selection_label.winfo_viewable():
            self.selection_label.grid_forget()
        if not self.shell.winfo_viewable():
            self.shell.grid(row=0, column=0, sticky="nsew")

        # Update the shell's content.
        self.shell.update_content(tags[0], tags[-1])
