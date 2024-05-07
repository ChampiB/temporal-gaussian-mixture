import customtkinter as ctk
from screeninfo import get_monitors
import matplotlib as mpl

from tgm.agents.debug.widgets.TreeView import TreeView


class DebuggerGUI:

    def __init__(self, data, checkpoints):

        # Save the tree view data and associated checkpoints.
        self.data = data
        self.checkpoints = checkpoints

        # Set the mode and theme, as well as matplotlib backend.
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        mpl.use("TkAgg")

        # Create the GUI.
        self.root = ctk.CTk()
        self.root.title("Gaussian Mixture Debugger")
        monitor = get_monitors()[0]
        self.root.geometry(f"{monitor.width}x{monitor.width}")
        self.root.columnconfigure(3, weight=3)
        self.root.rowconfigure(0, weight=1)

        # Create the navigation bar.
        navigation = ctk.CTkFrame(self.root)
        self.nav_tree = TreeView(self.root, navigation, print, self.data)
        navigation.grid(row=0, column=2, sticky="ns")
        navigation.rowconfigure(0, weight=1)

    def run(self):

        # Launch the application.
        self.root.mainloop()
