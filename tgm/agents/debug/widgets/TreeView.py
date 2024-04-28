from tkinter import ttk
import customtkinter as ctk


class TreeView:

    def __init__(self, root, parent, on_select_function, data, padding=10):

        # Treeview Customisation.
        bg_color = root._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
        text_color = root._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"])
        selected_color = root._apply_appearance_mode(ctk.ThemeManager.theme["CTkButton"]["fg_color"])

        style = ttk.Style()
        style.theme_use("default")
        config = {
            "background": bg_color,
            "foreground": text_color,
            "fieldbackground": bg_color,
            "borderwidth": 0,
            "font": ("Helvetica", 20, "normal"),
            "rowheight": 40,
        }
        style.configure("Treeview", **config)
        style.map("Treeview", background=[("selected", bg_color)], foreground=[("selected", selected_color)])
        parent.bind("<<TreeviewSelect>>", lambda event: parent.focus_set())

        # Store the function to call when an element is clicked.
        self.on_select_function = on_select_function

        # Treeview widget data
        self.treeview = ttk.Treeview(parent, show="tree", selectmode="browse")
        self.treeview.column("#0", width=300)
        self.insert(data)
        self.treeview.bind("<<TreeviewSelect>>", self.on_select)
        self.treeview.grid(row=0, column=0, padx=padding, pady=(padding, 0), sticky="nsew")

        scrollbar = ctk.CTkScrollbar(parent, command=self.treeview.yview)
        scrollbar.grid(row=0, column=1, sticky="nsew")

        self.treeview.configure(yscrollcommand=scrollbar.set)

    def on_select(self, _):
        item = self.treeview.selection()[0]
        tags = self.treeview.item(item, "tags")
        self.on_select_function(tags)

    def insert(self, data):
        for iid, tags in data:
            indices = iid.split(".")
            parent = "" if len(indices) == 1 else ".".join(indices[:-1])
            text = indices[-1]
            self.treeview.insert(parent, "end", iid, text=text, tags=tags)

    def selected(self):
        items = self.treeview.selection()
        if len(items) == 0:
            return None
        return self.treeview.item(items[0], "tags")
