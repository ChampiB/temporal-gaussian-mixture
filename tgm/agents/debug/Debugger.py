from tgm.agents.debug.DebuggerGUI import DebuggerGUI


class Debugger:

    def __init__(self, model, debug):

        # Save the model and whether debug information should be stored.
        self.model = model
        self.debug = debug

        # Store the current iid and a dictionary keeping track of the automatic indices for each prefix.
        # Example:
        # - iid = "fit_1.vi_step_1.update_Z"
        # - prefixes = {"fit", "vi_step", "update_Z"}
        # - auto_indices = {"fit": 1, "vi_step": 1, "update_Z": -1}
        self.current_iid = ""
        self.auto_indices = {}

        # Store the last checkpoints corresponding to each prefix.
        # Example:
        # - prefixes = {"fit", "vi_step", "update_Z"}
        # - last_checkpoints = {"fit": [0], "vi_step": [1], "update_Z": [1]}
        self.last_checkpoints = {}

        # Store all the checkpoints.
        # Each checkpoint corresponds to an instantaneous screenshot of the relevant data such as the model and dataset.
        self.checkpoints = []

        # Store the data that must be displayed in the navigation tree.
        # Example:
        # - data = {"fit_1": [0, 3], "fit_1.vi_step_1": [0, 3], "fit_1.vi_step_1.update_Z": [0, 1], ...}
        self.data = []

    def run(self):

        # Check that debugging is required.
        if self.debug is False:
            return

        # Create and run the debugger's graphical user interface.
        gui = DebuggerGUI(self.data, self.checkpoints)
        gui.run()

    def before(self, prefix, auto_index=False):

        # Check that debugging is required.
        if self.debug is False:
            return

        # Initialize the auto-index corresponding to the prefix.
        if prefix not in self.auto_indices.keys():
            self.auto_indices[prefix] = 1 if auto_index is True else -1
        auto_id = self.auto_indices[prefix]

        # Add a new level in the iid.
        level = prefix
        if auto_id != -1:
            level += f"_{auto_id}"
            self.auto_indices[prefix] += 1
        self.add_iid_level(level)

        # Reset all auto-indices of (sub) levels.
        for auto_prefix, auto_index in self.auto_indices.items():
            if auto_prefix not in self.current_iid and auto_index != -1:
                self.auto_indices[auto_prefix] = 1

        # Add checkpoint and update last checkpoints of prefix.
        self.add_checkpoint(prefix)

    def add_iid_level(self, level):
        if len(self.current_iid) != 0:
            self.current_iid += "."
        self.current_iid += level

    def middle(self, prefix):

        # Add checkpoint and update last checkpoints of prefix.
        self.add_checkpoint(prefix)

    def after(self, prefix):

        # Check that debugging is required.
        if self.debug is False:
            return

        # Check that the prefix is in the last level of the iid.
        levels = self.current_iid.split(".")
        if len(levels) == 0 or prefix not in levels[-1]:
            print(f"[Debugger][Warning] Prefix {prefix} not in last level of iid {self.current_iid}.")
            return

        # Add checkpoint and update last checkpoints of prefix.
        self.add_checkpoint(prefix)

        # Add an entry in the tree view data.
        insert_id = [i for i, (iid, _) in enumerate(self.data) if self.current_iid in iid]
        insert_id = len(self.data) if len(insert_id) == 0 else insert_id[0]
        self.data.insert(insert_id, (self.current_iid, self.last_checkpoints[prefix]))
        self.last_checkpoints[prefix] = []

        # Remove the last level of the iid.
        self.remove_iid_level()

    def remove_iid_level(self):
        levels = self.current_iid.split(".")
        self.current_iid = ".".join(levels[:-1])

    def add_checkpoint(self, prefix):

        # Check whether a next checkpoint needs to be created.
        if len(self.checkpoints) == 0 or self.model.diff(self.checkpoints[-1]) is not None:
            self.checkpoints.append(self.model.clone())

        # Update the list of last checkpoints corresponding to the prefix.
        checkpoints_id = len(self.checkpoints) - 1
        if prefix in self.last_checkpoints.keys():
            self.last_checkpoints[prefix].append(checkpoints_id)
        else:
            self.last_checkpoints[prefix] = [checkpoints_id]
