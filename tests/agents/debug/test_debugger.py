import pytest
from unittest.mock import Mock

from tgm.agents.TemporalGaussianMixture import TemporalGaussianMixture
from tgm.agents.debug.Debugger import Debugger


class TestDebugger:

    @pytest.fixture()
    def model(self):
        tgm = Mock()

        # Set some fake return values for the clone and diff functions.
        tgm.clone.return_value = [1, 1]

        return tgm

    @pytest.fixture()
    def debugger(self, model):
        return Debugger(model, debug=True)

    def test_before_middle_after(self, debugger):

        # Check that the attributes are initialized properly.
        assert debugger.current_iid == ""
        assert debugger.auto_indices == {}
        assert debugger.last_checkpoints == {}
        assert debugger.checkpoints == []
        assert debugger.data == []

        # Check that the before command works properly.
        debugger.before("initialize")

        assert debugger.current_iid == "initialize"
        assert debugger.auto_indices["initialize"] == -1
        assert debugger.last_checkpoints["initialize"] == [0]
        assert len(debugger.checkpoints) == 1
        assert debugger.data == []

        # Check that the middle command works properly.
        debugger.middle("initialize")

        assert debugger.current_iid == "initialize"
        assert debugger.auto_indices["initialize"] == -1
        assert debugger.last_checkpoints["initialize"] == [0, 1]
        assert len(debugger.checkpoints) == 2
        assert debugger.data == []

        # Check that the middle command works properly.
        debugger.middle("initialize")

        assert debugger.current_iid == "initialize"
        assert debugger.auto_indices["initialize"] == -1
        assert debugger.last_checkpoints["initialize"] == [0, 1, 2]
        assert len(debugger.checkpoints) == 3
        assert debugger.data == []

        # Check that the after command works properly.
        debugger.after("initialize")

        assert debugger.current_iid == ""
        assert debugger.auto_indices["initialize"] == -1
        assert debugger.last_checkpoints["initialize"] == []
        assert len(debugger.checkpoints) == 4
        assert len(debugger.data) == 1
        assert debugger.data[0] == ("initialize", [0, 1, 2, 3])

        # Check that the before command works properly.
        debugger.before("fit", auto_index=True)

        assert debugger.current_iid == "fit_1"
        assert debugger.auto_indices["initialize"] == -1
        assert debugger.auto_indices["fit"] == 2
        assert debugger.last_checkpoints["initialize"] == []
        assert debugger.last_checkpoints["fit"] == [4]
        assert len(debugger.checkpoints) == 5
        assert len(debugger.data) == 1
        assert debugger.data[0] == ("initialize", [0, 1, 2, 3])

        # Check that the before command works properly.
        debugger.before("vi_step", auto_index=True)

        assert debugger.current_iid == "fit_1.vi_step_1"
        assert debugger.auto_indices["initialize"] == -1
        assert debugger.auto_indices["fit"] == 2
        assert debugger.auto_indices["vi_step"] == 2
        assert debugger.last_checkpoints["initialize"] == []
        assert debugger.last_checkpoints["fit"] == [4]
        assert debugger.last_checkpoints["vi_step"] == [5]
        assert len(debugger.checkpoints) == 6
        assert len(debugger.data) == 1
        assert debugger.data[0] == ("initialize", [0, 1, 2, 3])

        # Check that the before command works properly.
        debugger.before("update_Z")

        assert debugger.current_iid == "fit_1.vi_step_1.update_Z"
        assert debugger.auto_indices["initialize"] == -1
        assert debugger.auto_indices["fit"] == 2
        assert debugger.auto_indices["vi_step"] == 2
        assert debugger.auto_indices["update_Z"] == -1
        assert debugger.last_checkpoints["initialize"] == []
        assert debugger.last_checkpoints["fit"] == [4]
        assert debugger.last_checkpoints["vi_step"] == [5]
        assert debugger.last_checkpoints["update_Z"] == [6]
        assert len(debugger.checkpoints) == 7
        assert len(debugger.data) == 1
        assert debugger.data[0] == ("initialize", [0, 1, 2, 3])

        # Check that the after command works properly.
        debugger.after("update_Z")

        assert debugger.current_iid == "fit_1.vi_step_1"
        assert debugger.auto_indices["initialize"] == -1
        assert debugger.auto_indices["fit"] == 2
        assert debugger.auto_indices["vi_step"] == 2
        assert debugger.auto_indices["update_Z"] == -1
        assert debugger.last_checkpoints["initialize"] == []
        assert debugger.last_checkpoints["fit"] == [4]
        assert debugger.last_checkpoints["vi_step"] == [5]
        assert debugger.last_checkpoints["update_Z"] == []
        assert len(debugger.checkpoints) == 8
        assert len(debugger.data) == 2
        assert debugger.data[0] == ("initialize", [0, 1, 2, 3])
        assert debugger.data[1] == ("fit_1.vi_step_1.update_Z", [6, 7])

        # Check that the before command works properly.
        debugger.before("update_D")

        assert debugger.current_iid == "fit_1.vi_step_1.update_D"
        assert debugger.auto_indices["initialize"] == -1
        assert debugger.auto_indices["fit"] == 2
        assert debugger.auto_indices["vi_step"] == 2
        assert debugger.auto_indices["update_Z"] == -1
        assert debugger.last_checkpoints["initialize"] == []
        assert debugger.last_checkpoints["fit"] == [4]
        assert debugger.last_checkpoints["vi_step"] == [5]
        assert debugger.last_checkpoints["update_Z"] == []
        assert debugger.last_checkpoints["update_D"] == [8]
        assert len(debugger.checkpoints) == 9
        assert len(debugger.data) == 2
        assert debugger.data[0] == ("initialize", [0, 1, 2, 3])
        assert debugger.data[1] == ("fit_1.vi_step_1.update_Z", [6, 7])

        # Check that the after command works properly.
        debugger.after("update_D")

        assert debugger.current_iid == "fit_1.vi_step_1"
        assert debugger.auto_indices["initialize"] == -1
        assert debugger.auto_indices["fit"] == 2
        assert debugger.auto_indices["vi_step"] == 2
        assert debugger.auto_indices["update_Z"] == -1
        assert debugger.last_checkpoints["initialize"] == []
        assert debugger.last_checkpoints["fit"] == [4]
        assert debugger.last_checkpoints["vi_step"] == [5]
        assert debugger.last_checkpoints["update_Z"] == []
        assert debugger.last_checkpoints["update_D"] == []
        assert len(debugger.checkpoints) == 10
        assert len(debugger.data) == 3
        assert debugger.data[0] == ("initialize", [0, 1, 2, 3])
        assert debugger.data[1] == ("fit_1.vi_step_1.update_Z", [6, 7])
        assert debugger.data[2] == ("fit_1.vi_step_1.update_D", [8, 9])

        # Check that the after command works properly.
        debugger.after("vi_step")

        assert debugger.current_iid == "fit_1"
        assert debugger.auto_indices["initialize"] == -1
        assert debugger.auto_indices["fit"] == 2
        assert debugger.auto_indices["vi_step"] == 2
        assert debugger.auto_indices["update_Z"] == -1
        assert debugger.last_checkpoints["initialize"] == []
        assert debugger.last_checkpoints["fit"] == [4]
        assert debugger.last_checkpoints["vi_step"] == []
        assert debugger.last_checkpoints["update_Z"] == []
        assert debugger.last_checkpoints["update_D"] == []
        assert len(debugger.checkpoints) == 11
        assert len(debugger.data) == 4
        assert debugger.data[0] == ("initialize", [0, 1, 2, 3])
        assert debugger.data[1] == ("fit_1.vi_step_1", [5, 10])
        assert debugger.data[2] == ("fit_1.vi_step_1.update_Z", [6, 7])
        assert debugger.data[3] == ("fit_1.vi_step_1.update_D", [8, 9])

        # Check that the before command works properly.
        debugger.before("vi_step", auto_index=True)

        assert debugger.current_iid == "fit_1.vi_step_2"
        assert debugger.auto_indices["initialize"] == -1
        assert debugger.auto_indices["fit"] == 2
        assert debugger.auto_indices["vi_step"] == 3
        assert debugger.auto_indices["update_Z"] == -1
        assert debugger.last_checkpoints["initialize"] == []
        assert debugger.last_checkpoints["fit"] == [4]
        assert debugger.last_checkpoints["vi_step"] == [11]
        assert debugger.last_checkpoints["update_Z"] == []
        assert debugger.last_checkpoints["update_D"] == []
        assert len(debugger.checkpoints) == 12
        assert len(debugger.data) == 4
        assert debugger.data[0] == ("initialize", [0, 1, 2, 3])
        assert debugger.data[1] == ("fit_1.vi_step_1", [5, 10])
        assert debugger.data[2] == ("fit_1.vi_step_1.update_Z", [6, 7])
        assert debugger.data[3] == ("fit_1.vi_step_1.update_D", [8, 9])

        # Check that the before command works properly.
        debugger.before("update_Z")

        assert debugger.current_iid == "fit_1.vi_step_2.update_Z"
        assert debugger.auto_indices["initialize"] == -1
        assert debugger.auto_indices["fit"] == 2
        assert debugger.auto_indices["vi_step"] == 3
        assert debugger.auto_indices["update_Z"] == -1
        assert debugger.last_checkpoints["initialize"] == []
        assert debugger.last_checkpoints["fit"] == [4]
        assert debugger.last_checkpoints["vi_step"] == [11]
        assert debugger.last_checkpoints["update_Z"] == [12]
        assert debugger.last_checkpoints["update_D"] == []
        assert len(debugger.checkpoints) == 13
        assert len(debugger.data) == 4
        assert debugger.data[0] == ("initialize", [0, 1, 2, 3])
        assert debugger.data[1] == ("fit_1.vi_step_1", [5, 10])
        assert debugger.data[2] == ("fit_1.vi_step_1.update_Z", [6, 7])
        assert debugger.data[3] == ("fit_1.vi_step_1.update_D", [8, 9])

        # Check that the after command works properly.
        debugger.after("update_Z")

        assert debugger.current_iid == "fit_1.vi_step_2"
        assert debugger.auto_indices["initialize"] == -1
        assert debugger.auto_indices["fit"] == 2
        assert debugger.auto_indices["vi_step"] == 3
        assert debugger.auto_indices["update_Z"] == -1
        assert debugger.last_checkpoints["initialize"] == []
        assert debugger.last_checkpoints["fit"] == [4]
        assert debugger.last_checkpoints["vi_step"] == [11]
        assert debugger.last_checkpoints["update_Z"] == []
        assert debugger.last_checkpoints["update_D"] == []
        assert len(debugger.checkpoints) == 14
        assert len(debugger.data) == 5
        assert debugger.data[0] == ("initialize", [0, 1, 2, 3])
        assert debugger.data[1] == ("fit_1.vi_step_1", [5, 10])
        assert debugger.data[2] == ("fit_1.vi_step_1.update_Z", [6, 7])
        assert debugger.data[3] == ("fit_1.vi_step_1.update_D", [8, 9])
        assert debugger.data[4] == ("fit_1.vi_step_2.update_Z", [12, 13])

        # Check that the before command works properly.
        debugger.before("update_D")

        assert debugger.current_iid == "fit_1.vi_step_2.update_D"
        assert debugger.auto_indices["initialize"] == -1
        assert debugger.auto_indices["fit"] == 2
        assert debugger.auto_indices["vi_step"] == 3
        assert debugger.auto_indices["update_Z"] == -1
        assert debugger.last_checkpoints["initialize"] == []
        assert debugger.last_checkpoints["fit"] == [4]
        assert debugger.last_checkpoints["vi_step"] == [11]
        assert debugger.last_checkpoints["update_Z"] == []
        assert debugger.last_checkpoints["update_D"] == [14]
        assert len(debugger.checkpoints) == 15
        assert len(debugger.data) == 5
        assert debugger.data[0] == ("initialize", [0, 1, 2, 3])
        assert debugger.data[1] == ("fit_1.vi_step_1", [5, 10])
        assert debugger.data[2] == ("fit_1.vi_step_1.update_Z", [6, 7])
        assert debugger.data[3] == ("fit_1.vi_step_1.update_D", [8, 9])
        assert debugger.data[4] == ("fit_1.vi_step_2.update_Z", [12, 13])

        # Check that the after command works properly.
        debugger.after("update_D")

        assert debugger.current_iid == "fit_1.vi_step_2"
        assert debugger.auto_indices["initialize"] == -1
        assert debugger.auto_indices["fit"] == 2
        assert debugger.auto_indices["vi_step"] == 3
        assert debugger.auto_indices["update_Z"] == -1
        assert debugger.last_checkpoints["initialize"] == []
        assert debugger.last_checkpoints["fit"] == [4]
        assert debugger.last_checkpoints["vi_step"] == [11]
        assert debugger.last_checkpoints["update_Z"] == []
        assert debugger.last_checkpoints["update_D"] == []
        assert len(debugger.checkpoints) == 16
        assert len(debugger.data) == 6
        assert debugger.data[0] == ("initialize", [0, 1, 2, 3])
        assert debugger.data[1] == ("fit_1.vi_step_1", [5, 10])
        assert debugger.data[2] == ("fit_1.vi_step_1.update_Z", [6, 7])
        assert debugger.data[3] == ("fit_1.vi_step_1.update_D", [8, 9])
        assert debugger.data[4] == ("fit_1.vi_step_2.update_Z", [12, 13])
        assert debugger.data[5] == ("fit_1.vi_step_2.update_D", [14, 15])

        # Check that the after command works properly.
        debugger.after("vi_step")

        assert debugger.current_iid == "fit_1"
        assert debugger.auto_indices["initialize"] == -1
        assert debugger.auto_indices["fit"] == 2
        assert debugger.auto_indices["vi_step"] == 3
        assert debugger.auto_indices["update_Z"] == -1
        assert debugger.last_checkpoints["initialize"] == []
        assert debugger.last_checkpoints["fit"] == [4]
        assert debugger.last_checkpoints["vi_step"] == []
        assert debugger.last_checkpoints["update_Z"] == []
        assert debugger.last_checkpoints["update_D"] == []
        assert len(debugger.checkpoints) == 17
        assert len(debugger.data) == 7
        assert debugger.data[0] == ("initialize", [0, 1, 2, 3])
        assert debugger.data[1] == ("fit_1.vi_step_1", [5, 10])
        assert debugger.data[2] == ("fit_1.vi_step_1.update_Z", [6, 7])
        assert debugger.data[3] == ("fit_1.vi_step_1.update_D", [8, 9])
        assert debugger.data[4] == ("fit_1.vi_step_2", [11, 16])
        assert debugger.data[5] == ("fit_1.vi_step_2.update_Z", [12, 13])
        assert debugger.data[6] == ("fit_1.vi_step_2.update_D", [14, 15])

        # Check that the after command works properly.
        debugger.after("fit")

        assert debugger.current_iid == ""
        assert debugger.auto_indices["initialize"] == -1
        assert debugger.auto_indices["fit"] == 2
        assert debugger.auto_indices["vi_step"] == 3
        assert debugger.auto_indices["update_Z"] == -1
        assert debugger.last_checkpoints["initialize"] == []
        assert debugger.last_checkpoints["fit"] == []
        assert debugger.last_checkpoints["vi_step"] == []
        assert debugger.last_checkpoints["update_Z"] == []
        assert debugger.last_checkpoints["update_D"] == []
        assert len(debugger.checkpoints) == 18
        assert len(debugger.data) == 8
        assert debugger.data[0] == ("initialize", [0, 1, 2, 3])
        assert debugger.data[1] == ("fit_1", [4, 17])
        assert debugger.data[2] == ("fit_1.vi_step_1", [5, 10])
        assert debugger.data[3] == ("fit_1.vi_step_1.update_Z", [6, 7])
        assert debugger.data[4] == ("fit_1.vi_step_1.update_D", [8, 9])
        assert debugger.data[5] == ("fit_1.vi_step_2", [11, 16])
        assert debugger.data[6] == ("fit_1.vi_step_2.update_Z", [12, 13])
        assert debugger.data[7] == ("fit_1.vi_step_2.update_D", [14, 15])
