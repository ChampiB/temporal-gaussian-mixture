import pytest
import torch
from unittest.mock import Mock

from tgm.agents.datasets.Dataset import Dataset
from tgm.agents.datasets.GaussianMixtureView import GaussianMixtureView
from tgm.agents.datasets.TemporalModelView import TemporalModelView


class TestDatasetAndViews:

    @pytest.fixture()
    def dataset(self):

        # Create a dataset.
        dataset = Dataset()

        # Create the observations.
        x0 = torch.ones([2]) * 0
        x1 = torch.ones([2]) * 1
        x2 = torch.ones([2]) * 2
        x3 = torch.ones([2]) * 3
        x4 = torch.ones([2]) * 4
        x5 = torch.ones([2]) * 5
        x6 = torch.ones([2]) * 6
        x7 = torch.ones([2]) * 7
        x8 = torch.ones([2]) * 8

        # Create the actions.
        a0 = 0
        a1 = 1
        a2 = 2
        a3 = 3
        a5 = 4
        a6 = 5
        a7 = 6

        # Create the boolean defining whether the trial ends.
        r1 = 0
        r2 = 0
        r3 = 0
        r4 = 1
        r6 = 0
        r7 = 0
        r8 = 0

        # Create the boolean defining whether the trial ends.
        d1 = False
        d2 = False
        d3 = False
        d4 = True
        d6 = False
        d7 = False
        d8 = False

        # Fill the dataset with the data points.
        dataset.start_new_trial(x0)
        dataset.append(x1, a0, r1, d1)
        dataset.append(x2, a1, r2, d2)
        dataset.append(x3, a2, r3, d3)
        dataset.append(x4, a3, r4, d4)
        dataset.start_new_trial(x5)
        dataset.append(x6, a5, r6, d6)
        dataset.append(x7, a6, r7, d7)
        dataset.append(x8, a7, r8, d8)
        return dataset

    @pytest.fixture()
    def gm_1(self):

        # Create a Gaussian mixture mock.
        gm = Mock()
        gm.compute_responsibilities.return_value = torch.tensor([
            [0.10, 0.90],
            [0.09, 0.91],
            [0.08, 0.92],
            [0.07, 0.93],
            [0.06, 0.94],
            [0.05, 0.95],
            [0.04, 0.96],
            [0.03, 0.97],
            [0.02, 0.98]
        ])
        gm.fixed_components.return_value = [1]
        return gm

    @pytest.fixture()
    def gm_2(self):

        # Create a Gaussian mixture mock.
        gm = Mock()
        gm.compute_responsibilities.return_value = torch.tensor([
            [0.10, 0.90],
            [0.91, 0.09],
            [0.08, 0.92],
            [0.93, 0.07],
            [0.06, 0.94],
            [0.95, 0.05],
            [0.04, 0.96],
            [0.97, 0.03],
            [0.02, 0.98]
        ])
        gm.fixed_components.return_value = [1]
        return gm

    @pytest.fixture()
    def gm_3(self):

        # Create a Gaussian mixture mock.
        gm = Mock()
        gm.compute_responsibilities.return_value = torch.tensor([
            [0.90, 0.10],
            [0.91, 0.09],
            [0.08, 0.92],
            [0.07, 0.93],
            [0.06, 0.94],
            [0.95, 0.05],
            [0.96, 0.04],
            [0.03, 0.97],
            [0.02, 0.98]
        ])
        gm.fixed_components.return_value = [0]
        return gm

    @pytest.fixture()
    def gm_4(self):

        # Create a Gaussian mixture mock.
        gm = Mock()
        gm.compute_responsibilities.return_value = torch.tensor([
            [0.10, 0.90],
            [0.91, 0.09],
            [0.92, 0.08],
            [0.93, 0.07],
            [0.06, 0.94],
            [0.05, 0.95],
            [0.96, 0.04],
            [0.97, 0.03],
            [0.98, 0.02]
        ])
        gm.fixed_components.return_value = [0]
        return gm

    @pytest.fixture()
    def gm_5(self):

        # Create a Gaussian mixture mock.
        gm = Mock()
        gm.compute_responsibilities.return_value = torch.tensor([
            [0.10, 0.90],
            [0.09, 0.91],
            [0.08, 0.92],
            [0.93, 0.07],
            [0.94, 0.06],
            [0.05, 0.95],
            [0.04, 0.96],
            [0.97, 0.03],
            [0.98, 0.02]
        ])
        gm.fixed_components.return_value = [0]
        return gm

    def test_dataset_creation(self, dataset):

        # Check that all lists contain nine elements.
        assert len(dataset.x) == 9
        assert len(dataset.a) == 9
        assert len(dataset.r) == 9
        assert len(dataset.d) == 9
        assert len(dataset.forgettable) == 9

        # Check that the observations were stored properly.
        assert dataset.x[0][0] == 0
        assert dataset.x[1][0] == 1
        assert dataset.x[2][0] == 2
        assert dataset.x[3][0] == 3
        assert dataset.x[4][0] == 4
        assert dataset.x[5][0] == 5
        assert dataset.x[6][0] == 6
        assert dataset.x[7][0] == 7
        assert dataset.x[8][0] == 8

        # Check that the actions were stored properly.
        assert dataset.a[0] == 0
        assert dataset.a[1] == 1
        assert dataset.a[2] == 2
        assert dataset.a[3] == 3
        assert dataset.a[4] is None
        assert dataset.a[5] == 4
        assert dataset.a[6] == 5
        assert dataset.a[7] == 6
        assert dataset.a[8] is None

        # Check that the rewards were stored properly.
        assert dataset.r[0] is None
        assert dataset.r[1] == 0
        assert dataset.r[2] == 0
        assert dataset.r[3] == 0
        assert dataset.r[4] == 1
        assert dataset.r[5] is None
        assert dataset.r[6] == 0
        assert dataset.r[7] == 0
        assert dataset.r[8] == 0

        # Check that the booleans (defining whether each trial ended) have been stored properly.
        assert dataset.d[0] is False
        assert dataset.d[1] is False
        assert dataset.d[2] is False
        assert dataset.d[3] is False
        assert dataset.d[4] is True
        assert dataset.d[5] is False
        assert dataset.d[6] is False
        assert dataset.d[7] is False
        assert dataset.d[8] is False

        # Check that the booleans (defining whether each data point can be forgotten) have been initialized.
        assert dataset.forgettable[0] is False
        assert dataset.forgettable[1] is False
        assert dataset.forgettable[2] is False
        assert dataset.forgettable[3] is False
        assert dataset.forgettable[4] is False
        assert dataset.forgettable[5] is False
        assert dataset.forgettable[6] is False
        assert dataset.forgettable[7] is False
        assert dataset.forgettable[8] is False

    @pytest.mark.parametrize("gm, expected_forgettable", [
        ("gm_1", [True, True, True, True, True, True, True, True, False]),
        ("gm_2", [False, False, False, False, False, False, False, False, False]),
        ("gm_3", [True, False, False, False, False, True, False, False, False]),
        ("gm_4", [False, False, True, False, False, False, False, True, False]),
        ("gm_5", [False, False, False, False, True, False, False, False, False])
    ])
    def test_update_forgettable_set(self, dataset, gm, expected_forgettable, request):

        # Get the Gaussian mixture mock object.
        gm = request.getfixturevalue(gm)

        # Check that the forgettable boolean have been properly updated.
        dataset.update_forgettable_set(gm)
        for forgettable, expected in zip(dataset.forgettable, expected_forgettable):
            assert forgettable == expected

    def test_forget_1(self, dataset, gm_1):

        # Forget the data points based on the first Gaussian mixture.
        dataset.update_forgettable_set(gm_1)
        dataset.forget()

        # Check that all lists contain only one element.
        assert len(dataset.x) == 1
        assert len(dataset.a) == 1
        assert len(dataset.r) == 1
        assert len(dataset.d) == 1
        assert len(dataset.forgettable) == 1

        # Check that the observations were forget properly.
        assert dataset.x[0][0] == 8

        # Check that the actions were forget properly.
        assert dataset.a[0] is None

        # Check that the rewards were forget properly.
        assert dataset.r[0] == 0

        # Check that the booleans (defining whether each trial ended) have been forgotten properly.
        assert dataset.d[0] is False

        # Check that the booleans (defining whether each data point can be forgotten) have been forgotten properly.
        assert dataset.forgettable[0] is False

    def test_forget_2(self, dataset, gm_2):

        # Forget the data points based on the second Gaussian mixture.
        dataset.update_forgettable_set(gm_2)
        dataset.forget()

        # Check that all lists contain nine elements.
        assert len(dataset.x) == 9
        assert len(dataset.a) == 9
        assert len(dataset.r) == 9
        assert len(dataset.d) == 9
        assert len(dataset.forgettable) == 9

        # Check that all the observations are still stored properly.
        assert dataset.x[0][0] == 0
        assert dataset.x[1][0] == 1
        assert dataset.x[2][0] == 2
        assert dataset.x[3][0] == 3
        assert dataset.x[4][0] == 4
        assert dataset.x[5][0] == 5
        assert dataset.x[6][0] == 6
        assert dataset.x[7][0] == 7
        assert dataset.x[8][0] == 8

        # Check that all the actions are still stored properly.
        assert dataset.a[0] == 0
        assert dataset.a[1] == 1
        assert dataset.a[2] == 2
        assert dataset.a[3] == 3
        assert dataset.a[4] is None
        assert dataset.a[5] == 4
        assert dataset.a[6] == 5
        assert dataset.a[7] == 6
        assert dataset.a[8] is None

        # Check that all the rewards are still stored properly.
        assert dataset.r[0] is None
        assert dataset.r[1] == 0
        assert dataset.r[2] == 0
        assert dataset.r[3] == 0
        assert dataset.r[4] == 1
        assert dataset.r[5] is None
        assert dataset.r[6] == 0
        assert dataset.r[7] == 0
        assert dataset.r[8] == 0

        # Check that all the booleans (defining whether each trial ended) are still stored properly.
        assert dataset.d[0] is False
        assert dataset.d[1] is False
        assert dataset.d[2] is False
        assert dataset.d[3] is False
        assert dataset.d[4] is True
        assert dataset.d[5] is False
        assert dataset.d[6] is False
        assert dataset.d[7] is False
        assert dataset.d[8] is False

        # Check that all the booleans (defining whether each data point can be forgotten) are unchanged.
        assert dataset.forgettable[0] is False
        assert dataset.forgettable[1] is False
        assert dataset.forgettable[2] is False
        assert dataset.forgettable[3] is False
        assert dataset.forgettable[4] is False
        assert dataset.forgettable[5] is False
        assert dataset.forgettable[6] is False
        assert dataset.forgettable[7] is False
        assert dataset.forgettable[8] is False

    def test_forget_3(self, dataset, gm_3):

        # Forget the data points based on the third Gaussian mixture.
        dataset.update_forgettable_set(gm_3)
        dataset.forget()

        # Check that all lists contain seven elements.
        assert len(dataset.x) == 7
        assert len(dataset.a) == 7
        assert len(dataset.r) == 7
        assert len(dataset.d) == 7
        assert len(dataset.forgettable) == 7

        # Check that the observations were stored properly.
        assert dataset.x[0][0] == 1
        assert dataset.x[1][0] == 2
        assert dataset.x[2][0] == 3
        assert dataset.x[3][0] == 4
        assert dataset.x[4][0] == 6
        assert dataset.x[5][0] == 7
        assert dataset.x[6][0] == 8

        # Check that the actions were stored properly.
        assert dataset.a[0] == 1
        assert dataset.a[1] == 2
        assert dataset.a[2] == 3
        assert dataset.a[3] is None
        assert dataset.a[4] == 5
        assert dataset.a[5] == 6
        assert dataset.a[6] is None

        # Check that the rewards were stored properly.
        assert dataset.r[0] == 0
        assert dataset.r[1] == 0
        assert dataset.r[2] == 0
        assert dataset.r[3] == 1
        assert dataset.r[4] == 0
        assert dataset.r[5] == 0
        assert dataset.r[6] == 0

        # Check that the booleans (defining whether each trial ended) have been stored properly.
        assert dataset.d[0] is False
        assert dataset.d[1] is False
        assert dataset.d[2] is False
        assert dataset.d[3] is True
        assert dataset.d[4] is False
        assert dataset.d[5] is False
        assert dataset.d[6] is False

        # Check that the booleans (defining whether each data point can be forgotten) have been initialized.
        assert dataset.forgettable[0] is False
        assert dataset.forgettable[1] is False
        assert dataset.forgettable[2] is False
        assert dataset.forgettable[3] is False
        assert dataset.forgettable[4] is False
        assert dataset.forgettable[5] is False
        assert dataset.forgettable[6] is False

    def test_forget_4(self, dataset, gm_4):

        # Forget the data points based on the fourth Gaussian mixture.
        dataset.update_forgettable_set(gm_4)
        dataset.forget()

        # Check that all lists contain seven elements.
        assert len(dataset.x) == 7
        assert len(dataset.a) == 7
        assert len(dataset.r) == 7
        assert len(dataset.d) == 7
        assert len(dataset.forgettable) == 7

        # Check that the observations were stored properly.
        assert dataset.x[0][0] == 0
        assert dataset.x[1][0] == 1
        assert dataset.x[2][0] == 3
        assert dataset.x[3][0] == 4
        assert dataset.x[4][0] == 5
        assert dataset.x[5][0] == 6
        assert dataset.x[6][0] == 8

        # Check that the actions were stored properly.
        assert dataset.a[0] == 0
        assert dataset.a[1] == 1
        assert dataset.a[2] == 3
        assert dataset.a[3] is None
        assert dataset.a[4] == 4
        assert dataset.a[5] == 5
        assert dataset.a[6] is None

        # Check that the rewards were stored properly.
        assert dataset.r[0] is None
        assert dataset.r[1] == 0
        assert dataset.r[2] == 0
        assert dataset.r[3] == 1
        assert dataset.r[4] is None
        assert dataset.r[5] == 0
        assert dataset.r[6] == 0

        # Check that the booleans (defining whether each trial ended) have been stored properly.
        assert dataset.d[0] is False
        assert dataset.d[1] is True
        assert dataset.d[2] is False
        assert dataset.d[3] is True
        assert dataset.d[4] is False
        assert dataset.d[5] is True
        assert dataset.d[6] is False

        # Check that the booleans (defining whether each data point can be forgotten) have been initialized.
        assert dataset.forgettable[0] is False
        assert dataset.forgettable[1] is False
        assert dataset.forgettable[2] is False
        assert dataset.forgettable[3] is False
        assert dataset.forgettable[4] is False
        assert dataset.forgettable[5] is False
        assert dataset.forgettable[6] is False

    def test_forget_5(self, dataset, gm_5):

        # Forget the data points based on the fifth Gaussian mixture.
        dataset.update_forgettable_set(gm_5)
        dataset.forget()

        # Check that all lists contain eight elements.
        assert len(dataset.x) == 8
        assert len(dataset.a) == 8
        assert len(dataset.r) == 8
        assert len(dataset.d) == 8
        assert len(dataset.forgettable) == 8

        # Check that the observations were stored properly.
        assert dataset.x[0][0] == 0
        assert dataset.x[1][0] == 1
        assert dataset.x[2][0] == 2
        assert dataset.x[3][0] == 3
        assert dataset.x[4][0] == 5
        assert dataset.x[5][0] == 6
        assert dataset.x[6][0] == 7
        assert dataset.x[7][0] == 8

        # Check that the actions were stored properly.
        assert dataset.a[0] == 0
        assert dataset.a[1] == 1
        assert dataset.a[2] == 2
        assert dataset.a[3] == 3
        assert dataset.a[4] == 4
        assert dataset.a[5] == 5
        assert dataset.a[6] == 6
        assert dataset.a[7] is None

        # Check that the rewards were stored properly.
        assert dataset.r[0] is None
        assert dataset.r[1] == 0
        assert dataset.r[2] == 0
        assert dataset.r[3] == 0
        assert dataset.r[4] is None
        assert dataset.r[5] == 0
        assert dataset.r[6] == 0
        assert dataset.r[7] == 0

        # Check that the booleans (defining whether each trial ended) have been stored properly.
        assert dataset.d[0] is False
        assert dataset.d[1] is False
        assert dataset.d[2] is False
        assert dataset.d[3] is True
        assert dataset.d[4] is False
        assert dataset.d[5] is False
        assert dataset.d[6] is False
        assert dataset.d[7] is False

        # Check that the booleans (defining whether each data point can be forgotten) have been initialized.
        assert dataset.forgettable[0] is False
        assert dataset.forgettable[1] is False
        assert dataset.forgettable[2] is False
        assert dataset.forgettable[3] is False
        assert dataset.forgettable[4] is False
        assert dataset.forgettable[5] is False
        assert dataset.forgettable[6] is False
        assert dataset.forgettable[7] is False

    def test_gaussian_mixture_view_1(self, dataset, gm_1):

        # Retrieve the full dataset, as well as the forget and keep splits.
        gm_data = GaussianMixtureView(dataset)
        dataset.update_forgettable_set(gm_1)
        x = gm_data.get()
        x_forget, x_keep = gm_data.get(split=True)

        # Check that full dataset has the correct shape.
        assert x.shape[0] == 9
        assert x.shape[1] == 2

        # Check that the observations in the full dataset were retrieved properly.
        for i in range(9):
            assert x[i][0] == i

        # Check that the forget and keep splits have the correct shape.
        assert x_forget.shape[0] == 8
        assert x_forget.shape[1] == 2
        assert x_keep.shape[0] == 1
        assert x_keep.shape[1] == 2

        # Check that the forgettable data points were retrieved properly.
        for i in range(8):
            assert x_forget[i][0] == i

        # Check that the data points to keep were retrieved properly.
        assert x_keep[0][0] == 8

    def test_gaussian_mixture_view_2(self, dataset, gm_2):

        # Retrieve the full dataset, as well as the forget and keep splits.
        gm_data = GaussianMixtureView(dataset)
        dataset.update_forgettable_set(gm_2)
        x = gm_data.get()
        x_forget, x_keep = gm_data.get(split=True)

        # Check that full dataset has the correct shape.
        assert x.shape[0] == 9
        assert x.shape[1] == 2

        # Check that the observations in the full dataset were retrieved properly.
        for i in range(9):
            assert x[i][0] == i

        # Check that the forget and keep splits have the correct shape.
        assert x_forget.shape[0] == 0
        assert x_keep.shape[0] == 9
        assert x_keep.shape[1] == 2

        # Check that the data points to keep were retrieved properly.
        for i in range(9):
            assert x_keep[i][0] == i

    def test_gaussian_mixture_view_3(self, dataset, gm_3):

        # Retrieve the full dataset, as well as the forget and keep splits.
        gm_data = GaussianMixtureView(dataset)
        dataset.update_forgettable_set(gm_3)
        x = gm_data.get()
        x_forget, x_keep = gm_data.get(split=True)

        # Check that full dataset has the correct shape.
        assert x.shape[0] == 9
        assert x.shape[1] == 2

        # Check that the observations in the full dataset were retrieved properly.
        for i in range(9):
            assert x[i][0] == i

        # Check that the forget and keep splits have the correct shape.
        assert x_forget.shape[0] == 2
        assert x_forget.shape[1] == 2
        assert x_keep.shape[0] == 7
        assert x_keep.shape[1] == 2

        # Check that the forgettable data points were retrieved properly.
        assert x_forget[0][0] == 0
        assert x_forget[1][0] == 5

        # Check that the data points to keep were retrieved properly.
        assert x_keep[0][0] == 1
        assert x_keep[1][0] == 2
        assert x_keep[2][0] == 3
        assert x_keep[3][0] == 4
        assert x_keep[4][0] == 6
        assert x_keep[5][0] == 7
        assert x_keep[6][0] == 8

    def test_gaussian_mixture_view_4(self, dataset, gm_4):

        # Retrieve the full dataset, as well as the forget and keep splits.
        gm_data = GaussianMixtureView(dataset)
        dataset.update_forgettable_set(gm_4)
        x = gm_data.get()
        x_forget, x_keep = gm_data.get(split=True)

        # Check that full dataset has the correct shape.
        assert x.shape[0] == 9
        assert x.shape[1] == 2

        # Check that the observations in the full dataset were retrieved properly.
        for i in range(9):
            assert x[i][0] == i

        # Check that the forget and keep splits have the correct shape.
        assert x_forget.shape[0] == 2
        assert x_forget.shape[1] == 2
        assert x_keep.shape[0] == 7
        assert x_keep.shape[1] == 2

        # Check that the forgettable data points were retrieved properly.
        assert x_forget[0][0] == 2
        assert x_forget[1][0] == 7

        # Check that the data points to keep were retrieved properly.
        assert x_keep[0][0] == 0
        assert x_keep[1][0] == 1
        assert x_keep[2][0] == 3
        assert x_keep[3][0] == 4
        assert x_keep[4][0] == 5
        assert x_keep[5][0] == 6
        assert x_keep[6][0] == 8

    def test_gaussian_mixture_view_5(self, dataset, gm_5):

        # Retrieve the full dataset, as well as the forget and keep splits.
        gm_data = GaussianMixtureView(dataset)
        dataset.update_forgettable_set(gm_5)
        x = gm_data.get()
        x_forget, x_keep = gm_data.get(split=True)

        # Check that full dataset has the correct shape.
        assert x.shape[0] == 9
        assert x.shape[1] == 2

        # Check that the observations in the full dataset were retrieved properly.
        for i in range(9):
            assert x[i][0] == i

        # Check that the forget and keep splits have the correct shape.
        assert x_forget.shape[0] == 1
        assert x_forget.shape[1] == 2
        assert x_keep.shape[0] == 8
        assert x_keep.shape[1] == 2

        # Check that the forgettable data points were retrieved properly.
        assert x_forget[0][0] == 4

        # Check that the data points to keep were retrieved properly.
        assert x_keep[0][0] == 0
        assert x_keep[1][0] == 1
        assert x_keep[2][0] == 2
        assert x_keep[3][0] == 3
        assert x_keep[4][0] == 5
        assert x_keep[5][0] == 6
        assert x_keep[6][0] == 7
        assert x_keep[7][0] == 8

    def test_temporal_model_view_1(self, dataset, gm_1):

        # Retrieve the full dataset, as well as the forget and keep splits.
        tm_data = TemporalModelView(dataset)
        dataset.update_forgettable_set(gm_1)
        r0, r1, a0 = tm_data.get(gm_1)
        (r0_forget, r1_forget, a0_forget), (r0_keep, r1_keep, a0_keep) = tm_data.get(gm_1, split=True)

        # Check that the full dataset has the correct shape.
        assert r0.shape[0] == 7
        assert r0.shape[1] == 2
        assert r1.shape[0] == 7
        assert r1.shape[1] == 2
        assert a0.shape[0] == 7
        assert a0.shape[1] == 1

        # Check that the responsibilities and actions in the full dataset were retrieved properly.
        assert r0[0][0] == 0.10
        assert r0[1][0] == 0.09
        assert r0[2][0] == 0.08
        assert r0[3][0] == 0.07
        assert r0[4][0] == 0.05
        assert r0[5][0] == 0.04
        assert r0[6][0] == 0.03

        assert r1[0][0] == 0.09
        assert r1[1][0] == 0.08
        assert r1[2][0] == 0.07
        assert r1[3][0] == 0.06
        assert r1[4][0] == 0.04
        assert r1[5][0] == 0.03
        assert r1[6][0] == 0.02

        for i in range(7):
            assert a0[i][0] == i

        # Check that the dataset of forgettable data points has the correct shape.
        assert r0_forget.shape[0] == 7
        assert r0_forget.shape[1] == 2
        assert r1_forget.shape[0] == 7
        assert r1_forget.shape[1] == 2
        assert a0_forget.shape[0] == 7
        assert a0_forget.shape[1] == 1

        # Check that the responsibilities and actions in the dataset of forgettable data points were retrieved properly.
        assert r0_forget[0][0] == 0.10
        assert r0_forget[1][0] == 0.09
        assert r0_forget[2][0] == 0.08
        assert r0_forget[3][0] == 0.07
        assert r0_forget[4][0] == 0.05
        assert r0_forget[5][0] == 0.04
        assert r0_forget[6][0] == 0.03

        assert r1_forget[0][0] == 0.09
        assert r1_forget[1][0] == 0.08
        assert r1_forget[2][0] == 0.07
        assert r1_forget[3][0] == 0.06
        assert r1_forget[4][0] == 0.04
        assert r1_forget[5][0] == 0.03
        assert r1_forget[6][0] == 0.02

        for i in range(7):
            assert a0_forget[i][0] == i

        # Check that the dataset of data points to keep has the correct shape.
        assert r0_keep.shape[0] == 0
        assert r1_keep.shape[0] == 0
        assert a0_keep.shape[0] == 0

    def test_temporal_model_view_2(self, dataset, gm_2):

        # Retrieve the full dataset, as well as the forget and keep splits.
        tm_data = TemporalModelView(dataset)
        dataset.update_forgettable_set(gm_2)
        r0, r1, a0 = tm_data.get(gm_2)
        (r0_forget, r1_forget, a0_forget), (r0_keep, r1_keep, a0_keep) = tm_data.get(gm_2, split=True)
        
        # Check that the full dataset has the correct shape.
        assert r0.shape[0] == 7
        assert r0.shape[1] == 2
        assert r1.shape[0] == 7
        assert r1.shape[1] == 2
        assert a0.shape[0] == 7
        assert a0.shape[1] == 1

        # Check that the responsibilities and actions in the full dataset were retrieved properly.
        assert r0[0][0] == 0.10
        assert r0[1][0] == 0.91
        assert r0[2][0] == 0.08
        assert r0[3][0] == 0.93
        assert r0[4][0] == 0.95
        assert r0[5][0] == 0.04
        assert r0[6][0] == 0.97

        assert r1[0][0] == 0.91
        assert r1[1][0] == 0.08
        assert r1[2][0] == 0.93
        assert r1[3][0] == 0.06
        assert r1[4][0] == 0.04
        assert r1[5][0] == 0.97
        assert r1[6][0] == 0.02

        for i in range(7):
            assert a0[i][0] == i

        # Check that the dataset of forgettable data points has the correct shape.
        assert r0_forget.shape[0] == 0
        assert r1_forget.shape[0] == 0
        assert a0_forget.shape[0] == 0

        # Check that the dataset of the data points to keep has the correct shape.
        assert r0_keep.shape[0] == 7
        assert r0_keep.shape[1] == 2
        assert r1_keep.shape[0] == 7
        assert r1_keep.shape[1] == 2
        assert a0_keep.shape[0] == 7
        assert a0_keep.shape[1] == 1

        # Check that the responsibilities and actions in the dataset of the data points to keep were retrieved properly.
        assert r0_keep[0][0] == 0.10
        assert r0_keep[1][0] == 0.91
        assert r0_keep[2][0] == 0.08
        assert r0_keep[3][0] == 0.93
        assert r0_keep[4][0] == 0.95
        assert r0_keep[5][0] == 0.04
        assert r0_keep[6][0] == 0.97

        assert r1_keep[0][0] == 0.91
        assert r1_keep[1][0] == 0.08
        assert r1_keep[2][0] == 0.93
        assert r1_keep[3][0] == 0.06
        assert r1_keep[4][0] == 0.04
        assert r1_keep[5][0] == 0.97
        assert r1_keep[6][0] == 0.02

        for i in range(7):
            assert a0_keep[i][0] == i

    def test_temporal_model_view_3(self, dataset, gm_3):

        # Retrieve the full dataset, as well as the forget and keep splits.
        tm_data = TemporalModelView(dataset)
        dataset.update_forgettable_set(gm_3)
        r0, r1, a0 = tm_data.get(gm_3)
        (r0_forget, r1_forget, a0_forget), (r0_keep, r1_keep, a0_keep) = tm_data.get(gm_3, split=True)

        # Check that the full dataset has the correct shape.
        assert r0.shape[0] == 7
        assert r0.shape[1] == 2
        assert r1.shape[0] == 7
        assert r1.shape[1] == 2
        assert a0.shape[0] == 7
        assert a0.shape[1] == 1

        # Check that the responsibilities and actions in the full dataset were retrieved properly.
        assert r0[0][0] == 0.90
        assert r0[1][0] == 0.91
        assert r0[2][0] == 0.08
        assert r0[3][0] == 0.07
        assert r0[4][0] == 0.95
        assert r0[5][0] == 0.96
        assert r0[6][0] == 0.03

        assert r1[0][0] == 0.91
        assert r1[1][0] == 0.08
        assert r1[2][0] == 0.07
        assert r1[3][0] == 0.06
        assert r1[4][0] == 0.96
        assert r1[5][0] == 0.03
        assert r1[6][0] == 0.02

        for i in range(7):
            assert a0[i][0] == i

        # Check that the dataset of forgettable data points has the correct shape.
        assert r0_forget.shape[0] == 2
        assert r0_forget.shape[1] == 2
        assert r1_forget.shape[0] == 2
        assert r1_forget.shape[1] == 2
        assert a0_forget.shape[0] == 2
        assert a0_forget.shape[1] == 1

        # Check that the responsibilities and actions in the full dataset were retrieved properly.
        assert r0_forget[0][0] == 0.90
        assert r0_forget[1][0] == 0.95

        assert r1_forget[0][0] == 0.91
        assert r1_forget[1][0] == 0.96

        assert a0_forget[0][0] == 0
        assert a0_forget[1][0] == 4

        # Check that the dataset of the data points to keep has the correct shape.
        assert r0_keep.shape[0] == 5
        assert r0_keep.shape[1] == 2
        assert r1_keep.shape[0] == 5
        assert r1_keep.shape[1] == 2
        assert a0_keep.shape[0] == 5
        assert a0_keep.shape[1] == 1

        # Check that the responsibilities and actions in the dataset of the data points to keep were retrieved properly.
        assert r0_keep[0][0] == 0.91
        assert r0_keep[1][0] == 0.08
        assert r0_keep[2][0] == 0.07
        assert r0_keep[3][0] == 0.96
        assert r0_keep[4][0] == 0.03

        assert r1_keep[0][0] == 0.08
        assert r1_keep[1][0] == 0.07
        assert r1_keep[2][0] == 0.06
        assert r1_keep[3][0] == 0.03
        assert r1_keep[4][0] == 0.02

        assert a0_keep[0][0] == 1
        assert a0_keep[1][0] == 2
        assert a0_keep[2][0] == 3
        assert a0_keep[3][0] == 5
        assert a0_keep[4][0] == 6

    def test_temporal_model_view_4(self, dataset, gm_4):

        # Retrieve the full dataset, as well as the forget and keep splits.
        tm_data = TemporalModelView(dataset)
        dataset.update_forgettable_set(gm_4)
        r0, r1, a0 = tm_data.get(gm_4)
        (r0_forget, r1_forget, a0_forget), (r0_keep, r1_keep, a0_keep) = tm_data.get(gm_4, split=True)

        # Check that the full dataset has the correct shape.
        assert r0.shape[0] == 7
        assert r0.shape[1] == 2
        assert r1.shape[0] == 7
        assert r1.shape[1] == 2
        assert a0.shape[0] == 7
        assert a0.shape[1] == 1

        # Check that the responsibilities and actions in the full dataset were retrieved properly.
        assert r0[0][0] == 0.10
        assert r0[1][0] == 0.91
        assert r0[2][0] == 0.92
        assert r0[3][0] == 0.93
        assert r0[4][0] == 0.05
        assert r0[5][0] == 0.96
        assert r0[6][0] == 0.97

        assert r1[0][0] == 0.91
        assert r1[1][0] == 0.92
        assert r1[2][0] == 0.93
        assert r1[3][0] == 0.06
        assert r1[4][0] == 0.96
        assert r1[5][0] == 0.97
        assert r1[6][0] == 0.98

        for i in range(7):
            assert a0[i][0] == i

        # Check that the dataset of forgettable data points has the correct shape.
        assert r0_forget.shape[0] == 4
        assert r0_forget.shape[1] == 2
        assert r1_forget.shape[0] == 4
        assert r1_forget.shape[1] == 2
        assert a0_forget.shape[0] == 4
        assert a0_forget.shape[1] == 1

        # Check that the responsibilities and actions in the full dataset were retrieved properly.
        assert r0_forget[0][0] == 0.91
        assert r0_forget[1][0] == 0.92
        assert r0_forget[2][0] == 0.96
        assert r0_forget[3][0] == 0.97

        assert r1_forget[0][0] == 0.92
        assert r1_forget[1][0] == 0.93
        assert r1_forget[2][0] == 0.97
        assert r1_forget[3][0] == 0.98

        assert a0_forget[0][0] == 1
        assert a0_forget[1][0] == 2
        assert a0_forget[2][0] == 5
        assert a0_forget[3][0] == 6

        # Check that the dataset of the data points to keep has the correct shape.
        assert r0_keep.shape[0] == 3
        assert r0_keep.shape[1] == 2
        assert r1_keep.shape[0] == 3
        assert r1_keep.shape[1] == 2
        assert a0_keep.shape[0] == 3
        assert a0_keep.shape[1] == 1

        # Check that the responsibilities and actions in the dataset of the data points to keep were retrieved properly.
        assert r0_keep[0][0] == 0.10
        assert r0_keep[1][0] == 0.93
        assert r0_keep[2][0] == 0.05

        assert r1_keep[0][0] == 0.91
        assert r1_keep[1][0] == 0.06
        assert r1_keep[2][0] == 0.96

        assert a0_keep[0][0] == 0
        assert a0_keep[1][0] == 3
        assert a0_keep[2][0] == 4

    def test_temporal_model_view_5(self, dataset, gm_5):

        # Retrieve the full dataset, as well as the forget and keep splits.
        tm_data = TemporalModelView(dataset)
        dataset.update_forgettable_set(gm_5)
        r0, r1, a0 = tm_data.get(gm_5)
        (r0_forget, r1_forget, a0_forget), (r0_keep, r1_keep, a0_keep) = tm_data.get(gm_5, split=True)

        # Check that the full dataset has the correct shape.
        assert r0.shape[0] == 7
        assert r0.shape[1] == 2
        assert r1.shape[0] == 7
        assert r1.shape[1] == 2
        assert a0.shape[0] == 7
        assert a0.shape[1] == 1

        # Check that the responsibilities and actions in the full dataset were retrieved properly.
        assert r0[0][0] == 0.10
        assert r0[1][0] == 0.09
        assert r0[2][0] == 0.08
        assert r0[3][0] == 0.93
        assert r0[4][0] == 0.05
        assert r0[5][0] == 0.04
        assert r0[6][0] == 0.97

        assert r1[0][0] == 0.09
        assert r1[1][0] == 0.08
        assert r1[2][0] == 0.93
        assert r1[3][0] == 0.94
        assert r1[4][0] == 0.04
        assert r1[5][0] == 0.97
        assert r1[6][0] == 0.98

        for i in range(7):
            assert a0[i][0] == i

        # Check that the dataset of forgettable data points has the correct shape.
        assert r0_forget.shape[0] == 1
        assert r0_forget.shape[1] == 2
        assert r1_forget.shape[0] == 1
        assert r1_forget.shape[1] == 2
        assert a0_forget.shape[0] == 1
        assert a0_forget.shape[1] == 1

        # Check that the responsibilities and actions in the full dataset were retrieved properly.
        assert r0_forget[0][0] == 0.93

        assert r1_forget[0][0] == 0.94

        assert a0_forget[0][0] == 3

        # Check that the dataset of the data points to keep has the correct shape.
        assert r0_keep.shape[0] == 6
        assert r0_keep.shape[1] == 2
        assert r1_keep.shape[0] == 6
        assert r1_keep.shape[1] == 2
        assert a0_keep.shape[0] == 6
        assert a0_keep.shape[1] == 1

        # Check that the responsibilities and actions in the dataset of the data points to keep were retrieved properly.
        assert r0_keep[0][0] == 0.10
        assert r0_keep[1][0] == 0.09
        assert r0_keep[2][0] == 0.08
        assert r0_keep[3][0] == 0.05
        assert r0_keep[4][0] == 0.04
        assert r0_keep[5][0] == 0.97

        assert r1_keep[0][0] == 0.09
        assert r1_keep[1][0] == 0.08
        assert r1_keep[2][0] == 0.93
        assert r1_keep[3][0] == 0.04
        assert r1_keep[4][0] == 0.97
        assert r1_keep[5][0] == 0.98

        assert a0_keep[0][0] == 0
        assert a0_keep[1][0] == 1
        assert a0_keep[2][0] == 2
        assert a0_keep[3][0] == 4
        assert a0_keep[4][0] == 5
        assert a0_keep[5][0] == 6
