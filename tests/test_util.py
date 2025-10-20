import os

import pytest

import dgl_abm
import torch
import scipy.stats as stats
from dgl_abm.util.network_metrics import average_degree
from dgl_abm.util.network_metrics import average_weighted_degree
from dgl_abm.util.network_metrics import node_degree
from dgl_abm.util.network_metrics import node_weighted_degree
from dgl_abm.util.utils import sample_distribution_tensor

os.environ["DGLBACKEND"] = "pytorch"

@pytest.fixture
def model():
    model = dgl_abm.Model(experiment_identifier='util', root_path='test_models')
    # to make sure the results are reproducible
    model.set_model_parameters(
        overwrite=True,
        mode="w",
        initial_graph_args={'seed': 100, 'new_node_edges': 1},
        number_agents=100,
        initial_graph_type="barabasi-albert")
    model.initialize_model()
    return model


class TestNetworkMetrics:
    def test_average_degree_initialize(self, model):
        expected_ad = 3.96
        assert model.average_degree == expected_ad

    def test_average_degree_step(self, model):
        model.step()
        ad = average_degree(model.graph)
        assert model.average_degree == ad
        assert isinstance(ad, float)

    def test_average_weighted_degree(self, model):
        model.step()
        model.step()
        model.step()
        awd = average_weighted_degree(model.graph)
        assert isinstance(awd, float)
        assert awd == pytest.approx(3.96*2)  # default null step edge weights = timestep

    def test_node_degree(self, model):
        nd = node_degree(model.graph)
        assert isinstance(nd, torch.Tensor)
        assert nd.shape == (model.graph.num_nodes(),)

    def test_node_weighted_degree(self, model):
        model.step()
        nwd = node_weighted_degree(model.graph)
        assert isinstance(nwd, torch.Tensor)
        assert nwd.shape == (model.graph.num_nodes(),)

class TestDistributionSampling:

    def test_uniform(self):
        dist = sample_distribution_tensor("uniform", [0, 1], 5)
        assert isinstance(dist, torch.Tensor)
        assert dist.shape == (5,)
        assert torch.all(dist >= 0)
        assert torch.all(dist < 1)

    def test_uniform_3d(self):
        dist = sample_distribution_tensor("uniform", [1, 2], (5,1,2))
        assert isinstance(dist, torch.Tensor)
        assert dist.shape == (5, 1, 2)
        assert torch.all(dist >= 1)
        assert torch.all(dist < 2)
    
    def test_uniform_no_range(self):
        with pytest.raises(ValueError):
            sample_distribution_tensor("uniform", [3, 3], [10, 1])

    def test_normal(self):
        dist = sample_distribution_tensor("normal", [0, 1], 100)
        assert dist.shape == (100,)
        assert stats.kstest(dist.numpy(), "norm", args=(0, 1)).pvalue > 0.05
    
    def test_bernoulli(self):
        dist = sample_distribution_tensor("bernoulli", [0.5, None], [10,2])
        assert dist.shape == (10, 2)
        assert torch.all((dist == 0) | (dist == 1))

    def test_multinomial(self):
        dist = sample_distribution_tensor("multinomial", [[0.02, 0.98], [10, 20]], 5)
        assert dist.shape == (5,)
        assert torch.all((dist == 10) | (dist == 20))
        assert dist.sum() >= 20*3 + 10*2 # 0.0076832% chance of fewer than three 20s

    def test_multinomial_shape_rounding(self):
        dist = sample_distribution_tensor("multinomial", [[1], [10.3]], (5,2),rounding=True,decimals=0)
        assert dist.shape == (5,2)
        assert torch.all(dist == 10)

    def test_truncnorm(self):
        dist = sample_distribution_tensor("truncnorm", [0, 1, -2, 2], 3000)
        assert dist.shape == (3000,)
        assert torch.all(dist >= -2)
        assert torch.all(dist <= 2)

    def test_beta(self):
        dist = sample_distribution_tensor("beta", [2.0, 5.0], 10)
        assert dist.shape == (10,)
        assert torch.all((dist >= 0) & (dist <= 1))

    def test_random(self):
        dist0p = sample_distribution_tensor("random", [], 4)
        assert dist0p.shape == (4,)
        dist1p = sample_distribution_tensor("random", [5], 4)
        assert torch.all(dist1p < 5)
        dist2p = sample_distribution_tensor("random", [2, 5], 4)
        assert torch.all((dist2p >= 2) & (dist2p < 5))

    def test_degenerate_constant_scalar(self):
        dist = sample_distribution_tensor("degenerate", [3], 10)
        assert torch.all(dist == 3)
        assert dist.shape == (10,)

    def test_degenerate_constant_tensor(self):
        dist = sample_distribution_tensor("constant", [torch.tensor([1, 2])], 2)
        assert dist.shape == (2, 2)
        assert torch.equal(dist[0], torch.tensor([1, 2]))

    def test_invalid_sample_type(self):
        with pytest.raises(TypeError):
            sample_distribution_tensor("uniform", [0, 1], "square")

    def test_not_implemented(self):
        with pytest.raises(NotImplementedError):
            sample_distribution_tensor("undefined", [0, 1], 10)


    

