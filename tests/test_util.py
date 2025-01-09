import pytest
import dgl_abm
import os

from dgl_abm.util.network_metrics import average_degree

os.environ["DGLBACKEND"] = "pytorch"

@pytest.fixture
def model():
    model = dgl_abm.AgentBasedModel(model_identifier='util', root_path='data/test')
    # to make sure the results are reproducible
    model.set_model_parameters(
        overwrite=True,
        initial_graph_args={'seed': 100, 'new_node_edges': 1},
        number_agents=100,
        initial_graph_type="barabasi-albert")
    model.initialize_model()
    return model


class TestNetworkMetrics:
    def test_average_degree_initialize(self, model):
        expected_ad = 1.98
        assert model.average_degree == expected_ad

    def test_average_degree_step(self, model):
        model.step() # timestep 1
        ad = average_degree(model.graph)
        assert model.average_degree == ad
        assert isinstance(ad, float)
