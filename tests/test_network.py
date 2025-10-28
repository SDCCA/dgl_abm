import pytest
import dgl_abm
import os
import dgl
import torch
from dgl import AddEdge, AddReverse

from src.dgl_abm.network.global_attachment import global_attachment
from src.dgl_abm.network.local_attachment import local_attachment, local_attachment_homophily
from src.dgl_abm.network.network_creation import network_creation
from src.dgl_abm.network.link_deletion import link_deletion
from src.dgl_abm.network.random_edge_noise import random_edge_noise


os.environ["DGLBACKEND"] = "pytorch"

@pytest.fixture
def model():
    model = dgl_abm.Model(experiment_identifier='network', root_path='test_models')
    model.set_model_parameters(overwrite=True, mode = "w", steering_parameters={
        "agent_attributes": {"AttributeA": {"distribution_type": "degenerate", "parameters": [3]}, 
                             "AttributeB": {"distribution_type": "random", "parameters": [0, 1]}},
        "noise_ratio": 0.25, 
        "attachment_ratio": 0.1, 
        "local_ratio": 0.1,
        "deletion_method": "size",
        "deletion_threshold": 3})
    model.initialize_model()
    return model
@pytest.fixture
def simple_graph():
    graph = dgl.graph(([0, 1, 1, 2,1,3,1,4], [1, 0, 2, 1, 3,1,4,1]))
    graph.edata['weight'] = torch.ones(graph.number_of_edges(), dtype=torch.float32)
    graph.ndata['AttributeA'] = torch.tensor([0, 1, 2, 3, 4])
    return graph


class TestGlobalAttachment:
    def test_global_attachment(self, model):
        ratio = 0.25
        current_number_of_edges = model.graph.number_of_edges()

        global_attachment(model.graph, ratio)
        updated_number_of_edges = model.graph.number_of_edges()
        assert updated_number_of_edges > 0
        assert updated_number_of_edges > ratio * current_number_of_edges
        assert updated_number_of_edges <= (1 + ratio) * current_number_of_edges

    def test_global_attachment_to_simple(self, model):
        agent_graph = model.graph
        params = model.steering_parameters
        model.step()

        # step operations on agent_graph
        global_attachment(agent_graph, params["attachment_ratio"])
        link_deletion(agent_graph, method=params['deletion_method'], threshold=params['deletion_threshold'])


        # edata are not copied by default
        simple_agent_graph = dgl.to_simple(agent_graph, return_counts='cnt')
        assert 'weight' not in simple_agent_graph.edata
        assert 'AttributeA' in simple_agent_graph.ndata  # check ndata

        # copy edata explicitly
        simple_agent_graph = dgl.to_simple(agent_graph, return_counts='cnt', copy_edata=True)
        assert 'weight' in simple_agent_graph.edata


class TestLinkDeletion:
    def test_link_deletion(self, model):
        deletion_method = model.steering_parameters['deletion_method']
        deletion_threshold = model.steering_parameters['deletion_threshold']
        current_number_of_edges = model.graph.number_of_edges()

        link_deletion(model.graph, method = deletion_method, threshold = deletion_threshold)
        updated_number_of_edges = model.graph.number_of_edges()

        assert updated_number_of_edges > 0
        assert updated_number_of_edges < current_number_of_edges


class TestLocalAttachment:
    def test_local_attachment(self, simple_graph):
        current_edges = simple_graph.number_of_edges()

        local_attachment(simple_graph, n_links=1)
        updated_edges = simple_graph.number_of_edges()
        assert updated_edges >= current_edges

        nodes = simple_graph.edges('all')[0][-2:] # new nodes

        # assert if the new nodes are in the graph
        assert nodes[0] in simple_graph.edges('all')[1]
        assert nodes[1] in simple_graph.edges('all')[1]

    def test_local_attachment_homophily(self, simple_graph):
        current_edges = simple_graph.number_of_edges()

        local_attachment_homophily(simple_graph, n_links=1, attribute={'AttributeA':{'characteristic_distance': 1e12}})
        updated_edges = simple_graph.number_of_edges()

        assert updated_edges >= current_edges

        nodes = simple_graph.edges('all')[0][-2:] # new nodes

        # assert if the new nodes are in the graph
        assert nodes[0] in simple_graph.edges('all')[1]
        assert nodes[1] in simple_graph.edges('all')[1]

class TestNetworkCreation:
    def test_network_creation_barabasi_albert(self, model):
        agent_graph = network_creation(model.config.number_agents, model.config.initial_graph_type)

        # TODO: fix: not very informative tests due to not setting the random generator
        assert agent_graph.number_of_nodes() == model.config.number_agents
        assert agent_graph.number_of_edges() == model.graph.number_of_edges()
        assert agent_graph.number_of_nodes() == model.graph.number_of_nodes()

    def test_network_creation_not_implemented(self, model):
        with pytest.raises(NotImplementedError):
            network_creation(model.config.number_agents, 'not_implemented')
