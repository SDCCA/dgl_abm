import os
import pytest
import dgl_ptm
import torch
from dgl_ptm.agent.capital_update import capital_update
from dgl_ptm.agent.income_generation import income_generation
from dgl_ptm.agent.wealth_consumption import wealth_consumption
from dgl_ptm.agent.agent_update import agent_update
from dgl_ptm.environment.grid_creation import grid_creation, GridEnvironment
import dgl_ptm.agent.movement as movement


os.environ["DGLBACKEND"] = "pytorch"

@pytest.fixture
def model():
    model = dgl_ptm.PovertyTrapModel(model_identifier='agent', root_path='test_models')
    model.set_model_parameters(overwrite=True)
    model.initialize_model()
    return model

@pytest.fixture
def test_pt_file(tmp_path):
    example_grid = torch.tensor([[[10], [0.05], [0.05]], [[0], [0], [0]], [[0], [0], [0]]])
    file_path = tmp_path / "movement_grid.pt"
    torch.save(example_grid, file_path)
    return file_path

@pytest.fixture
def grid_environment(test_pt_file):
    kwargs = {
        'method': 'custom_import',
        'path': str(test_pt_file),
        'properties': {'propertyA': 0}
    }

    grid_environment = grid_creation(**kwargs)
    return grid_environment

class TestCapitalUpdate:

    def test_capital_update_default(self, model):

        k = model.graph.ndata['wealth']
        c = model.graph.ndata['wealth_consumption']
        i_a = model.graph.ndata['i_a']
        m = model.graph.ndata['m']
        global_θ = model.steering_parameters['global_theta'][0]
        𝛿 = model.steering_parameters['depreciation']
        # generate random income with size 100
        income = torch.rand(100)
        expected_wealth = (global_θ + m * (1-global_θ)) * (income - c - i_a + (1-𝛿) * k)

        # because income_generation is not called yet
        model.graph.ndata['income'] = income

        capital_update(
            model.graph,
            model.steering_parameters,
            1,
            "present_shock"
            )

        # assert _agent_capital_update was called
        assert (model.graph.ndata['wealth'] == expected_wealth).all()

    def test_capital_update_other(self, model):
        # assert NotImplementedError was raised
        with pytest.raises(NotImplementedError):
            capital_update(
                model.graph,
                1,
                method='other'
                )


class TestIncomeGeneration:

    def test_income_generation_default(self, model):
        gamma = model.steering_parameters['tech_gamma']
        cost = model.steering_parameters['tech_cost']
        alpha = model.graph.ndata['alpha']
        wealth = model.graph.ndata['wealth']
        expected_income,expected_tech_index = torch.max((alpha[:,None]*wealth[:,None]**gamma - cost), axis=1)

        income_generation(
            model.graph,
            model.config.device,
            model.steering_parameters,
            method='income_generation'
            )

        # assert _agent_income_generator was called
        assert (model.graph.ndata['income'] == expected_income).all()
        assert (model.graph.ndata['tech_index'] == expected_tech_index).all()

    def test_income_generation_other(self, model):
        # assert NotImplementedError was raised
        with pytest.raises(NotImplementedError):
            income_generation(
                model.graph,
                model.config.device,
                model.steering_parameters,
                method='other'
                )


class TestWealthConsumption:

    def test_wealth_consumption_default(self, model):
        wealth = model.graph.ndata['wealth']
        expected_wealth_consumption = wealth*1./3.

        wealth_consumption(
            model.graph,
            model.steering_parameters,
            1,
            model.config.device
        )

        # assert _fitted_agent_wealth_consumption was called
        assert (model.graph.ndata['wealth_consumption'] == expected_wealth_consumption).all()

    def test_wealth_consumption_other(self, model):
        # assert NotImplementedError was raised
        with pytest.raises(NotImplementedError):
            wealth_consumption(
                model.graph,
                model.steering_parameters,
                method='other'
                )

    def test_wealth_consumption_bellman(self, model):
        # assert NotImplementedError was raised
        with pytest.raises(NotImplementedError):
            wealth_consumption(
                model.graph,
                model.steering_parameters,
                method='bellman_consumption'
                )


class TestAgentUpdate:
    def test_agent_update_default(self, model):
        # assert NotImplementedError was raised
        with pytest.raises(NotImplementedError):
            agent_update(
                model.graph,
                model_params=model.steering_parameters,
                device=model.config.device,
                timestep=1,
                method='unavailable'
                )

    def test_agent_update_capital(self, model):
        # because income_generation is not called yet
        model.graph.ndata['income'] = torch.rand(100)
        agent_update(
            model.graph,
            model_params=model.steering_parameters,
            timestep=1,
            method='capital'
            )
        assert (model.graph.ndata['wealth'] != 0).all()

    def test_agent_update_theta(self, model):
        agent_update(
            model.graph,
            device=model.config.device,
            model_params=model.steering_parameters,
            timestep=1,
            method='theta'
            )
        assert (model.graph.ndata['theta'] != 0).all()

    def test_agent_update_consumption(self, model):
        agent_update(
            model.graph,
            device=model.config.device,
            model_params=model.steering_parameters,
            timestep=1,
            method='consumption'
            )
        assert (model.graph.ndata['wealth_consumption'] != 0).all()

    def test_agent_update_income(self, model):
        agent_update(
            model.graph,
            device=model.config.device,
            model_params=model.steering_parameters,
            timestep=1,
            method='income'
            )

        assert (model.graph.ndata['income'] != 0).all()
        assert (model.graph.ndata['tech_index'] == 0).all() # all indices are 0

    def test_agent_update_degree(self, model):
        model.graph.ndata['degree'] = torch.full((model.graph.num_nodes(),), 0)
        agent_update(model.graph,
            method='degree'
            )
        assert (model.graph.ndata['degree'] != 0).all()
    
    def test_agent_update_degree(self, model):
        model.graph.ndata['weighted_degree'] = torch.full((model.graph.num_nodes(),), 0)
        agent_update(model.graph,
            method='weighted_degree'
            )
        assert (model.graph.ndata['weighted_degree'] != 0).all()

    def test_agent_update_position(self,model,grid_environment):
        model.graph.ndata['x'] = torch.full((model.graph.num_nodes(),), 0)
        model.graph.ndata['y'] = torch.full((model.graph.num_nodes(),), 0)
        agentIDs = torch.tensor([0,1,2,3,4])
        new_positions = torch.tensor([[0,1],[0,2],[1,0],[1,1],[2,0]])
        agent_update(model.graph,
            grid_environment=grid_environment,
            model_params=model.steering_parameters,
            device=model.config.device,
            method='position',
            agentIDs=agentIDs,
            new_positions=new_positions
            )
        updated_x = model.graph.ndata['x'][agentIDs]
        updated_y = model.graph.ndata['y'][agentIDs]
        assert updated_x.flatten().tolist() == [0,0,1,1,2]
        assert updated_y.flatten().tolist() == [1,2,0,1,0]

class TestAgentMovement:

    def test_square_mask(self):
        mask = movement._square(0, 1)
        assert mask.shape == (3, 3)
        assert torch.all(mask == 1)

    def test_square_mask_min(self):
        mask = movement._square(1, 1)
        assert mask.shape == (3, 3)
        assert mask[1,1] == 0
        assert mask.sum() == 8

    def test_circle_mask(self):
        mask = movement._circle(2,3)
        assert mask.shape == (7, 7)
        assert mask[2:4, 2:4].sum() == 0 
        assert mask.sum() == 28

    def test_radial_mask(self):
        mask = movement._radial(0, 2)
        assert mask.shape == (5, 5)
        assert mask.sum() == 17

    def test_cross_mask(self):
        mask = movement._cross(1, 2)
        assert mask.shape == (5, 5)
        assert mask[2,2] == 0
        assert mask.sum() == 8

    def test_move_agents_all_random_jump(self, model, grid_environment):
        model.graph.ndata['x'] = torch.full((model.graph.num_nodes(),), 99)
        model.graph.ndata['y'] = torch.full((model.graph.num_nodes(),), 99)
        model.steering_parameters['movement_function'] = "random_jump"
        movement.move_agents(model.graph, 
                    model.steering_parameters, 
                    grid_environment, 
                    model.config.device)
        assert (model.graph.ndata['x'] != 99).all()
        assert (model.graph.ndata['y'] != 99).all()
        assert (model.graph.ndata['x'] < grid_environment.grid_shape[0]).all()
        assert (model.graph.ndata['y'] < grid_environment.grid_shape[1]).all()
        assert (model.graph.ndata['x'] >= 0).all()
        assert (model.graph.ndata['y'] >= 0).all()

    def test_randomly_select_random_agents(self, model):
        agentIDs = torch.tensor([-99])
        model.steering_parameters['moving_agents'] = "random"
        agentIDs = movement._identify_moving_agents(model.graph, model.steering_parameters)
        assert agentIDs[0].item() >= 0
        assert agentIDs.shape[0] >= 0
        assert agentIDs.shape[0] <= model.graph.num_nodes()
        assert torch.unique(agentIDs).shape[0] == agentIDs.shape[0]

    def test_select_ratio_agents(self, model, grid_environment):
        agentIDs = torch.tensor([-99])
        model.steering_parameters['moving_agents'] = "ratio_random"
        model.steering_parameters['ratio_moving_agents'] = 0.5
        agentIDs = movement._identify_moving_agents(model.graph, model.steering_parameters)
        assert  (agentIDs >= 0).all().item()
        assert (agentIDs < model.graph.num_nodes()).all().item()
        assert torch.unique(agentIDs).shape[0] == agentIDs.shape[0]
        assert agentIDs.shape[0] == int(model.steering_parameters['ratio_moving_agents']*model.graph.num_nodes())

    def test_select_n_agents(self, model, grid_environment):
        agentIDs = torch.tensor([-99])
        model.steering_parameters['moving_agents'] = "n_random"
        model.steering_parameters['n_moving_agents'] = int(0.5*model.graph.num_nodes())
        agentIDs = movement._identify_moving_agents(model.graph, model.steering_parameters)
        assert (agentIDs >= 0).all().item()
        assert (agentIDs < model.graph.num_nodes()).all().item()
        assert torch.unique(agentIDs).shape[0] == agentIDs.shape[0]
        assert agentIDs.shape[0] == model.steering_parameters['n_moving_agents']

    def test_select_probable_agents(self, model, grid_environment):
        agentIDs = torch.tensor([-99])
        model.graph.ndata['move_prob'] = torch.randint(0, 1, (model.graph.num_nodes(),))
        model.steering_parameters['moving_agents'] = "variable_based"
        model.steering_parameters['moving_probability_variable'] = "move_prob"
        agentIDs = movement._identify_moving_agents(model.graph, model.steering_parameters)
        assert (agentIDs >= 0).all().item()
        assert (agentIDs < model.graph.num_nodes()).all().item()
        assert torch.unique(agentIDs).shape[0] == agentIDs.shape[0]
        assert agentIDs.shape[0] == model.graph.ndata['move_prob'].sum()

    def test_position_impossible(self, model,grid_environment):
        model.graph.ndata['x'] = torch.full((model.graph.num_nodes(),), 99)
        model.graph.ndata['y'] = torch.full((model.graph.num_nodes(),), 99)
        agentIDs = torch.tensor([0])
        model.steering_parameters['movement_function']="square"
        new_positions = movement._select_agent_positions(model.graph, model.steering_parameters, grid_environment, agentIDs)
        assert model.graph.ndata['x'][0]==99
        assert model.graph.ndata['x'][0]==99

    def test_position_bounded(self, model, grid_environment):
        model.graph.ndata['x'] = torch.full((model.graph.num_nodes(),), grid_environment.grid_shape[0]-1)
        model.graph.ndata['y'] = torch.full((model.graph.num_nodes(),), grid_environment.grid_shape[1]-1)
        agentIDs = torch.tensor([0,1])
        model.steering_parameters['movement_function']={"pattern": "square","range_min": 1, "range_max": 1}
        new_positions,_ = movement._select_agent_positions(model.graph, model.steering_parameters,grid_environment, agentIDs)
        assert new_positions[0][0]<=grid_environment.grid_shape[0]-1
        assert new_positions[0][1]<=grid_environment.grid_shape[1]-1
        assert new_positions[0][0]-new_positions[0][1] <=1
        assert (new_positions[0][0]!=grid_environment.grid_shape[0]-1 or
                new_positions[0][1]!=grid_environment.grid_shape[1]-1)

    def test_position_probability(self, model, grid_environment):
        # Also tests range max warning
        model.graph.ndata['x'] = torch.full((model.graph.num_nodes(),), grid_environment.grid_shape[0]-1)
        model.graph.ndata['y'] = torch.full((model.graph.num_nodes(),), grid_environment.grid_shape[1]-1)
        agentIDs = torch.tensor([0,1])
        model.steering_parameters['movement_function']={"pattern": "square","range_min": 0, "range_max": max([grid_environment.grid_shape[0], grid_environment.grid_shape[1]]), "seeking_property": "propertyA", "seeking_behavior": "probability"}
        new_positions, _ = movement._select_agent_positions(model.graph, model.steering_parameters, grid_environment, agentIDs)
        assert new_positions[0][1]==0

    def test_position_max(self, model, grid_environment):
        model.graph.ndata['x'] = torch.full((model.graph.num_nodes(),), grid_environment.grid_shape[0]-1)
        model.graph.ndata['y'] = torch.full((model.graph.num_nodes(),), grid_environment.grid_shape[1]-1)
        agentIDs = torch.tensor([0,1])
        model.steering_parameters['movement_function']={"pattern": "square","range_min": 0, "range_max": max([grid_environment.grid_shape[0]-1, grid_environment.grid_shape[1]-1]), "seeking_property": "propertyA", "seeking_behavior": "maximum"}
        new_positions,_ = movement._select_agent_positions(model.graph, model.steering_parameters, grid_environment, agentIDs)
        assert new_positions[0][0]==0
        assert new_positions[0][1]==0

    def test_position_min(self, model, grid_environment):
        model.graph.ndata['x'] = torch.full((model.graph.num_nodes(),), grid_environment.grid_shape[0]-1)
        model.graph.ndata['y'] = torch.full((model.graph.num_nodes(),), grid_environment.grid_shape[1]-1)
        agentIDs = torch.tensor([0])
        model.steering_parameters['movement_function']={"pattern": "square","range_min": 0, "range_max": max([grid_environment.grid_shape[0]-1, grid_environment.grid_shape[1]-1]), "seeking_property": "propertyA", "seeking_behavior": "min"}
        new_positions,_ = movement._select_agent_positions(model.graph, model.steering_parameters, grid_environment, agentIDs)
        assert new_positions[0][0]!=0
