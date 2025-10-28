import os
import shutil
from pathlib import Path

import pytest
import torch
import xarray as xr

import dgl_abm
from src.dgl_abm.model.data_collection import data_collection

os.environ["DGLBACKEND"] = "pytorch"

@pytest.fixture
def model():
    model = dgl_abm.Model(
        experiment_identifier='null_step', root_path='test_models'
        )
    model.set_model_parameters(overwrite=True, mode= "w")
    model.initialize_model()
    return model

@pytest.fixture
def data_collection_model():
    model = dgl_abm.Model(
        experiment_identifier='data_collection', root_path='test_models'
        )
    model.set_model_parameters(overwrite=True, mode = "w",steering_parameters={
            "agent_attributes": {"AttributeA": {"distribution_type": "degenerate", "parameters": [3]}, 
                                 "AttributeB": {"distribution_type": "random", "parameters": [0, 1]}}})
    model.initialize_model()
    return model

@pytest.fixture
def initialization_model():
    model = dgl_abm.Model(
        experiment_identifier='initialize_model', root_path='test_models'
        )
    model.set_model_parameters(overwrite=True, mode= "w")
    model.initialize_model()
    return model


@pytest.fixture
def config_file(tmp_path):
    with open(tmp_path / 'test_config.yaml', 'w') as file:
        file.write(
        "experiment_identifier: new_model\n"
        "mode: w\n"
        "steering_parameters:\n"
        "  deletion_method: probability\n"
        "  agent_attributes:\n"
        "    AttributeA:\n"
        "      distribution_type: degenerate\n"
        "      parameters: [3]\n"
        "    AttributeB:\n"
        "      distribution_type: random\n"
        "      parameters: [0, 1]\n"
        )
    return tmp_path / 'test_config.yaml'


class TestStep:
    def test_abm_time_step0(self, model):
        model.step()

        assert model.step_count==1
        assert model.graph.num_nodes() == 100
        assert model.experiment_identifier == 'null_step'
        assert Path('test_models/null_step/agent_data.zarr').exists()
        assert Path('test_models/null_step/edge_data/0.zarr').exists()

    def test_abm_time_step1(self, model):
         model.step()
         model.step()
         assert Path('test_models/null_step/edge_data/1.zarr').exists()

    def test_data_collection_period(self, model):
        if Path('test_models/null_step/edge_data/').exists():
            shutil.rmtree('test_models/null_step/edge_data/')

        model.config.data_collection_period = 3

        model.run()

        assert model.step_count == 5
        assert Path('test_models/null_step/agent_data.zarr').exists()
        assert not Path('test_models/null_step/edge_data/2.zarr').exists()
        assert Path('test_models/null_step/edge_data/3.zarr').exists()

    def test_data_collection_list(self, model):
        if Path('test_models/null_step/edge_data/').exists():
            shutil.rmtree('test_models/null_step/edge_data/')

        model.config.data_collection_period = None
        model.config.data_collection_step_list = [1, 4]

        model.run()

        assert model.step_count == 5
        assert Path('test_models/null_step/agent_data.zarr').exists()
        assert not Path('test_models/null_step/edge_data/0.zarr').exists()
        assert Path('test_models/null_step/edge_data/1.zarr').exists()
        assert not Path('test_models/null_step/edge_data/2.zarr').exists()
        assert not Path('test_models/null_step/edge_data/3.zarr').exists()
        assert Path('test_models/null_step/edge_data/4.zarr').exists()

    def test_data_collection_period_and_list(self, model):
        if Path('test_models/null_step/edge_data/').exists():
            shutil.rmtree('test_models/null_step/edge_data/')

        model.config.step_target = 10 # run the model till step 10

        # Set periodical progress check as well as
        # collecting data before and after specific step and at the end of the
        # process.
        # Note that the pediod and list could have overlapping values;
        # this will result in collecting the data once at that step.
        model.config.data_collection_period = 4
        model.config.data_collection_step_list = [4, 5, 9]

        model.run()

        assert model.step_count == 10
        assert Path('test_models/null_step/agent_data.zarr').exists()

        assert Path('test_models/null_step/edge_data/0.zarr').exists()
        assert Path('test_models/null_step/edge_data/4.zarr').exists()
        assert Path('test_models/null_step/edge_data/8.zarr').exists()

        assert Path('test_models/null_step/edge_data/5.zarr').exists()
        assert Path('test_models/null_step/edge_data/9.zarr').exists()


class TestDataCollection:
    def test_data_collection(self, data_collection_model):
        model = data_collection_model
        data_collection(model.graph,
                        timestep=0,
                        parameters= model.config.__dict__,
                        npath = model.npath,
                        epath = model.epath
                        )

        assert Path('test_models/data_collection/agent_data.zarr').exists()
        assert Path('test_models/data_collection/edge_data/0.zarr').exists()

    def test_data_collection_timestep1(self, data_collection_model):
        model = data_collection_model
        model.step() # timestep 0
        data_collection(model.graph,
                        timestep=1,
                        parameters= model.config.__dict__,
                        npath = model.npath,
                        epath = model.epath)

        assert Path('test_models/data_collection/agent_data.zarr').exists()
        assert Path('test_models/data_collection/edge_data/0.zarr').exists()
        assert Path('test_models/data_collection/edge_data/1.zarr').exists()

        # check if dimension 'n_time' exist in agent_data.zarr
        agent_data = xr.open_zarr('test_models/data_collection/agent_data.zarr')
        assert 'n_time' in agent_data.dims

        # check variable names in edge_data/1.zarr
        edge_data = xr.open_zarr('test_models/data_collection/edge_data/1.zarr')
        assert 'weight' in edge_data.variables


class TestInitializeModel:
    def test_set_model_parameters(self):
        model = dgl_abm.Model(
            experiment_identifier='initialize_model', root_path='test_models'
            )
        model.set_model_parameters(overwrite=True)

        assert model.step_count == 0
        initialize_dir ='test_models/initialize_model'
        assert Path(model.npath) == Path(f'{initialize_dir}/agent_data.zarr')  # noqa: E501
        assert Path(model.epath) == Path(f'{initialize_dir}/edge_data') # noqa: E501
        assert Path(f'{initialize_dir}/initialize_model_0.yaml').exists()
        assert model.config.edata == ['all']
        assert model.config.format == 'xarray'
        assert model.config.number_agents == 100
        assert model.config.step_target == 5

    def test_set_model_parameters_with_file(self, config_file):
        model = dgl_abm.Model(
            experiment_identifier='initialize_model', root_path='test_models'
            )
        model.set_model_parameters(parameter_file_path=config_file)

        assert model.experiment_identifier == 'initialize_model'
        assert model.steering_parameters['deletion_method'] == "probability"
        assert model.steering_parameters['deletion_threshold'] == None
        assert model.steering_parameters['step_type'] == 'default'
        assert model.config.number_agents == 100

    def test_set_model_parameters_with_kwargs(self):
        model = dgl_abm.Model(
            experiment_identifier='initialize_model', root_path='test_models'
            )
        model.set_model_parameters(
            steering_parameters={'deletion_threshold': 0.04}
            )

        assert model.steering_parameters['deletion_method'] == None
        assert model.steering_parameters['deletion_threshold'] == 0.04
        assert model.config.number_agents == 100

    def test_set_model_parameters_with_file_and_kwargs(self, config_file):
        model = dgl_abm.Model(
            experiment_identifier='initialize_model', root_path='test_models'
            )
        model.set_model_parameters(
            parameter_file_path=config_file,
            steering_parameters={'deletion_threshold': 0.06}
            )
        # Note, not 'new_model' as set in config_file.
        assert model.experiment_identifier == 'initialize_model'
        assert model.steering_parameters['deletion_method'] == 'probability'
        assert model.steering_parameters['deletion_threshold'] == 0.06
        assert model.steering_parameters['step_type'] == "default"
        assert model.config.number_agents == 100

    def test_save_model_parameters(self):
        model = dgl_abm.Model(
            experiment_identifier='initialize_model', root_path='test_models'
            )
        model.set_model_parameters(overwrite=True)
        model.save_model_parameters(overwrite=False)
        model.save_model_parameters()

        # Saving to unique config files becomes useful when starting multiple
        # runs from the same step.
        assert Path('test_models/initialize_model/initialize_model_0.yaml').exists()
        assert Path('test_models/initialize_model/initialize_model_0_1.yaml').exists()
        assert Path('test_models/initialize_model/initialize_model_0_2.yaml').exists()

    def test_initialize_model(self, initialization_model):
        model = initialization_model
        assert model.graph is not None
        assert model.number_of_edges is not None
        assert str(model.graph.device) == 'cpu'

    def test_create_network(self):
        model = dgl_abm.Model(
            experiment_identifier='initialize_model', root_path='test_models'
            )
        model.set_model_parameters(overwrite=True)
        model.create_network()

        assert model.graph is not None
        assert model.graph.number_of_nodes() == 100

    def test_initialize_global_properties(self):
        model = dgl_abm.Model(
            experiment_identifier='initialize_model', root_path='test_models'
            )
        model.set_model_parameters(overwrite=True, steering_parameters={
                "global_properties": {"GlobalParameterDist": {"distribution": {"distribution_type": "uniform", 'parameters': [1,2] },"shape":1},
                                      "GlobalParameterList": [1,2],
                                      "GlobalParameterInt": 1}})
        model.create_network()
        model.initialize_global_properties()

        assert model.global_properties["GlobalParameterDist"]
        assert torch.all(model.global_properties["GlobalParameterDist"] <= 2)
        assert torch.all(model.global_properties["GlobalParameterDist"] > 1)
        assert torch.equal(model.global_properties["GlobalParameterList"], torch.tensor([1,2]))
        assert torch.equal(model.global_properties["GlobalParameterInt"], torch.tensor(1))

    def test_initialize_agent_attributes(self):
        model = dgl_abm.Model(
            experiment_identifier='initialize_model', root_path='test_models'
            )
        model.set_model_parameters(overwrite=True)
        model.create_network()
        model.initialize_agent_attributes()

        assert model.graph.ndata['DefaultAttribute'] is not None
    
    def test_initialize_agent_attributes_wrong_length(self):
        model = dgl_abm.Model(
            experiment_identifier='initialize_model', root_path='test_models'
            )
        model.set_model_parameters(overwrite=True, steering_parameters={
                "agent_attributes": {"DefaultAttribute": [1,2,3]}})
        model.create_network()
        with pytest.raises(RuntimeError, match="Agent attribute DefaultAttribute must be a "):
            model.initialize_agent_attributes()

    def test_step(self, initialization_model):
        model = initialization_model
        model.step()

        assert model.step_count == 1
        assert model.graph.number_of_nodes() == 100

    def test_run(self, initialization_model):
        model = initialization_model
        model.run()

        assert model.step_count == 5
        assert model.graph.number_of_nodes() == 100

    def test_model_init_savestate(self, initialization_model):
        model = initialization_model
        model.config.checkpoint_period = 1
        model.run()

        assert model.inputs is not None
        assert Path('test_models/initialize_model/graph.bin').exists()
        assert Path('test_models/initialize_model/generator_state.bin').exists()
        assert Path('test_models/initialize_model/process_version.md').exists()

        # Note that the inputs are set at the end of the last step, which is the
        # step before the step target.
        assert model.inputs["step_count"] == 4

    def test_model_init_savestate_not_default(self, initialization_model):
        model = initialization_model
        model.config.checkpoint_period = 2
        model.run()

        # Note that the inputs are set at the end of the last step, which is the
        # step before the step target.
        assert model.inputs["step_count"] == 4

    def test_model_init_restart(self, initialization_model):
        model = initialization_model
        model.config.checkpoint_period = 1
        model.config.step_target = 3 # only run the model till step 3
        model.run()
        assert model.config.step_target == 3
        assert Path('test_models/initialize_model/initialize_model_0.yaml').exists()
        assert Path('test_models/initialize_model/graph.bin').exists()
        assert Path('test_models/initialize_model/generator_state.bin').exists()
        assert Path('test_models/initialize_model/process_version.md').exists()
        expected_generator_state = set(model.inputs["generator_state"].tolist())

        model.initialize_model(restart=True)
        model.config.step_target = 5 # restart the model and run till step 5
        model.run()
        assert model.config.step_target == 5
        assert Path('test_models/initialize_model/initialize_model_0_1.yaml').exists()
        stored_generator_state = set(model.inputs["generator_state"].tolist())

        assert model.inputs is not None
        # Note that the inputs are set at the end of the last step, which is the
        # step before the step target.
        assert model.inputs["step_count"] == 4

        assert stored_generator_state == expected_generator_state

        # The second run also saves its config at the start.
        assert Path('test_models/initialize_model/initialize_model_2.yaml').exists()

    def test_model_milestone(self, initialization_model):
        model = initialization_model
        model.config.milestones = [2]
        model.run()

        assert model.inputs is not None
        milstone_dir = 'test_models/initialize_model/milestone_2'
        assert Path(f'{milstone_dir}/graph.bin').exists()
        assert Path(f'{milstone_dir}/generator_state.bin').exists()
        assert Path(f'{milstone_dir}/process_version.md').exists()
        assert model.inputs["step_count"] == 2

    def test_model_milestone_continue(self, initialization_model):
        model = initialization_model
        model.config.milestones = [1]
        model.config.description = "Initialize network."
        model.config.step_target = 3 # only run the model till step 3
        assert model.step_count == 0 # The step count is the start of the run.
        model.run()
        assert model.config.step_target == 3
        expected_generator_state = set(model.inputs["generator_state"].tolist())

        model.initialize_model(restart=(1,0))
        model.config.step_target = 5 # continue the model and run till step 5
        model.config.description = "Policy 0: just run using default parameters."
        assert model.step_count == 1 # The step count is that of the milestone.
        model.run()
        assert model.config.step_target == 5
        stored_generator_state = set(model.inputs["generator_state"].tolist())

        assert model.inputs is not None
        assert model.inputs["step_count"] == 1
        assert model.step_count == 5
        assert stored_generator_state == expected_generator_state

        # The second run also saves its config at the start.
        assert Path('test_models/initialize_model/initialize_model_1.yaml').exists()

    def test_model_milestone_multiple(self, initialization_model):
        model = initialization_model

        # Run once.
        model.config.milestones = [3]
        assert model.step_count == 0 # The step count is the start of the run.
        model.run()
        assert model.config.step_target == 5
        expected_generator_state = set(model.inputs["generator_state"].tolist())

        # Re-run from the start.
        model.initialize_model(restart=True)
        assert model.step_count == 0 # The step count is the start of the run.
        model.run()
        assert model.config.step_target == 5

        assert model.inputs is not None
        assert model.inputs["step_count"] == 3

        # Note, the first instance of a milestone at step 3 is stored in milestone_3
        milstone_dir = 'test_models/initialize_model/milestone_3'
        assert Path(f'{milstone_dir}/graph.bin').exists()
        assert Path(f'{milstone_dir}/generator_state.bin').exists()
        assert Path(f'{milstone_dir}/process_version.md').exists()

        # Note, the second instance of a milestone at step 3 is stored in milestone_3_1
        milstone_dir = 'test_models/initialize_model/milestone_3_1'
        assert Path(f'{milstone_dir}/graph.bin').exists()
        assert Path(f'{milstone_dir}/generator_state.bin').exists()
        assert Path(f'{milstone_dir}/process_version.md').exists()

        # Continue from the second milestone.
        model.initialize_model(restart=(3,1))
        assert model.step_count == 3 # The step count is that of the milestone.
        model.run()
        assert model.config.step_target == 5
        stored_generator_state = set(model.inputs["generator_state"].tolist())

        assert model.inputs is not None
        assert model.inputs["step_count"] == 3
        assert model.step_count == 5
        assert stored_generator_state == expected_generator_state

        # The third run also saves its config at the start.
        assert Path('test_models/initialize_model/initialize_model_3.yaml').exists()
