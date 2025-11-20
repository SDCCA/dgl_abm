"""Demo script to run model on CPU with default configuration."""

import os

os.environ["DGLBACKEND"] = "pytorch"

import dgl_abm

model = dgl_abm.Model(experiment_identifier="default_demo")
model.set_model_parameters()
model.initialize_model()
model.run()
