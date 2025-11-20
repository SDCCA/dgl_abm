import os
os.environ["DGLBACKEND"] = "pytorch"

import dgl_abm


model = dgl_abm.Model(experiment_identifier=f'default_demo')
model.set_model_parameters()
model.initialize_model()
model.run()

