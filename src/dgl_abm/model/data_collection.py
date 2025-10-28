"""This module contains functions enabling collection of data from agents and edges.

Function(s):
- data_collection: Collects data from agents and edges
- _node_property_collector: Collects node properties from the agent graph
- _edge_property_collector: Collects edge properties from the agent graph
- _check_nprop_in_graph: Checks if node property is in agent graph
- _check_eprop_in_graph: Checks if edge property is in agent graph
"""

import os
from pathlib import Path
import xarray as xr
from dgl import DGLGraph


def data_collection(
    agent_graph: DGLGraph, timestep: int, parameters: dict, npath: str = "./agent_data", epath: str = "./edge_data"
) -> None:
    """Collect data from agents and edges.

    Args:
        agent_graph (DGLGraph): All agent node and edge data
        timestep (int): current timestep to name folder for edge properties
        parameters (dict): model parameters
            data_format (str): storage format
            mode (str): zarr write mode
        npath (str): path to store node data
        epath (str): path to store edge data with one file for each timestep

    Returns:
        None

    """
    ndata = parameters.get("ndata")
    edata = parameters.get("edata")
    data_format = parameters.get("format", "xarray")
    mode = parameters.get("mode", "w-")

    ndata, initial_only = _parse_ndata(ndata, timestep, agent_graph)
    edata = _parse_edata(edata, agent_graph)
    if ndata is not None:
        if timestep == 0 and locals().get("initial_only", []) != []:
            initialpath = npath.split(".")[0] + "_initial.zarr"
            _node_property_collector(agent_graph, initialpath, initial_only, timestep, data_format, mode)
        _node_property_collector(agent_graph, npath, ndata, timestep, parameters)
    if edata is not None:
        _edge_property_collector(agent_graph, epath, edata, timestep, parameters)


def _parse_ndata(ndata: list | None, timestep: int, agent_graph: DGLGraph) -> tuple[list | None, list]:
    """Parse node data collection specification.

    Args:
        ndata (list | None): format specifies node data properties to be stored
            [str] - specifies node properties to be stored at every time step
            ['all'] - implies all node properties will be saved at every time step
            ['all_except', [str]] - specifies that all but the listed properties
                will be saved
            ['initial_only', [str]] - specifies properties saved at timestep 0 only
            Notes:
            [str] and ['all_except', [list]] can be used with [initial_only, [list]]
            formatted as [[specification list],[specification list]].
            ['all'] should not be used together with any other specification.
        timestep (int): current timestep to name folder for edge properties
        agent_graph (DGLGraph): All agent node and edge data

    Returns:
        tuple: list of node data properties to be saved, list of node properties to be saved
            at initial timestep only
    """
    initial_only = []
    if ndata == ["all"]:
        return list(agent_graph.node_attr_schemes().keys()), initial_only
    if ndata[0] == "all_except":
        return list(agent_graph.node_attr_schemes().keys() - ndata[1]), initial_only
    if sum(1 for item in ndata if isinstance(item, list)) > 1:
        ndata_list = ndata
        for specification in ndata_list:
            if specification == ["all"]:
                all_message = 'Use of "all" is not compatible with multiple datacollection specification lists.'
                raise ValueError(all_message)
            if specification[0] == "all_except":
                ndata = list(agent_graph.node_attr_schemes().keys() - specification[1])
            if specification[0] == "initial_only" and timestep == 0:
                initial_only = specification[1]
        return ndata, initial_only
    message = f"Invalid node data collection specification: {ndata}"
    raise ValueError(message)


def _parse_edata(edata: list | None, agent_graph: DGLGraph) -> list | None:
    """Parse edge data collection specification.

    Args:
        edata (list): edge data properties to be stored
            ['all'] implies all edge properties will be saved
        agent_graph (DGLGraph): All agent node and edge data
    Returns:
        list | None: list of edge data properties to be saved
    """
    if edata == ["all"]:
        edata = list(agent_graph.edge_attr_schemes().keys())
    return edata


def _node_property_collector(agent_graph: DGLGraph, npath: str, ndata: list, timestep: int, parameters: dict) -> None:
    """Collect node properties from the agent graph.

    Args:
        agent_graph (DGLGraph): All agent node and edge data
        npath (str): path to store node data
        ndata (list): node data properties to be stored
        timestep (int): current timestep
        parameters (dict): model parameters
            data_format (str): storage format
            mode (str): zarr write mode

    Returns:
        None

    Effects:
        Creates a folder at npath with a zarr file for each node property
    """
    data_format = parameters.get("format", "xarray")
    mode = parameters.get("mode", "w-")

    if os.environ["DGLBACKEND"] == "pytorch":
        if data_format == "xarray":
            agent_data_instance = xr.Dataset()
            for prop in ndata:
                _check_nprop_in_graph(agent_graph, prop)
                agent_data_instance = agent_data_instance.assign(
                    prop=(["n_agents", "n_time"], agent_graph.ndata[prop][:, None].cpu().numpy())
                )
                agent_data_instance = agent_data_instance.rename(name_dict={"prop": prop})
            if timestep == 0:
                agent_data_instance.to_zarr(npath, mode=mode)
            else:
                agent_data_instance.to_zarr(npath, append_dim="n_time")
        else:
            xarray_message = "Only 'xarray' format currently available"
            raise NotImplementedError(xarray_message)
    else:
        message = "Data collection is currently only implemented for pytorch backend."
        raise NotImplementedError(message)


def _edge_property_collector(agent_graph: DGLGraph, epath: str, edata: list, timestep: int, parameters: dict) -> None:
    """Collect edge properties from the agent graph.

    Args:
        agent_graph (DGLGraph): All agent node and edge data
        epath (str): path to store edge data
        edata (list): edge data properties to be stored
        timestep (int): current timestep
        parameters (dict): model parameters
            data_format (str): storage format
            mode (str): zarr write mode

    Returns:
        None

    Effects:
        Creates a folder at epath with a zarr file for each edge property
    """
    data_format = parameters.get("format", "xarray")
    mode = parameters.get("mode", "w-")

    if os.environ["DGLBACKEND"] == "pytorch":
        if data_format == "xarray":
            edge_data_instance = xr.Dataset(
                coords={
                    "source": (["n_edges"], agent_graph.edges()[0].cpu()),
                    "dest": (["n_edges"], agent_graph.edges()[1].cpu()),
                }
            )
            for prop in edata:
                _check_eprop_in_graph(agent_graph, prop)
                edge_data_instance = edge_data_instance.assign(
                    property=(["n_edges", "time"], agent_graph.edata[prop][:, None].cpu().numpy())
                )

                edge_data_instance = edge_data_instance.rename_vars(name_dict={"property": prop})
            edge_data_instance.to_zarr(Path(epath) / (str(timestep) + ".zarr"), mode=mode)
        else:
            xarray_message = "Only 'xarray' mode currently available"
            raise NotImplementedError(xarray_message)
    else:
        message = "Data collection is currently only implemented for pytorch backend."
        raise NotImplementedError(message)


def _check_nprop_in_graph(agent_graph: DGLGraph, prop: str) -> None:
    """Check if node property is in agent graph.

    Args:
        agent_graph (DGLGraph): All agent node and edge data
        prop (str): node property to check

    Returns:
        None

    Raises:
        ValueError: If node property is not in agent graph
    """
    if prop not in agent_graph.node_attr_schemes():
        property_message = f"{prop} is not a node property.Please choose from {agent_graph.node_attr_schemes().keys()}"
        raise ValueError(property_message)


def _check_eprop_in_graph(agent_graph: DGLGraph, prop: str) -> None:
    """Check if edge property is in agent graph.

    Args:
        agent_graph (DGLGraph): All agent node and edge data
        prop (str): edge property to check

    Returns:
        None

    Raises:
        ValueError: If edge property is not in agent graph
    """
    if prop not in agent_graph.edge_attr_schemes():
        property_message = f"{prop} is not an edge property.Please choose from {agent_graph.edge_attr_schemes().keys()}"
        raise ValueError(property_message)
