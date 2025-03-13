"""Local attachment tensor."""
import dgl
import torch
from dgl.sparse import spmatrix

from dgl_ptm.util import matrix_utils

# TODO: check readability variables.

# TODO: confirm that the behavior for sum(weight)=0 agents is the same as for plain
#  local attachment
def local_attachment_tensor(graph,n_FoF_links,edge_prop=None,p_attach=1.):#noqa N806
    """Update graph with local attachment based on edge property.

    Args:
        graph (DGLGraph): All agent node and edge data
        n_FoF_links (int): Number of new links to attempt
        edge_prop (str): edge property to use for local attachment
        p_attach (float): probability of attaching a new edge
    
    Returns:
        None    
    """
    adj_matrix = adjacency_matrix_with_edge_prop(graph,eprop=edge_prop)
    norm_prop = adj_matrix.val/adj_matrix.val.sum()

    #sample n_FoF_links from the entire normalized edge property graph weighted by 
    # edge weight
    selected_links = norm_prop.multinomial(num_samples=n_FoF_links,#noqa N806
                                           replacement=False)
    selected_links_matrix = matrix_utils.apply_mask_to_sparse_matrix(adj_matrix, 
                                                                     selected_links)
    FoF_field_matrix, FoF_field_matrix_norm_eprop = neighbour_field_matrix(#noqa N806
                                                                selected_links_matrix,
                                                                adj_matrix)
    new_FoF = FoF_field_matrix.val > 0#noqa N806
    new_FoF_norm_eprop = matrix_utils.apply_mask_to_sparse_matrix(#noqa N806
                                                FoF_field_matrix_norm_eprop, new_FoF)
    if torch.count_nonzero(new_FoF) <= n_FoF_links:
        probe_p_attach = torch.rand(new_FoF_norm_eprop.val.shape[0]) 
        to_link = probe_p_attach < p_attach
    else:
        new_FoF_renorm = new_FoF_norm_eprop.val/new_FoF_norm_eprop.val.sum()#noqa N806
        selected_FoF = new_FoF_renorm.flatten().multinomial(#noqa N806
                                                            num_samples=n_FoF_links,
                                                            replacement=False)
        #probe_p_attach = torch.rand(new_FoF_norm_eprop.val.shape[0])
        probe_p_attach = torch.rand(n_FoF_links)
        #to_link = torch.logical_and((probe_p_attach > p_attach),selected_FoF)
        probe_selected = probe_p_attach < p_attach
        to_link = selected_FoF[probe_selected]
    FoF_to_link_matrix = matrix_utils.apply_mask_to_sparse_matrix(#noqa N806
                                                                  new_FoF_norm_eprop, 
                                                                  to_link)
    graph.add_edges(FoF_to_link_matrix.row,FoF_to_link_matrix.col)
    graph.add_edges(FoF_to_link_matrix.col,FoF_to_link_matrix.row)

def adjacency_matrix_with_edge_prop(graph,etype=None, eprop=None):
    """Create adjacency matrix with edge property.
    
    Args:
        graph (DGLGraph): All agent node and edge data
        etype (str): edge type
        eprop (str): edge property
    
    Returns:
        spmatrix: adjacency matrix with edge property
    """
    etype = graph.to_canonical_etype(etype)
    indices = torch.stack(graph.all_edges(etype=etype))
    shape = (graph.num_nodes(etype[0]),graph.number_of_nodes(etype[2]))
    if eprop is not None:
        val =graph.edges[etype].data[eprop].flatten()
    else:
        val=None
    return spmatrix(
        indices,
        val=val,
        shape=shape,
    )

def construct_neighbour_field_tensors(selm,adjm):
    """Create infrastructure to support neighbour field matrix.

    This function creates the row tensor, column tensor, link tensor, and value tensor
    needed to construct the neighbour field matrix/tensor in sparse representation. 
    This is done by creating a list of the tensor representations for each element i,j 
    of the matrix of selected edges by adding row i and row j of the adjacency
    matrix, and storing the result in row i, for each element an insttance of row i 
    as well as the entry i,i is removed. The list is subsequently concatenated to obtain
    a single tensor representation.
    
    Notes:
        In addition to the link tensor, which denotes the link status with and integer, 
        the function also returns a value tensor with entries corresponding to the 
        weight of edge j,k. 
        Neighbours with no direct connecton appear as > 0 values.
        The resulting tensors can/will contain significant numbers of multiple 
        assigments for an element i,k. This is addressed in subsequent processing, 
        which also handles the combination of weights for eligible connections arising 
        from multiple possible links.
    
    Input:
        selm (dgl.sparse.spmatrix): matrix of selected edges in sparse format
        adjm (dgl.sparse.spmatrix): adjacency matrix with edge weights as sparse values
    
    Output:
        rowtensor (torch.tensor): row tensor of the neighbour field matrix, compatible
            with sparse format
        coltensor (torch.tensor): column tensor of the neighbour field matrix, 
            compatible with sparse format
        ltensor (torch.tensor): link tensor of the neighbour field matrix, compatible 
            with sparse format
        valtensor (torch.tensor): tensor with edge weights of link jk
    """
    rtl = list()
    ctl = list()
    ltl = list()
    vtl = list()
    for i in range(selm.row.shape[0]):
        src = selm.row[i]
        dst = selm.col[i]
        srcten = torch.tensor([src])
        wsrcten = torch.tensor([0.])
        lsrcten = torch.tensor([-1])
        src_nids = adjm.col[adjm.row==src]
        wsrc_nids = adjm.val[adjm.row==src]
        lsrc_nids = torch.ones_like(src_nids,dtype=int)*(-1)
        dst_nids = adjm.col[adjm.row==dst]
        wdst_nids = adjm.val[adjm.row==dst]
        ldst_nids = torch.ones_like(dst_nids,dtype=int)
        rv_src_nids = torch.ones_like(src_nids)*src
        rv_dst_nids = torch.ones_like(dst_nids)*src
        rv_nids = torch.cat((rv_src_nids,rv_dst_nids,srcten))
        cv_nids = torch.cat((src_nids,dst_nids,srcten))
        lv_nids = torch.cat((lsrc_nids,ldst_nids,lsrcten))
        wv_nids = torch.cat((wsrc_nids,wdst_nids,wsrcten))
        rtl.append(rv_nids)
        ctl.append(cv_nids)
        ltl.append(lv_nids)
        vtl.append(wv_nids)
    rowtensor = torch.cat(rtl)
    coltensor = torch.cat(ctl)
    ltensor   = torch.cat(ltl)
    valtensor = torch.cat(vtl)
    return (rowtensor, coltensor, ltensor, valtensor)

def neighbour_field_matrix(selm, adjm):
    """Create neighbour field matrix.

    Args:
        selm (dgl.sparse.spmatrix): matrix of selected edges in sparse format
        adjm (dgl.sparse.spmatrix): adjacency matrix with edge weights as sparse values

    Returns:
        nfmc (dgl.sparse.spmatrix): link neighbour field matrix
        nfmvc (dgl.sparse.spmatrix): value neighbour field matrix
    """
    nf = construct_neighbour_field_tensors(selm,adjm)
     #stack indices into tensor for matrix construction
    indices= torch.stack((nf[0],nf[1]))
    # create link neighbour fields
    nfm = dgl.sparse.spmatrix(indices,nf[2],shape=selm.shape)
    # create value neighbour fields
    nfmv = dgl.sparse.spmatrix(indices,nf[3],shape=selm.shape)
    nfmc = nfm.coalesce()
    nfmvc = nfmv.coalesce()
    return nfmc, nfmvc 