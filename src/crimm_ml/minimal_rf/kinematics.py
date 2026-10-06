import numpy as np
import torch
from .chemical import INIT_CRDS
from .util import generate_Cbeta

PARAMS = {
    "DMIN"    : 2.0,
    "DMAX"    : 20.0,
    "DBINS"   : 36,
    "ABINS"   : 36,
}

# ============================================================
def get_pair_dist(a, b):
    """calculate pair distances between two sets of points
    
    Parameters
    ----------
    a,b : pytorch tensors of shape [batch,nres,3]
          store Cartesian coordinates of two sets of atoms
    Returns
    -------
    dist : pytorch tensor of shape [batch,nres,nres]
           stores paitwise distances between atoms in a and b
    """

    dist = torch.cdist(a, b, p=2)
    return dist

# ============================================================
def get_ang(a, b, c):
    """calculate planar angles for all consecutive triples (a[i],b[i],c[i])
    from Cartesian coordinates of three sets of atoms a,b,c 

    Parameters
    ----------
    a,b,c : pytorch tensors of shape [batch,nres,3]
            store Cartesian coordinates of three sets of atoms
    Returns
    -------
    ang : pytorch tensor of shape [batch,nres]
          stores resulting planar angles
    """
    v = a - b
    w = c - b
    v /= torch.norm(v, dim=-1, keepdim=True)
    w /= torch.norm(w, dim=-1, keepdim=True)
    vw = torch.sum(v*w, dim=-1)

    return torch.acos(vw)

# ============================================================
def get_dih(a, b, c, d):
    """calculate dihedral angles for all consecutive quadruples (a[i],b[i],c[i],d[i])
    given Cartesian coordinates of four sets of atoms a,b,c,d

    Parameters
    ----------
    a,b,c,d : pytorch tensors or numpy array of shape [batch,nres,3]
              store Cartesian coordinates of four sets of atoms
    Returns
    -------
    dih : pytorch tensor or numpy array of shape [batch,nres]
          stores resulting dihedrals
    """
    convert_to_torch = lambda *arrays: [torch.from_numpy(arr) for arr in arrays]
    output_np=False
    if isinstance(a, np.ndarray):
        output_np=True
        a,b,c,d = convert_to_torch(a,b,c,d)
    b0 = a - b
    b1 = c - b
    b2 = d - c

    b1 /= torch.norm(b1, dim=-1, keepdim=True)

    v = b0 - torch.sum(b0*b1, dim=-1, keepdim=True)*b1
    w = b2 - torch.sum(b2*b1, dim=-1, keepdim=True)*b1

    x = torch.sum(v*w, dim=-1)
    y = torch.sum(torch.cross(b1,v,dim=-1)*w, dim=-1)
    output = torch.atan2(y, x)
    if output_np:
        return output.numpy()
    return output

# ============================================================
def xyz_to_c6d(xyz, params=PARAMS):
    """convert cartesian coordinates into 2d distance 
    and orientation maps
    
    Parameters
    ----------
    xyz : pytorch tensor of shape [batch,nres,3,3]
          stores Cartesian coordinates of backbone N,Ca,C atoms
    Returns
    -------
    c6d : pytorch tensor of shape [batch,nres,nres,4]
          stores stacked dist,omega,theta,phi 2D maps 
    """
    
    batch = xyz.shape[0]
    nres = xyz.shape[1]

    # three anchor atoms
    N  = xyz[:,:,0]
    Ca = xyz[:,:,1]
    C  = xyz[:,:,2]
    Cb = generate_Cbeta(N, Ca, C)

    # 6d coordinates order: (dist,omega,theta,phi)
    c6d = torch.zeros([batch,nres,nres,4],dtype=xyz.dtype,device=xyz.device)

    dist = get_pair_dist(Cb,Cb)
    dist[torch.isnan(dist)] = 999.9
    c6d[...,0] = dist + 999.9*torch.eye(nres,device=xyz.device)[None,...]
    b,i,j = torch.where(c6d[...,0]<params['DMAX'])

    c6d[b,i,j,torch.full_like(b,1)] = get_dih(Ca[b,i], Cb[b,i], Cb[b,j], Ca[b,j])
    c6d[b,i,j,torch.full_like(b,2)] = get_dih(N[b,i], Ca[b,i], Cb[b,i], Cb[b,j])
    c6d[b,i,j,torch.full_like(b,3)] = get_ang(Ca[b,i], Cb[b,i], Cb[b,j])

    # fix long-range distances
    c6d[...,0][c6d[...,0]>=params['DMAX']] = 999.9
    
    mask = torch.zeros((batch, nres,nres), dtype=xyz.dtype, device=xyz.device)
    mask[b,i,j] = 1.0
    return c6d, mask
    
def xyz_to_t2d(xyz_t, params=PARAMS):
    """convert template cartesian coordinates into 2d distance 
    and orientation maps
    
    Parameters
    ----------
    xyz_t : pytorch tensor of shape [batch,templ,nres,3,3]
            stores Cartesian coordinates of template backbone N,Ca,C atoms

    Returns
    -------
    t2d : pytorch tensor of shape [batch,nres,nres,37+6+3]
          stores stacked dist,omega,theta,phi 2D maps 
    """
    B, T, L = xyz_t.shape[:3]
    c6d, mask = xyz_to_c6d(xyz_t[:,:,:,:3].view(B*T,L,3,3), params=params)
    c6d = c6d.view(B, T, L, L, 4)
    mask = mask.view(B, T, L, L, 1)
    #
    # dist to one-hot encoded
    dist = dist_to_onehot(c6d[...,0], params)
    orien = torch.cat((torch.sin(c6d[...,1:]), torch.cos(c6d[...,1:])), dim=-1)*mask # (B, T, L, L, 6)
    #
    mask = ~torch.isnan(c6d[:,:,:,:,0]) # (B, T, L, L)
    t2d = torch.cat((dist, orien, mask.unsqueeze(-1)), dim=-1)
    t2d[torch.isnan(t2d)] = 0.0
    return t2d



# ============================================================
def dist_to_onehot(dist, params=PARAMS):
    dist[torch.isnan(dist)] = 999.9
    dstep = (params['DMAX'] - params['DMIN']) / params['DBINS']
    dbins = torch.linspace(params['DMIN']+dstep, params['DMAX'], params['DBINS'],dtype=dist.dtype,device=dist.device)
    db = torch.bucketize(dist.contiguous(),dbins).long()
    dist = torch.nn.functional.one_hot(db, num_classes=params['DBINS']+1).float()
    return dist



# ============================================================


# ============================================================

def get_init_xyz(xyz_t):
    # input: xyz_t (B, T, L, 14, 3)
    # ouput: xyz (B, T, L, 14, 3)
    B, T, L = xyz_t.shape[:3]
    init = torch.from_numpy(INIT_CRDS).to(xyz_t.device).reshape(1,1,1,27,3).repeat(B,T,L,1,1)
    if torch.isnan(xyz_t).all():
        return init

    mask = torch.isnan(xyz_t[:,:,:,:3]).any(dim=-1).any(dim=-1) # (B, T, L)
    #
    center_CA = ((~mask[:,:,:,None]) * torch.nan_to_num(xyz_t[:,:,:,1,:])).sum(dim=2) / ((~mask[:,:,:,None]).sum(dim=2)+1e-4) # (B, T, 3)
    xyz_t = xyz_t - center_CA.view(B,T,1,1,3)
    #
    idx_s = list()
    for i_b in range(B):
        for i_T in range(T):
            if mask[i_b, i_T].all():
                continue
            exist_in_templ = torch.where(~mask[i_b, i_T])[0] # (L_sub)
            seqmap = (torch.arange(L, device=xyz_t.device)[:,None] - exist_in_templ[None,:]).abs() # (L, L_sub)
            seqmap = torch.argmin(seqmap, dim=-1) # (L)
            idx = torch.gather(exist_in_templ, -1, seqmap) # (L)
            offset_CA = torch.gather(xyz_t[i_b, i_T, :, 1, :], 0, idx.reshape(L,1).expand(-1,3))
            init[i_b,i_T] += offset_CA.reshape(L,1,3)
    #
    xyz = torch.where(mask.view(B, T, L, 1, 1), init, xyz_t)
    return xyz
