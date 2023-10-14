#!/usr/bin/env python

import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# general poisson
def poisson2d(n):
  n2 = n*n
  row = np.zeros(5*n2-4*n)
  col = np.zeros(5*n2-4*n)
  dat = np.zeros(5*n2-4*n)
  
  iz = -1
  for i in range(n):
    for j in range(n):
      k = i + n*j
      iz = iz + 1
      row[iz] = k
      col[iz] = k
      dat[iz] = -4
      
      if i > 0:
        iz = iz + 1
        row[iz] = k
        col[iz] = k-1
        dat[iz] = 1
      if i < n-1:
        iz = iz + 1
        row[iz] = k
        col[iz] = k+1
        dat[iz] = 1
      if j > 0:
        iz = iz + 1
        row[iz] = k
        col[iz] = k-n
        dat[iz] = 1
      if j < n-1:
        iz = iz + 1
        row[iz] = k
        col[iz] = k+n
        dat[iz] = 1
              
  L = coo_matrix((dat, (row, col)), shape=(n2, n2))   
  return L

# poisson with mask
def poisson_2d_mask(mask,*args, **kwargs):

  # bct: boundary condition type: 1=dirichlet, 0=neuman
  bct = kwargs.get('bct',1)
  
  si_y,si_x = mask.shape
  n = si_x
  n2 = n*n

  # for now, no metric term
  idx2 = 1.0
  idy2 = 1.0

  mat_li  = np.zeros(5*n2,dtype='int')
  mat_co  = np.zeros(5*n2,dtype='int')
  mat_val = np.zeros(5*n2)
                  
  nze = 0
  flag_singular = 1
  for ny in range(0,si_y) :
    for nx in range(0,si_x):

      if mask[ny,nx] == 1:
        if flag_singular and bct == 0 :
          mat_li[nze] = ny*si_x + nx; mat_co[nze] = ny*si_x + nx; mat_val[nze] = 1; 
          nze = nze + 1;
          singular_flag = 0;
        else:
          mat_li[nze+0] = ny*si_x + nx; mat_co[nze+0] = (ny  )*si_x + nx  ; mat_val[nze+0] = -2*idx2 -2*idy2; 
          mat_li[nze+1] = ny*si_x + nx; mat_co[nze+1] = (ny  )*si_x + nx+1; mat_val[nze+1] = idx2;
          mat_li[nze+2] = ny*si_x + nx; mat_co[nze+2] = (ny  )*si_x + nx-1; mat_val[nze+2] = idx2;
          mat_li[nze+3] = ny*si_x + nx; mat_co[nze+3] = (ny+1)*si_x + nx  ; mat_val[nze+3] = idy2;
          mat_li[nze+4] = ny*si_x + nx; mat_co[nze+4] = (ny-1)*si_x + nx  ; mat_val[nze+4] = idy2;
          nze = nze + 5;
      elif mask[ny,nx] == 2:
          mat_li[nze+0] = ny*si_x + nx; mat_co[nze+0] = (ny  )*si_x + nx  ; mat_val[nze+0] = -2*idx2 -idy2 - bct*idy2; 
          mat_li[nze+1] = ny*si_x + nx; mat_co[nze+1] = (ny  )*si_x + nx+1; mat_val[nze+1] = idx2;
          mat_li[nze+2] = ny*si_x + nx; mat_co[nze+2] = (ny  )*si_x + nx-1; mat_val[nze+2] = idx2;
          mat_li[nze+3] = ny*si_x + nx; mat_co[nze+3] = (ny+1)*si_x + nx  ; mat_val[nze+3] = idy2;
          nze = nze + 4;
      elif mask[ny,nx] == 3:
          mat_li[nze+0] = ny*si_x + nx; mat_co[nze+0] = (ny  )*si_x + nx  ; mat_val[nze+0] = -2*idx2 -idy2 - bct*idy2; 
          mat_li[nze+1] = ny*si_x + nx; mat_co[nze+1] = (ny  )*si_x + nx+1; mat_val[nze+1] = idx2;
          mat_li[nze+2] = ny*si_x + nx; mat_co[nze+2] = (ny  )*si_x + nx-1; mat_val[nze+2] = idx2;
          mat_li[nze+3] = ny*si_x + nx; mat_co[nze+3] = (ny-1)*si_x + nx  ; mat_val[nze+3] = idy2;
          nze = nze + 4;
      elif mask[ny,nx] == 4:
          mat_li[nze+0] = ny*si_x + nx; mat_co[nze+0] = (ny  )*si_x + nx  ; mat_val[nze+0] = -idx2 -2*idy2 - bct*idx2; 
          mat_li[nze+1] = ny*si_x + nx; mat_co[nze+1] = (ny  )*si_x + nx+1; mat_val[nze+1] = idx2;
          mat_li[nze+2] = ny*si_x + nx; mat_co[nze+2] = (ny+1)*si_x + nx  ; mat_val[nze+2] = idy2;
          mat_li[nze+3] = ny*si_x + nx; mat_co[nze+3] = (ny-1)*si_x + nx  ; mat_val[nze+3] = idy2;
          nze = nze + 4;
      elif mask[ny,nx] == 5:
          mat_li[nze+0] = ny*si_x + nx; mat_co[nze+0] = (ny  )*si_x + nx  ; mat_val[nze+0] = -idx2 -2*idy2 - bct*idx2; 
          mat_li[nze+1] = ny*si_x + nx; mat_co[nze+1] = (ny  )*si_x + nx-1; mat_val[nze+1] = idx2;
          mat_li[nze+2] = ny*si_x + nx; mat_co[nze+2] = (ny+1)*si_x + nx  ; mat_val[nze+2] = idy2;
          mat_li[nze+3] = ny*si_x + nx; mat_co[nze+3] = (ny-1)*si_x + nx  ; mat_val[nze+3] = idy2;
          nze = nze + 4;
      elif mask[ny,nx] == 6:
          mat_li[nze+0] = ny*si_x + nx; mat_co[nze+0] = (ny  )*si_x + nx  ; mat_val[nze+0] = -idx2 -idy2 - bct*idx2- bct*idy2; 
          mat_li[nze+1] = ny*si_x + nx; mat_co[nze+1] = (ny  )*si_x + nx+1; mat_val[nze+1] = idx2;
          mat_li[nze+2] = ny*si_x + nx; mat_co[nze+2] = (ny+1)*si_x + nx  ; mat_val[nze+2] = idy2;
          nze = nze + 3;
      elif mask[ny,nx] == 7:
          mat_li[nze+0] = ny*si_x + nx; mat_co[nze+0] = (ny  )*si_x + nx  ; mat_val[nze+0] = -idx2 -idy2- bct*idx2- bct*idy2; 
          mat_li[nze+1] = ny*si_x + nx; mat_co[nze+1] = (ny  )*si_x + nx-1; mat_val[nze+1] = idx2;
          mat_li[nze+2] = ny*si_x + nx; mat_co[nze+2] = (ny+1)*si_x + nx  ; mat_val[nze+2] = idy2;
          nze = nze + 3;
      elif mask[ny,nx] == 8:
          mat_li[nze+0] = ny*si_x + nx; mat_co[nze+0] = (ny  )*si_x + nx  ; mat_val[nze+0] = -idx2 -idy2- bct*idx2- bct*idy2; 
          mat_li[nze+1] = ny*si_x + nx; mat_co[nze+1] = (ny  )*si_x + nx+1; mat_val[nze+1] = idx2;
          mat_li[nze+2] = ny*si_x + nx; mat_co[nze+2] = (ny-1)*si_x + nx  ; mat_val[nze+2] = idy2;
          nze = nze + 3;
      elif mask[ny,nx] == 9:
          mat_li[nze+0] = ny*si_x + nx; mat_co[nze+0] = (ny  )*si_x + nx  ; mat_val[nze+0] = -idx2 -idy2- bct*idx2- bct*idy2; 
          mat_li[nze+1] = ny*si_x + nx; mat_co[nze+1] = (ny  )*si_x + nx-1; mat_val[nze+1] = idx2;
          mat_li[nze+2] = ny*si_x + nx; mat_co[nze+2] = (ny-1)*si_x + nx  ; mat_val[nze+2] = idy2;
          nze = nze + 3;
      elif mask[ny,nx] == 10:
          mat_li[nze+0] = ny*si_x + nx; mat_co[nze+0] = (ny  )*si_x + nx  ; mat_val[nze+0] = -2*idx2 - 2*bct*idy2; 
          mat_li[nze+1] = ny*si_x + nx; mat_co[nze+1] = (ny  )*si_x + nx+1; mat_val[nze+1] = idx2;
          mat_li[nze+2] = ny*si_x + nx; mat_co[nze+2] = (ny  )*si_x + nx-1; mat_val[nze+2] = idx2;
          nze = nze + 3;
      elif mask[ny,nx] == 11:
          mat_li[nze+0] = ny*si_x + nx; mat_co[nze+0] = (ny  )*si_x + nx  ; mat_val[nze+0] = -2*idy2 - 2*bct*idx2; 
          mat_li[nze+1] = ny*si_x + nx; mat_co[nze+1] = (ny+1)*si_x + nx  ; mat_val[nze+1] = idy2;
          mat_li[nze+2] = ny*si_x + nx; mat_co[nze+2] = (ny-1)*si_x + nx  ; mat_val[nze+2] = idy2;
          nze = nze + 3;
      elif mask[ny,nx] == 12:
          mat_li[nze+0] = ny*si_x + nx; mat_co[nze+0] = (ny  )*si_x + nx  ; mat_val[nze+0] = -idx2 -bct*(idx2 + 2*idy2)
          mat_li[nze+1] = ny*si_x + nx; mat_co[nze+1] = (ny  )*si_x + nx-1; mat_val[nze+1] = idx2;
          nze = nze + 2;
      elif mask[ny,nx] == 13:
          mat_li[nze+0] = ny*si_x + nx; mat_co[nze+0] = (ny  )*si_x + nx  ; mat_val[nze+0] = -idx2-bct*(idx2 + 2*idy2)
          mat_li[nze+1] = ny*si_x + nx; mat_co[nze+1] = (ny  )*si_x + nx+1; mat_val[nze+1] = idx2;
          nze = nze + 2;
      elif mask[ny,nx] == 14:
          mat_li[nze+0] = ny*si_x + nx; mat_co[nze+0] = (ny  )*si_x + nx  ; mat_val[nze+0] = -idy2-bct*(2*idx2 + idy2); 
          mat_li[nze+1] = ny*si_x + nx; mat_co[nze+1] = (ny-1)*si_x + nx  ; mat_val[nze+1] = idy2;
          nze = nze + 2;
      elif mask[ny,nx] == 15:
          mat_li[nze+0] = ny*si_x + nx; mat_co[nze+0] = (ny  )*si_x + nx  ; mat_val[nze+0] = -idy2-bct*(2*idx2 + idy2); 
          mat_li[nze+1] = ny*si_x + nx; mat_co[nze+1] = (ny+1)*si_x + nx  ; mat_val[nze+1] = idy2;
          nze = nze + 2;
        
      elif mask[ny,nx] == 0 : #% hole
        mat_li[nze] = ny*si_x + nx; mat_co[nze] = ny*si_x + nx; mat_val[nze] = 1; 
        nze = nze + 1;

  L = coo_matrix((mat_val, (mat_li, mat_co)), shape=(n2, n2))   
  return L




def sol(rhs,*arg, **kwargs):
  """
  Solve a linear System
  
  :param  rhs: the right hand side of the linear system (2d np array)
  :param  mat: The linear operator (optional: default: poisson)
  
  :returns: The solution of the 2d system (same shape as rhs)
  
  :raises: TODO
  """
  
  si_a = rhs.shape
  
  psi = np.array(rhs).flatten()

  n = np.int(np.sqrt(len(psi)))

  # get opt. args
#  L = kwargs.get('mat')
  L = kwargs.get('mat', poisson2d(n))
    
  x = spsolve(L.tocsr(),psi)

  xsol = x.reshape(si_a)
  return xsol
