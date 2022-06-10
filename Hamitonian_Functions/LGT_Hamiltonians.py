import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.base import isspmatrix
# ====================================================================
from .QMB_Operations.Mappings_1D_2D import zig_zag,inverse_zig_zag
from .QMB_Operations.Mappings_1D_2D import coords
from .QMB_Operations.QMB_Operators import local_op
from .QMB_Operations.QMB_Operators import two_body_op
from .QMB_Operations.QMB_Operators import four_body_operator
from .QMB_Operations.Simple_Checks import pause
# ====================================================================
from .LGT_Objects import Local_Operator
from .LGT_Objects import Two_Body_Correlator
from .LGT_Objects import Plaquette
from .LGT_Objects import Rishon_Modes
# ====================================================================
def Local_Hamiltonian(nx,ny,Operator,debug,staggered=False):
    # CHECK ON TYPES
    if not np.isscalar(nx) and not isinstance(nx, int):
        raise TypeError(f'nx must be a SCALAR & INTEGER, not a {type(nx)}')
    if not np.isscalar(ny) and not isinstance(ny, int):
        raise TypeError(f'ny must be a SCALAR & INTEGER, not a {type(ny)}')
    if not isinstance(Operator,Local_Operator):
       raise TypeError(f'Operator must be an  instance of <class: Local_Operator>, not a {type(Operator)}')
    if not isinstance(debug, bool):
        raise TypeError(f'debug must be a BOOL, not a {type(debug)}')
    if not isinstance(staggered, bool):
        raise TypeError(f'staggered must be a BOOL, not a {type(staggered)}')
    # PRINT THE PHRASE: IT USUALLY REFERS TO THE NAME OF THE HAMILTONIAN
    if staggered:
        pause(f'STAGGERED {Operator.Op_name} HAMILTONIAN',debug)
    else:
        pause(f'{Operator.Op_name} HAMILTONIAN',debug)
    # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
    n=nx*ny
    # LOCAL HAMILTONIAN
    H_Local=0
    for ii in range(n):
        if staggered:
            # Compute the corresponding (x,y) coords
            x,y=zig_zag(nx,ny,ii)
            # Compute the staggered factor depending on the site
            staggered_factor=(-1)**(x+y)
            H_Local=H_Local+staggered_factor*local_op(Operator.Op,Operator.ID,ii+1,n)
        else:
            H_Local=H_Local+local_op(Operator.Op,Operator.ID,ii+1,n)
    return H_Local

# ====================================================================
def Two_Body_Hamiltonian(nx,ny,Corr,phrase,debug,periodicity=False,staggered=False,add_dagger=False,coeffs=None):
    # CHECK ON TYPES
    if not np.isscalar(nx) and not isinstance(nx, int):
        raise TypeError(f'nx must be a SCALAR & INTEGER, not a {type(nx)}')
    if not np.isscalar(ny) and not isinstance(ny, int):
        raise TypeError(f'ny must be a SCALAR & INTEGER, not a {type(ny)}')
    if not isinstance(Corr,Two_Body_Correlator):
        raise TypeError(f'Corr must be a CLASS Two_Body_Correlator, not a {type(Corr)}')
    if not isinstance(phrase, str):
        raise TypeError(f'phrase must be a STRING, not a {type(phrase)}')
    if not isinstance(debug, bool):
        raise TypeError(f'debug must be a BOOL, not a {type(debug)}')
    if not isinstance(periodicity, bool):
        raise TypeError(f'periodicity must be a BOOL, not a {type(periodicity)}')
    if not isinstance(add_dagger, bool):
        raise TypeError(f'add_dagger must be a BOOL, not a {type(add_dagger)}')
    if not isinstance(staggered, bool):
        raise TypeError(f'staggered must be a BOOL, not a {type(staggered)}')
    # PRINT THE PHRASE: IT USUALLY REFERS TO THE NAME OF THE HAMILTONIAN
    if staggered==True:
        pause(f'STAGGERED {phrase}',debug)
    else:
        pause(phrase,debug)
    if coeffs is None:
        coeffs=[1,1]
    else:
        if not isinstance(coeffs,list):
            raise TypeError(f'coeffs must be a LIST, not a {type(coeffs)}')
    # Compute the total number of particles
    n=nx*ny
    # Hamiltonian
    Hamiltonian=0
    # Define a counter for the number of contributions involved in the Hamiltonian:
    n_contributions=0
    # Make two lists for Single-Site Operators involved in TwoBody Operators
    Op_HH_list=[Corr.Left,Corr.Right]
    Op_VV_list=[Corr.Bottom,Corr.Top]
    # Run over all the single lattice sites, ordered according to the ZIG ZAG CURVE
    for ii in range(n):
        # Compute the corresponding (x,y) coords
        x,y=zig_zag(nx,ny,ii)
        if staggered==True:
            # Compute the staggered factor depending on the site
            staggered_factor=(-1)**(x+y)
            coeffs[1]=-staggered_factor
        # ------------------------------------------------------------------------------
        # HORIZONTAL 2BODY HAMILTONIAN
        if x<nx-1:
            Op_sites_list=[ii+1,ii+2]
            Hamiltonian=Hamiltonian+coeffs[0]*two_body_op(Op_HH_list,Corr.ID,Op_sites_list,n)
            n_contributions+=1
        else:
            if periodicity==True:
                jj= inverse_zig_zag(nx,ny,0,y)
                Op_sites_list=[ii+1,jj+1]
                # PERIODIC BOUNDARY CONDITIONS
                Hamiltonian=Hamiltonian+coeffs[0]*two_body_op(Op_HH_list,Corr.ID,Op_sites_list,n)
                n_contributions+=1
        # ------------------------------------------------------------------------------
        # VERTICAL 2BODY HAMILTONIAN
        if y<ny-1:
            Op_sites_list=[ii+1,ii+nx+1]
            Hamiltonian=Hamiltonian+coeffs[1]*two_body_op(Op_VV_list,Corr.ID,Op_sites_list,n)
            n_contributions+=1
        else:
            if periodicity==True:
                jj=inverse_zig_zag(nx,ny,x,0)
                Op_sites_list=[ii+1,jj+1]
                # PERIODIC BOUNDARY CONDITIONS
                Hamiltonian=Hamiltonian+coeffs[1]*two_body_op(Op_VV_list,Corr.ID,Op_sites_list,n)
                n_contributions+=1
    if not isspmatrix(Hamiltonian):
        Hamiltonian=csr_matrix(Hamiltonian)
    if add_dagger==True:
        Hamiltonian=Hamiltonian+csr_matrix(Hamiltonian.conj().transpose())
        n_contributions=2*n_contributions
    if debug:
        print(f'    2BODY CONTRIBUTIONS {str(n_contributions)}')
    return Hamiltonian, n_contributions

# ====================================================================
def Plaquette_Hamiltonian(nx,ny,Plaq,phrase,debug,periodicity=False,add_dagger=False):
    # CHECK ON TYPES
    if not np.isscalar(nx) and not isinstance(nx, int):
        raise TypeError(f'nx must be a SCALAR & INTEGER, not a {type(nx)}')
    if not np.isscalar(ny) and not isinstance(ny, int):
        raise TypeError(f'ny must be a SCALAR & INTEGER, not a {type(ny)}')
    if not isinstance(Plaq,Plaquette):
        raise TypeError(f'Plaq must be a CLASS <Plaquette>, not a {type(Plaq)}')
    if not isinstance(phrase, str):
        raise TypeError(f'phrase must be a STRING, not a {type(phrase)}')
    if not isinstance(debug, bool):
        raise TypeError(f'debug must be a BOOL, not a {type(debug)}')
    if not isinstance(periodicity, bool):
        raise TypeError(f'periodicity must be a BOOL, not a {type(periodicity)}')
    if not isinstance(add_dagger, bool):
        raise TypeError(f'add_dagger must be a BOOL, not a {type(add_dagger)}')
    # PRINT THE PHRASE: IT USUALLY REFERS TO THE NAME OF THE HAMILTONIAN
    pause(phrase,debug)
    # Compute the total number of particles
    n=nx*ny
    # Number of Plaquettes
    n_plaquettes=0
    # Plaquette Hamiltonian
    H_Plaq=0
    # Define a list with the Four Operators involved in the Plaquette:
    Operator_list=[Plaq.BL,Plaq.BR,Plaq.TL,Plaq.TR]
    for ii in range(n):
        # Compute the corresponding (x,y) coords
        x,y=zig_zag(nx,ny,ii)
        if x<nx-1 and y<ny-1:
            # List of Sites where to apply Operators
            Sites_list=[ii+1,ii+2,ii+nx+1,ii+nx+2]
            H_Plaq=H_Plaq+four_body_operator(Operator_list,Plaq.ID,Sites_list,n)
            n_plaquettes+=1
        else:
            if periodicity:
                # PERIODIC BOUNDARY CONDITIONS
                if x<nx-1 and y==ny-1:
                    # On the UPPER BORDER
                    jj=inverse_zig_zag(nx,ny,x,0)
                    # List of Sites where to apply Operators
                    Sites_list=[ii+1,ii+2,jj+1,jj+2]
                    H_Plaq=H_Plaq+four_body_operator(Operator_list,Plaq.ID,Sites_list,n)
                    n_plaquettes+=1
                elif x==nx-1 and y<ny-1:
                    # On the RIGHT BORDER
                    # List of Sites where to apply Operators
                    Sites_list=[ii+1,ii+2-nx,ii+nx+1,ii+2]
                    H_Plaq=H_Plaq+four_body_operator(Operator_list,Plaq.ID,Sites_list,n)
                    n_plaquettes+=1
                else:
                    # On the UPPER RIGHT CORNER
                    # List of Sites where to apply Operators
                    Sites_list=[ii+1,ii+2-nx,nx,1]
                    H_Plaq=H_Plaq+four_body_operator(Operator_list,Plaq.ID,Sites_list,n)
                    n_plaquettes+=1
    if not isspmatrix(H_Plaq):
        H_Plaq=csr_matrix(H_Plaq)
    if add_dagger:
        H_Plaq=H_Plaq+csr_matrix(H_Plaq.conj().transpose())
        n_plaquettes=2*n_plaquettes
    if debug:
        print(f'    N PLAQUETTES {str(n_plaquettes)}')
    return H_Plaq, n_plaquettes

# ====================================================================
def Borders_Hamiltonian(nx,ny,Killed_Modes,phrase,debug):
    # CHECK ON TYPES
    if not np.isscalar(nx) and not isinstance(nx, int):
        raise TypeError(f'nx must be a SCALAR & INTEGER, not a {type(nx)}')
    if not np.isscalar(ny) and not isinstance(ny, int):
        raise TypeError(f'ny must be a SCALAR & INTEGER, not a {type(ny)}')
    if not isinstance(Killed_Modes,Rishon_Modes):
        raise TypeError(f'Killed_Modes must be a CLASS <Penalties>, not a {type(Killed_Modes)}')
    if not isinstance(phrase, str):
        raise TypeError(f'phrase must be a STRING, not a {type(phrase)}')
    if not isinstance(debug, bool):
        raise TypeError(f'debug must be a BOOL, not a {type(debug)}')
    # PRINT THE PHRASE: IT USUALLY REFERS TO THE NAME OF THE HAMILTONIAN
    pause(phrase,debug)
    # Compute the total number of particles
    n=nx*ny
    # Define a counter for the number of Border Penalties:
    n_border_penalties=0
    # Border Hamiltonian
    H_Borders=0
    for ii in range(n):
        x,y=zig_zag(nx,ny,ii)
        if y==0:
            if x==0:
                # Kill LEFT & BOTTOM Rishon modes
                H_Borders=H_Borders+local_op(Killed_Modes.Left*Killed_Modes.Bottom,Killed_Modes.ID,ii+1,n)
                n_border_penalties+=1
            elif x==nx-1:
                # Kill RIGHT & BOTTOM Rishon modes
                H_Borders=H_Borders+local_op(Killed_Modes.Right*Killed_Modes.Bottom,Killed_Modes.ID,ii+1,n)
                n_border_penalties+=1
            else:
                # Kill BOTTOM Rishon modes
                H_Borders=H_Borders+local_op(Killed_Modes.Bottom,Killed_Modes.ID,ii+1,n)
                n_border_penalties+=1
        elif y==ny-1:
            if x==0:
                # Kill LEFT & TOP Rishon modes
                H_Borders=H_Borders+local_op(Killed_Modes.Left*Killed_Modes.Top,Killed_Modes.ID,ii+1,n)
                n_border_penalties+=1
            elif x==nx-1:
                # Kill RIGHT & TOP Rishon modes
                H_Borders=H_Borders+local_op(Killed_Modes.Right*Killed_Modes.Top,Killed_Modes.ID,ii+1,n)
                n_border_penalties+=1
            else:
                # Kill TOP Rishon modes
                H_Borders=H_Borders+local_op(Killed_Modes.Top,Killed_Modes.ID,ii+1,n)
                n_border_penalties+=1
        else:
            if x==0:
                # Kill LEFT Rishon modes
                H_Borders=H_Borders+local_op(Killed_Modes.Left,Killed_Modes.ID,ii+1,n)
                n_border_penalties+=1
            elif x==nx-1:
                # Kill RIGHT Rishon modes
                H_Borders=H_Borders+local_op(Killed_Modes.Right,Killed_Modes.ID,ii+1,n)
                n_border_penalties+=1
    if debug:
        print(f'    PENALTIES {str(n_border_penalties)}')
    return H_Borders, n_border_penalties

# ====================================================================
def Vocabulary_Local_Operator(nx,ny,Operator,debug,staggered=False):
    # CHECK ON TYPES
    if not np.isscalar(nx) and not isinstance(nx, int):
        raise TypeError(f'nx must be a SCALAR & INTEGER, not a {type(nx)}')
    if not np.isscalar(ny) and not isinstance(ny, int):
        raise TypeError(f'ny must be a SCALAR & INTEGER, not a {type(ny)}')
    if not isinstance(Operator,Local_Operator):
       raise TypeError(f'Operator must be an instance of <class: Local_Operator>, not a {type(Operator)}')
    if not isinstance(debug, bool):
        raise TypeError(f'debug must be a BOOL, not a {type(debug)}')
    if not isinstance(staggered, bool):
        raise TypeError(f'staggered must be a BOOL, not a {type(staggered)}')
    # PRINT THE PHRASE:
    if staggered:
        pause(Operator.Op_name,debug)
    else:
        pause(Operator.Op_name,debug)
    # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
    n=nx*ny
    # DEFINE A VOCABULARY
    vocabulary={}
    # LIST of VOCABULARY NAMES
    list_Op_names=[]
    for ii in range(n):
        # Compute the corresponding (x,y) coords
        x,y=zig_zag(nx,ny,ii)
        point=coords(x,y)
        if staggered:
            # Compute the staggered factor depending on the site
            staggered_factor=(-1)**(x+y)
            vocabulary[f'{Operator.Op_name}_{point}']=\
                staggered_factor*local_op(Operator.Op,Operator.ID,ii+1,n)
            list_Op_names.append(f'{Operator.Op_name}_{point}')
        else:
            vocabulary[f'{Operator.Op_name}_{point}']=local_op(Operator.Op,Operator.ID,ii+1,n)
            list_Op_names.append(f'{Operator.Op_name}_{point}')
    return vocabulary, list_Op_names

# ====================================================================
def Vocabulary_Two_Body_Operator(nx,ny,Corr,Corr_name,debug,periodicity=False,staggered=False,add_dagger=False,coeffs=[1,1]):
    # CHECK ON TYPES
    if not np.isscalar(nx) and not isinstance(nx, int):
        raise TypeError(f'nx must be a SCALAR & INTEGER, not a {type(nx)}')
    if not np.isscalar(ny) and not isinstance(ny, int):
        raise TypeError(f'ny must be a SCALAR & INTEGER, not a {type(ny)}')
    if not isinstance(Corr,Two_Body_Correlator):
        raise TypeError(f'Corr must be a CLASS Two_Body_Correlator, not a {type(Corr)}')
    if not isinstance(Corr_name, str):
        raise TypeError(f'phrase must be a STRING, not a {type(Corr_name)}')
    if not isinstance(debug, bool):
        raise TypeError(f'debug must be a BOOL, not a {type(debug)}')
    if not isinstance(periodicity, bool):
        raise TypeError(f'periodicity must be a BOOL, not a {type(periodicity)}')
    if not isinstance(add_dagger, bool):
        raise TypeError(f'add_dagger must be a BOOL, not a {type(add_dagger)}')
    if not isinstance(staggered, bool):
        raise TypeError(f'staggered must be a BOOL, not a {type(staggered)}')
    # PRINT THE PHRASE: IT USUALLY REFERS TO THE NAME OF THE HAMILTONIAN
    if staggered:
        pause(f'{Corr_name}',debug)
    else:
        pause(Corr_name,debug)
    # Compute the total number of particles
    n=nx*ny
    # DEFINE A VOCABULARY
    vocabulary={}
    # LIST of VOCABULARY NAMES
    list_Op_names=[]
    # Make two lists for Single-Site Operators involved in TwoBody Operators
    Op_HH_list=[Corr.Left,Corr.Right]
    HH_Corr_Name=Corr_name+'_Left|Right'
    Op_VV_list=[Corr.Bottom,Corr.Top]
    VV_Corr_Name=Corr_name+'_Bottom|Top'
    # Run over all the single lattice sites, ordered according to the ZIG ZAG CURVE
    for ii in range(n):
        # Compute the corresponding (x,y) coords
        x,y=zig_zag(nx,ny,ii)
        if staggered:
            # Compute the staggered factor depending on the site
            staggered_factor=(-1)**(x+y)
            coeffs[1]=-staggered_factor
        # ------------------------------------------------------------------------------
        # HORIZONTAL 2BODY HAMILTONIAN TERMS
        if x<nx-1:
            Op_sites_list=[ii+1,ii+2]
            x1,y1=zig_zag(nx,ny,Op_sites_list[0]-1)
            x2,y2=zig_zag(nx,ny,Op_sites_list[1]-1)
            link=coords(x1,y1)+'__'+coords(x2,y2)
            vocabulary[f'{HH_Corr_Name}_{link}']=coeffs[0]*two_body_op(Op_HH_list,Corr.ID,Op_sites_list,n)
            list_Op_names.append(f'{HH_Corr_Name}_{link}')
        else:
            if periodicity:
                Op_sites_list=[ii+1,ii-nx+2]
                x1,y1=zig_zag(nx,ny,Op_sites_list[0]-1)
                x2,y2=zig_zag(nx,ny,Op_sites_list[1]-1)
                link=coords(x1,y1)+'__'+coords(x2,y2)
                vocabulary[f'{HH_Corr_Name}_{link}']=coeffs[0]*two_body_op(Op_HH_list,Corr.ID,Op_sites_list,n)
                list_Op_names.append(f'{HH_Corr_Name}_{link}')
        # ------------------------------------------------------------------------------
        # VERTICAL 2BODY HAMILTONIAN TERMS
        if y<ny-1:
            Op_sites_list=[ii+1,ii+nx+1]
            x1,y1=zig_zag(nx,ny,Op_sites_list[0]-1)
            x2,y2=zig_zag(nx,ny,Op_sites_list[1]-1)
            link=coords(x1,y1)+'__'+coords(x2,y2)
            vocabulary[f'{VV_Corr_Name}_{link}']=coeffs[0]*two_body_op(Op_VV_list,Corr.ID,Op_sites_list,n)
            list_Op_names.append(f'{VV_Corr_Name}_{link}')
        else:
            if periodicity:
                jj=inverse_zig_zag(nx,ny,x,0)
                Op_sites_list=[ii+1,jj+1]
                x1,y1=zig_zag(nx,ny,Op_sites_list[0]-1)
                x2,y2=zig_zag(nx,ny,Op_sites_list[1]-1)
                link=coords(x1,y1)+'__'+coords(x2,y2)
                vocabulary[f'{VV_Corr_Name}_{link}']=coeffs[0]*two_body_op(Op_VV_list,Corr.ID,Op_sites_list,n)
                list_Op_names.append(f'{VV_Corr_Name}_{link}')
    return vocabulary, list_Op_names

# ====================================================================
def Vocabulary_Plaquette(nx,ny,Plaq,phrase,debug,periodicity=False,add_dagger=False):
    # CHECK ON TYPES
    if not np.isscalar(nx) and not isinstance(nx, int):
        raise TypeError(f'nx must be a SCALAR & INTEGER, not a {type(nx)}')
    if not np.isscalar(ny) and not isinstance(ny, int):
        raise TypeError(f'ny must be a SCALAR & INTEGER, not a {type(ny)}')
    if not isinstance(Plaq,Plaquette):
        raise TypeError(f'Plaq must be a CLASS <Plaquette>, not a {type(Plaq)}')
    if not isinstance(phrase, str):
        raise TypeError(f'phrase must be a STRING, not a {type(phrase)}')
    if not isinstance(debug, bool):
        raise TypeError(f'debug must be a BOOL, not a {type(debug)}')
    if not isinstance(periodicity, bool):
        raise TypeError(f'periodicity must be a BOOL, not a {type(periodicity)}')
    if not isinstance(add_dagger, bool):
        raise TypeError(f'add_dagger must be a BOOL, not a {type(add_dagger)}')
    # PRINT THE PHRASE: IT USUALLY REFERS TO THE NAME OF THE HAMILTONIAN
    pause(phrase,debug)
    # Compute the total number of particles
    n=nx*ny
    # DEFINE A VOCABULARY
    vocabulary={}
    # LIST of VOCABULARY NAMES
    list_Op_names=[]
    # Define a list with the Four Operators involved in the Plaquette:
    Operator_list=[Plaq.BL,Plaq.BR,Plaq.TL,Plaq.TR]
    for ii in range(n):
        # Compute the corresponding (x,y) coords
        x,y=zig_zag(nx,ny,ii)
        point=coords(x,y)
        if x<nx-1 and y<ny-1:
            # List of Sites where to apply Operators
            Sites_list=[ii+1,ii+2,ii+nx+1,ii+nx+2]
            vocabulary[f'Plaq_{point}']=four_body_operator(Operator_list,Plaq.ID,Sites_list,n)
            list_Op_names.append(f'Plaq_{point}')
        else:
            if periodicity:
                # PERIODIC BOUNDARY CONDITIONS
                if x<nx-1 and y==ny-1:
                    # On the UPPER BORDER
                    jj=inverse_zig_zag(nx,ny,x,0)
                    # List of Sites where to apply Operators
                    Sites_list=[ii+1,ii+2,jj+1,jj+2]
                    vocabulary[f'Plaq_{point}']=four_body_operator(Operator_list,Plaq.ID,Sites_list,n)
                    list_Op_names.append(f'Plaq_{point}')
                elif x==nx-1 and y<ny-1:
                    # On the RIGHT BORDER
                    # List of Sites where to apply Operators
                    Sites_list=[ii+1,ii+2-nx,ii+nx+1,ii+2]
                    vocabulary[f'Plaq_{point}']=four_body_operator(Operator_list,Plaq.ID,Sites_list,n)
                    list_Op_names.append(f'Plaq_{point}')
                else:
                    # On the UPPER RIGHT CORNER
                    # List of Sites where to apply Operators
                    Sites_list=[ii+1,ii+2-nx,nx,1]
                    vocabulary[f'Plaq_{point}']=four_body_operator(Operator_list,Plaq.ID,Sites_list,n)
                    list_Op_names.append(f'Plaq_{point}')
    return vocabulary, list_Op_names

# ====================================================================
def Vocabulary_Penalty_Operators(nx,ny,Penalty,phrase,debug):
    # CHECK ON TYPES
    if not np.isscalar(nx) and not isinstance(nx, int):
        raise TypeError(f'nx must be a SCALAR & INTEGER, not a {type(nx)}')
    if not np.isscalar(ny) and not isinstance(ny, int):
        raise TypeError(f'ny must be a SCALAR & INTEGER, not a {type(ny)}')
    if not isinstance(Penalty,Rishon_Modes):
        raise TypeError(f'Killed_Modes must be a CLASS <Penalties>, not a {type(Penalty)}')
    if not isinstance(phrase, str):
        raise TypeError(f'phrase must be a STRING, not a {type(phrase)}')
    if not isinstance(debug, bool):
        raise TypeError(f'debug must be a BOOL, not a {type(debug)}')
    # PRINT THE PHRASE: IT USUALLY REFERS TO THE NAME OF THE HAMILTONIAN
    pause(phrase,debug)
    # Compute the total number of particles
    n=nx*ny
    # DEFINE A VOCABULARY
    vocabulary={}
    # LIST of VOCABULARY NAMES
    list_Op_names=[]
    for ii in range(n):
        x,y=zig_zag(nx,ny,ii)
        point=coords(x,y)
        if y==0:
            if x==0:
                # Kill LEFT & BOTTOM Rishon modes
                vocabulary[f'{Penalty.Left_name}_{Penalty.Bottom_name}_{point}']=\
                    local_op(Penalty.Left*Penalty.Bottom,Penalty.ID,ii+1,n)
                list_Op_names.append(f'{Penalty.Left_name}_{Penalty.Bottom_name}_{point}')
            elif x==nx-1:
                # Kill RIGHT & BOTTOM Rishon modes
                vocabulary[f'{Penalty.Right_name}_{Penalty.Bottom_name}_{point}']=\
                    local_op(Penalty.Right*Penalty.Bottom,Penalty.ID,ii+1,n)
                list_Op_names.append(f'{Penalty.Right_name}_{Penalty.Bottom_name}_{point}')
            else:
                # Kill BOTTOM Rishon modes
                vocabulary[f'{Penalty.Bottom_name}_{point}']=\
                    local_op(Penalty.Bottom,Penalty.ID,ii+1,n)
                list_Op_names.append(f'{Penalty.Bottom_name}_{point}')
        elif y==ny-1:
            if x==0:
                # Kill LEFT & TOP Rishon modes
                vocabulary[f'{Penalty.Left_name}_{Penalty.Top_name}_{point}']=\
                    local_op(Penalty.Left*Penalty.Top,Penalty.ID,ii+1,n)
                list_Op_names.append(f'{Penalty.Left_name}_{Penalty.Top_name}_{point}')
            elif x==nx-1:
                # Kill RIGHT & TOP Rishon modes
                vocabulary[f'{Penalty.Right_name}_{Penalty.Top_name}_{point}']=\
                    local_op(Penalty.Right*Penalty.Top,Penalty.ID,ii+1,n)
                list_Op_names.append(f'{Penalty.Right_name}_{Penalty.Top_name}_{point}')
            else:
                # Kill TOP Rishon modes
                vocabulary[f'{Penalty.Top_name}_{point}']=\
                    local_op(Penalty.Top,Penalty.ID,ii+1,n)
                list_Op_names.append(f'{Penalty.Top_name}_{point}')
        else:
            if x==0:
                # Kill LEFT Rishon modes
                vocabulary[f'{Penalty.Left_name}_{point}']=\
                    local_op(Penalty.Left,Penalty.ID,ii+1,n)
                list_Op_names.append(f'{Penalty.Left_name}_{point}')
            elif x==nx-1:
                # Kill RIGHT Rishon modes
                vocabulary[f'{Penalty.Right_name}_{point}']=\
                    local_op(Penalty.Right,Penalty.ID,ii+1,n)
                list_Op_names.append(f'{Penalty.Right_name}_{point}')
    return vocabulary, list_Op_names