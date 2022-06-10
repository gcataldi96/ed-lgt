import numpy as np
from scipy.sparse.base import isspmatrix
from scipy.sparse.csr import isspmatrix_csr
# =========================================================================================================
class Local_Operator:
    def __init__(self,Operator,Operator_name):
        # CHECK ON TYPES
        if not isspmatrix_csr(Operator):
            raise TypeError(f'Operator should be an CSR_MATRIX, not a {type(Operator)}')
        if not isinstance(Operator_name, str):
            raise TypeError(f'Operator_name should be a STRING, not a {type(Operator_name)}')
        self.Op=Operator
        self.Op_name=Operator_name
    
    def get_identity(self,Identity):
        # CHECK ON TYPES
        if not isspmatrix(Identity):
            raise TypeError(f'Identity should be a SPARSE MATRIX, not a {type(Identity)}')
        self.ID=Identity

# =========================================================================================================
class Plaquette:
    def __init__(self,Op_Bottom_Left,Op_Bottom_Right,Op_Top_Left,Op_Top_Right):
        # CHECK ON TYPES
        if not isspmatrix_csr(Op_Bottom_Left):
            raise TypeError(f'Op_Bottom_Left should be an CSR_MATRIX, not a {type(Op_Bottom_Left)}')
        if not isspmatrix_csr(Op_Bottom_Right):
            raise TypeError(f'Op_Bottom_Right should be an CSR_MATRIX, not a {type(Op_Bottom_Right)}')
        if not isspmatrix_csr(Op_Top_Left):
            raise TypeError(f'Op_Top_Left should be an CSR_MATRIX, not a {type(Op_Top_Left)}')
        if not isspmatrix_csr(Op_Top_Right):
            raise TypeError(f'Op_Top_Right should be an CSR_MATRIX, not a {type(Op_Top_Right)}')
        self.BL=Op_Bottom_Left
        self.BR=Op_Bottom_Right
        self.TL=Op_Top_Left
        self.TR=Op_Top_Right

    def add_Op_names(self,Op_Bottom_Left_name,Op_Bottom_Right_name,Op_Top_Left_name,Op_Top_Right_name):
        # CHECK ON TYPES
        if not isinstance(Op_Bottom_Left_name, str):
            raise TypeError(f'Op_Bottom_Left_name should be a STRING, not a {type(Op_Bottom_Left_name)}')
        if not isinstance(Op_Bottom_Right_name, str):
            raise TypeError(f'Op_Bottom_Right_name should be a STRING, not a {type(Op_Bottom_Right_name)}')
        if not isinstance(Op_Top_Left_name, str):
            raise TypeError(f'Op_Top_Left_name should be a STRING, not a {type(Op_Top_Left_name)}')
        if not isinstance(Op_Top_Right_name, str):
            raise TypeError(f'Op_Top_Right_name should be a STRING, not a {type(Op_Top_Right_name)}')
        self.BL_name=Op_Bottom_Left_name
        self.BR_name=Op_Bottom_Right_name
        self.TL_name=Op_Top_Left_name
        self.TR_name=Op_Top_Right_name

    def get_identity(self,Identity):
        # CHECK ON TYPES
        if not isspmatrix(Identity):
            raise TypeError(f'Identity should be a SPARSE MATRIX, not a {type(Identity)}')
        self.ID=Identity

    def print_Plaquette(self,sites_list,value):
        if not isinstance(sites_list, list):
            raise TypeError(f'sites_list should be a LIST, not a {type(sites_list)}')
        if len(sites_list)!=4:
            raise ValueError(f'sites_list should have 4 elements, not {str(len(sites_list))}')
        if not isinstance(value, float):
            raise TypeError(f'sites_list should be a FLOAT REAL NUMBER, not a {type(value)}')
        if value>0:
            value=format(value,'.5f')
        else:
            if np.abs(value)<10**(-10):
                value=format(np.abs(value),'.5f')
            else:
                value=format(value,'.4f')
        print(f'    ({sites_list[2]})-------({sites_list[3]})')
        print(f'      |           |')
        print(f'      |  {value}  |')
        print(f'      |           |')
        print(f'    ({sites_list[0]})-------({sites_list[1]})')
        print('')


# =========================================================================================================
class Rishon_Modes:
    def __init__(self,P_Left,P_Right,P_Bottom,P_Top):
        # CHECK ON TYPES
        if not isspmatrix_csr(P_Left):
            raise TypeError(f'P_Left should be an CSR_MATRIX, not a {type(P_Left)}')
        if not isspmatrix_csr(P_Right):
            raise TypeError(f'P_Right should be an CSR_MATRIX, not a {type(P_Right)}')
        if not isspmatrix_csr(P_Bottom):
            raise TypeError(f'P_Bottom should be an CSR_MATRIX, not a {type(P_Bottom)}')
        if not isspmatrix_csr(P_Top):
            raise TypeError(f'P_Top should be an CSR_MATRIX, not a {type(P_Top)}')
        self.Left=P_Left
        self.Right=P_Right
        self.Bottom=P_Bottom
        self.Top=P_Top

    def add_Op_names(self,P_Left_name,P_Right_name,P_Bottom_name,P_Top_name):
        if not isinstance(P_Left_name, str):
            raise TypeError(f'P_Left_name should be a STRING, not a {type(P_Left_name)}')
        if not isinstance(P_Right_name, str):
            raise TypeError(f'P_Right_name should be a STRING, not a {type(P_Right_name)}')
        if not isinstance(P_Bottom_name, str):
            raise TypeError(f'P_Bottom_name should be a STRING, not a {type(P_Bottom_name)}')
        if not isinstance(P_Top_name, str):
            raise TypeError(f'P_Top_name should be a STRING, not a {type(P_Top_name)}')
        self.Left_name=P_Left_name
        self.Right_name=P_Right_name
        self.Bottom_name=P_Bottom_name
        self.Top_name=P_Top_name

    def get_identity(self,Identity):
        if not isspmatrix(Identity):
            raise TypeError(f'Identity should be a SPARSE MATRIX, not a {type(Identity)}')
        self.ID=Identity


# =========================================================================================================
class Two_Body_Correlator:
    def __init__(self,Op_Left,Op_Right,Op_Bottom,Op_Top):
        # CHECK ON TYPES
        if not isspmatrix_csr(Op_Left):
            raise TypeError(f'Op_Left should be an CSR_MATRIX, not a {type(Op_Left)}')
        if not isspmatrix_csr(Op_Right):
            raise TypeError(f'Op_Right should be an CSR_MATRIX, not a {type(Op_Right)}')
        if not isspmatrix_csr(Op_Bottom):
            raise TypeError(f'Op_Bottom should be an CSR_MATRIX, not a {type(Op_Bottom)}')
        if not isspmatrix_csr(Op_Top):
            raise TypeError(f'Op_Top should be an CSR_MATRIX, not a {type(Op_Top)}')
        self.Left=Op_Left
        self.Right=Op_Right
        self.Bottom=Op_Bottom
        self.Top=Op_Top

    def get_identity(self,Identity):
        if not isspmatrix(Identity):
            raise TypeError(f'Identity should be a SPARSE MATRIX, not a {type(Identity)}')
        self.ID=Identity

    def add_Op_names(self,Op_Left_name,Op_Right_name,Op_Bottom_name,Op_Top_name):
        if not isinstance(Op_Left_name, str):
            raise TypeError(f'Op_Left_name should be a STRING, not a {type(Op_Left_name)}')
        if not isinstance(Op_Right_name, str):
            raise TypeError(f'Op_Right_name should be a STRING, not a {type(Op_Right_name)}')
        if not isinstance(Op_Bottom_name, str):
            raise TypeError(f'Op_Bottom_name should be a STRING, not a {type(Op_Bottom_name)}')
        if not isinstance(Op_Top_name, str):
            raise TypeError(f'Op_Top_name should be a STRING, not a {type(Op_Top_name)}')
        self.Left_name=Op_Left_name
        self.Right_name=Op_Right_name
        self.Bottom_name=Op_Bottom_name
        self.Top_name=Op_Top_name





