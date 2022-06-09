import numpy as np
import SU2_Matter_Operators as Op

P_left, P_right, P_bottom, P_top=Op.penalties()
gamma = Op.gamma_operator()
W_Left, W_Right, W_Bottom, W_Top=Op.W_operators()
C_Bottom_Left, C_Bottom_Right, C_Top_Left, C_Top_Right=Op.plaquette()
n_SINGLE, n_PAIR, n_TOTAL= Op.number_operators()
"""
print('-----')
print(P_left)
print('-----')
print(P_right)
print('-----')
print(P_bottom)
print('-----')
print(P_top)

print('-----')
print(W_Left)
print('-----')
print(W_Right)
print('-----')
print(W_Bottom)
print('-----')
print(W_Top)


print('-----')
print(C_Bottom_Left)
print('-----')
print(C_Bottom_Right)
print('-----')
print(C_Top_Left)
print('-----')
print(C_Top_Right)


print('-----')
print(n_SINGLE)
print('-----')
print(n_PAIR)
print('-----')
print(n_TOTAL)
"""
print('-----')
print(P_left*P_bottom)
print('-----')
print(P_bottom*P_right)
print('-----')
print(P_top*P_left)
print('-----')
print(P_top*P_right)