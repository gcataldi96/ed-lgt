import numpy as np
import os, re
import subprocess as sub
import argparse

# =======================================================================================
def ground_state(n,energy_file_name,alpha):
    # ACQUIRE THE FINAL GROUND STATE ENERGY VALUE:
    # PROPERLY, THE FILE IS A SEQUENCE OF LINES
    # (ONE FOR EACH ITERATION OF THE SIMULATION)
    # CONTAINING THE TIME NEEDED TO ESTIMATE THE GS ENERGY,
    # AND THE CORRESPONDING ENERGY ESTIMATED IN THAT ITERATION.
    # WE ARE ONLY INTERESTED IN THE GS VALUE OF THE LAST ITERATION,
    # HENCE, WE ONLY FOCUS ON THE LAST LINE OF THE FILE
    g=open(energy_file_name,'r')
    line=g.readlines()
    g.close()
    # SAVE THE LAST LINE OF THE FILE
    f=str(line[-1])
    # REMOVE THE INITIAL AND FINAL WHITE SPACES,
    # AND REPLACE THE MULTIPLE INTERNAL WITH A COMMA
    f=re.sub("\s+", ",", f.strip())
    # ISOLATE THE TERMS SEPARATED BY A COMMA
    f=f.split(',')
    # COMPUTE THE OFFSET DUE TO NON PHYSICAL HAMILTONIAN CONTRIBUTIONS
    single_site=4*(n-1)*alpha
    plaquette=alpha*((n-1)**2)
    rishons=alpha*(2*n*(n-1))
    tot=single_site+plaquette+rishons
    # GET THE GROUND STATE ENERGY PER SITE(gse)
    gse=(float(f[1])+tot)/(n**2)
    return gse
# =======================================================================================
def check_number(a, b, threshold,convergence):
    # DEFINE A COUNTER WHIHC HOLDS 0 IF a AND b CONVERGE, AND HOLDS 1 OTHERWISE
    counter=0
    if(type(a)==type(b)):
        # CASE OF FLOAT OR INT TYPE
        if(type(a)==float or type(a)==int):
            if convergence=='RELATIVE':
                c=abs(abs(a-b)/a)           # RELATIVE CONVERGENCE
            elif convergence=='ABSOLUTE':
                c=abs(a-b)                  # ABSOLUTE CONVERGENCE
            if(c>threshold):
                counter=1
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print('CHECK '+convergence+' DIFFERENCE')
                print('NUMBERS DIFFERENT WITH A '+str(threshold)+' PRECISION')
                print('a='+str(a))
                print('b='+str(b))
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    else:
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('ERROR: a & b ARE NOT OF THE SAME TYPE')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    return counter
# =======================================================================================
def check_correlations(name_file,threshold,option,n,convergence):
    # THIS FUNCTION CHECK THE VALUES OF 2 or 4 POINT CORRELATION FUNCTIONS
    # FOR NEAREST NEIGHBOR SITES WITH OPEN BOUNDARY CONDITIONS
    # SINCE WE EXPECT THEM TO BE EQUAL TO ONE, THE FUNCTION
    #  check_number  WILL BE CALLED IN ORDER TO CHECK ONE BY ONE ALL
    # THESE VALUES: THIS WOULD BE ENSURE TO LIE IN THE RIGHT SYMMETRY
    # SECTOR AND THE TORIC MAPPING TO BE EQUIVALENT TO THE INITIAL
    # FERMION MODEL.
    print('-----------------------------------------')
    print('CHECK '+option+' CORRELATIONS WITH '+convergence+' DIFF.')
    # DEFINE A VARIABLE WHIHC UPDATES WITH +1 EACH TIME A CORRELATION IS NOT CORRECT
    counter=0
    f=open(name_file,'r')
    for a in f:
        v=re.sub('\s+', ',', a.strip())
        v=v.strip().split(',')
        if option=='W':
            # CHECK THE VALUES OF W CORRELATIONS
            if int(v[2])==int(v[0])+1 and int(v[3])==int(v[1]):
                print('('+v[0]+','+v[1]+')    ('+v[2]+','+v[3]+')')
                counter+=check_number(float(v[4]),1.,threshold,convergence)
            if int(v[2])==int(v[0])   and int(v[3])==int(v[1])+1:
                print('('+v[0]+','+v[1]+')    ('+v[2]+','+v[3]+')')
                counter+=check_number(float(v[6]),1.,threshold,convergence)
        elif option=='P':
            # CHECK THE PLAQUETTE VALUES
            if len(v)==5:
                print('PLAQUETTE ','['+v[0]+' '+v[1]+' '+v[2]+' '+v[3]+']')
                counter+=check_number(float(v[4]),1.,threshold,convergence)
        elif option=='B':
            # ON THE LEFT BORDER
            if int(v[0])==1:
                # ON THE DOWN LEFT CORNER
                if int(v[1])==1:
                    print(' SITE ('+v[0]+','+v[1]+')')
                    # CHECK THE NUMBER OF RISHONS ON THE LEFT LINK
                    counter+=check_number(float(v[8]),0.,threshold,convergence)
                    # CHECK THE NUMBER OF RISHONS ON THE DOWN LINK
                    counter+=check_number(float(v[14]),0.,threshold,convergence)
                # ON THE TOP LEFT CORNER
                elif int(v[1])==n:
                    print(' SITE ('+v[0]+','+v[1]+')')
                    # CHECK THE NUMBER OF RISHONS ON THE LEFT LINK
                    counter+=check_number(float(v[8]),0.,threshold,convergence)
                    # CHECK THE NUMBER OF RISHONS ON THE UP LINK
                    counter+=check_number(float(v[12]),0.,threshold,convergence)
                # NOT ON THE CORNERS OF THE LEFT BORDER
                else:
                    print(' SITE ('+v[0]+','+v[1]+')')
                    # CHECK THE NUMBER OF RISHONS ON THE LEFT LINK
                    counter+=check_number(float(v[8]),0.,threshold,convergence)
            # ON THE RIGHT BORDER
            elif int(v[0])==n:
                # ON THE DOWN RIGHT CORNER
                if int(v[1])==1:
                    print(' SITE ('+v[0]+','+v[1]+')')
                    # CHECK THE NUMBER OF RISHONS ON THE RIGHT LINK
                    counter+=check_number(float(v[10]),0.,threshold,convergence)
                    # CHECK THE NUMBER OF RISHONS ON THE DOWN LINK
                    counter+=check_number(float(v[14]),0.,threshold,convergence)
                # ON THE TOP RIGHT CORNER
                elif int(v[1])==n:
                    print(' SITE ('+v[0]+','+v[1]+')')
                    # CHECK THE NUMBER OF RISHONS ON THE RIGHT LINK
                    counter+=check_number(float(v[10]),0.,threshold,convergence)
                    # CHECK THE NUMBER OF RISHONS ON THE UP LINK
                    counter+=check_number(float(v[12]),0.,threshold,convergence)
                # NOT ON THE CORNERS OF THE RIGHT BORDER
                else:
                    print(' SITE ('+v[0]+','+v[1]+')')
                    # CHECK THE NUMBER OF RISHONS ON THE RIGHT LINK
                    counter+=check_number(float(v[10]),0.,threshold,convergence)
            # NEITHER ON THE LEFT NOT ON THE RIGHT BORDER
            else:
                # ON THE DOWN BORDER
                if int(v[1])==1:
                    print(' SITE ('+v[0]+','+v[1]+')')
                    # CHECK THE NUMBER OF RISHONS ON THE DOWN LINK
                    counter+=check_number(float(v[14]),0.,threshold,convergence)
                # ON THE TOP BORDER
                elif int(v[1])==n:
                    print(' SITE ('+v[0]+','+v[1]+')')
                    # CHECK THE NUMBER OF RISHONS ON THE UP LINK
                    counter+=check_number(float(v[12]),0.,threshold,convergence)
    f.close()
    print('THE # OF WRONG '+option+' CORRELATIONS IS '+str(counter))
    print('-----------------------------------------')
# ========================================================================================
#   INTERFACE WITH THE PROGRAMMER: CHOICE OF THE PARAMETERS OF THE HAMILTONIAN
# ========================================================================================
parser = argparse.ArgumentParser(description='GUIDE LINES FOR ANALYSING SIMULATION')
parser.add_argument('-L','--L',nargs=1,type=int, help='SIZE n OF A LATTICE nxn')
parser.add_argument('-N','--N',nargs=1,type=int, help='TOTAL # OF PARTICLES')
parser.add_argument('-t','--t',nargs=1,type=float,help='COUPLING CONSTANT t' )
parser.add_argument('-U','--U',nargs=1,type=float,help='COUPLING CONSTANT U' )
parser.add_argument('-a','--a',nargs=1,type=int,help='PARAM FOR EXTERNAL CONSTRAINTS' )
parser.add_argument('-D','--D',nargs=1,type=int,help='BOND DIMENSION D' )
parser.add_argument('-tol','--tolerance',nargs=1,type=float,help='TOLERANCE FOR CHECKS')
parser.add_argument('-c','--conv',  type=str, help='TYPE OF CONVERGENCE')
parser.add_argument('-s','--simul', type=str, help='DESCRIPTION OF SIMULATION' )

args = parser.parse_args()

n=args.L[0]                         # SIZE OF THE LATTICE
N=args.N[0]                         # FIXED NUMBER OF PARTICLES
t=args.t[0]                         # HOPPING COUPLING CONSTANT
U=args.U[0]                         # ONE SITE COULOMB POTENTIAL
alpha=args.a[0]                     # COUPLIGN FOR EXTERNAL CONSTRAINTS
D=args.D[0]                         # BOND DIMENSION
threshold=args.tolerance[0]         # THRESHOLD FOR FURTHER CHECKS
conv=str(args.conv)                 # TYPE OF CONVERGENCE WHEN MAKING CHECKS
name_sim=str(args.simul)            # NAME OF THE SIMULATION TO BE ANALIZED
if conv=='R':
    convergence='RELATIVE'
elif conv=='A':
    convergence='ABSOLUTE'
#=========================================================================================
#=========================================================================================
# llabel IS THE POSTSCRIPT WHICH IDENTIFIES THE
# CHOSEN SET OF PARAMETERS. IT WILL BE APPENDED TO
# THE NAME OF FOLDERS OR FILES
llabel=labeling(n,N,t,U,alpha,D)
# DEFINE THE NAME OF THE FOLDER WHERE INPUT DATA LIE
results='results'+llabel
# CREATE A FILE WHERE ALL THE MAIN INFORMATIONS ARE STORED
outcomes_name=os.path.join(results,'outcomes'+llabel+'.txt')
out=open(outcomes_name,'w+')
out.write('========================================='+'\n')
out.write('       # N   # t   # U   # a   # D'+'\n')
out.write('        '+str(N)+'   '+str(t)+'   '+str(U)+'   '+str(alpha)+'    '+str(D)+'\n')
out.write('-----------------------------------------'+'\n')
# DEFINE THE PATH FOR THE FILE OF THE GROUND STATE ENERGY
# ACQUIRE THE GROUND STATE ENERGY VALUE
en_file=os.path.join(results,'convergence.log')
energy=ground_state(n,en_file,alpha)
# OPEN THE FILE CONTAINING THE 2 POINT CORRELATIONS
# AND CHECK THE FACT THESE CORRELATIONS ARE EXACTLY 1
# WITH A PRECISION GIVEN BY AN EXTERNAL IMPOSED THRESHOLD
corr_file_W=os.path.join(results,'observables_c.out')
check_correlations(corr_file_W,threshold,'W',n,convergence)
# OPEN THE FILE CONTAINING THE 4 POINT CORRELATIONS
# (PLAQUETTE TERMS) AND CHECK THEM TO BE EQUAL TO 1
corr_file_P=os.path.join(results,'observables_p.out')
check_correlations(corr_file_P,threshold,'P',n,convergence)
# DEFINE THE CORRESPONDING PATH TO THE OBSERVABLES DATA FILE
in_file=os.path.join(results,'observables_l.out')
# CHECK THE VALUES OF HALF LINKS ON THE BORDERS OF THE LATTICE
check_correlations(in_file,threshold,'B',n,convergence)
# DEFINE THE LISTS WHERE STORING DATA
x=list()                                     # list for x coord.
y=list()                                     # list for y coord.
value1=list()                                # list for n_up
value2=list()                                # list for n_down
value3=list()                                # list for n_pair
#=========================================================================================
#=========================================================================================
# OPEN THE INPUT FILE
s=open(in_file,'r')
for a in s:
    v=re.sub('\s+', ',', a.strip())
    v=v.strip().split(',')
    x.append(int(v[0]))
    y.append(int(v[1]))
    value1.append(float(v[2]))
    value2.append(float(v[4]))
    value3.append(float(v[6]))
s.close()
#=========================================================================================
#=========================================================================================
energy=format(energy,'.5f')
out.write('       ENERGY          '+str(energy)+'\n')
out.write('-----------------------------------------'+'\n')
out.write(' # SITE     # UP      # DOWN    # PAIRS |'+'\n')
for a in range(len(value1)):
    up  =format(value1[a],'.3f')
    down=format(value2[a],'.3f')
    pair=format(value3[a],'.3f')
    out.write('  ('+str(x[a])+','+str(y[a])+')     '+str(up)+\
            '      '+str(down)+'     '+str(pair)+'  |'+'\n')
up_particles  =format(sum(value1),'.3f')
down_particles=format(sum(value2),'.3f')
pair_particles=format(sum(value3),'.3f')
out.write('-----------------------------------------'+'\n')
out.write('       UP PARTICLES         '+str(up_particles)+'       |'+'\n')
out.write('       DOWN PARTICLES       '+str(down_particles)+'       |'+'\n')
out.write('       UP+DOWN PARTICLES    '+str(pair_particles)+'       |'+'\n')
out.write('-----------------------------------------'+'\n')
n_tot=sum(value1)+sum(value2)
n_tot=format(n_tot,'.3f')
out.write('       TOTAL PARTICLES      '+str(n_tot)+'      |'+'\n')
out.write('========================================='+'\n')
# ========================================================================================
out.close()








"""
# GENERATE A LIST WHERE THE FIRST ENTRY IS THE LABEL OF THE SIMULATION,
# AND ALL THE OTHERS ARE ENERGY VALUES THIS LIST IS DONE FOR EACH VALUE
# OF THE TOTAL NUMBER OF PARTICLES WE SIMULATE
data=list()
for ii in range(param.shape[0]):
    # FIRST LINE
    data.append(dest)
    for jj in range(param.shape[2]):
        data.append(float(energy[ii][0][jj][0]))
    # SAVE ENERGY VALUES OF THE LIST IN AN ALREADY
    # EXISTING FILE CONTAINING THE RESULTS OF OTHER SIMULATIONS
    # WITH THE SAME VALUE OF THE TOTAL NUMBER BUT OTHER
    # CHOICES OF THE REMAINING PARAMETERS
    name_results='energy_rho_'+str(format(param[ii][0][0][0][0]/(n**2),'.1f'))+'.txt'
    store_results(name_results,data)
    # EMPTY THE EXISTING LIST IN ORDER FOR IT TO BE REFILLED
    # WITH ENERGY VALUES ASSOCIATED TO A DIFFERENT N
    del data[:]

# SAVE THE FILE WITH THE ENERGIES
np.save(name_sim+'/'+name_sim+'_energy.npy',energy)
"""
