# =====================================================================================
#                                    ZIGZAG MAPPING
# =====================================================================================
def zig_zag(nx,ny,d):
    # GIVEN THE 1D POINT AT POSITION d OF THE ZIGZAG CURVE IN A nx x ny DISCRETE LATTICE
    # IT PROVIDES THE CORRESPONDING 2D COORDINATES (x,y) OF THE POINT
    # NOTE: THE ZIGZAG CURVE IS BUILT BY ALWAYS COUNTING FROM 0 (NOT 1)
    #       HENCE THE POINTS OF THE 1D CURVE START FROM 0 TO (nx * ny)-1
    #       AND THE COORDINATES (x,y) ARE SUPPOSED TO GO FROM 0 TO nx-1 (ny-1)
    #       FOR MATTER OF CODE AT THE END OF THE PROCEEDING BOTH
    #       THE COORDS (x,y) AND THE POINTS OF THE CURVE HAVE TO
    #       BE SHIFTED BY ADDING 1
    if(d==0):
        x=0
        y=0
    elif(d<nx):
        y=0
        x=d
    else:
        # COMPUTE THE REST OF THE DIVISION
        x=d%nx
        # COMPUTE THE INTEGER PART OF THE DIVISION
        y=int(d/nx)
    return x,y
# =====================================================================================
def inverse_zig_zag(nx,ny,x,y):
    # INVERSE ZIGZAG CURVE MAPPING (from coords to the 1D points)
    # NOTE: GIVEN THE SIZES (nx,ny) of a LATTICE, THE COORDS x,y HAS TO
    #       START FROM 0 AND ARRIVE TO nx-1 (ny-1). AT THE END, THE POINTS OF THE
    #       ZIGZAG CURVE START FROM 0. ADD 1 IF YOU WANT TO START FROM 1
    d=(y*nx)+x
    return d
# =====================================================================================
# =====================================================================================



# ========================================================================================
#                                    SNAKE MAPPING
# ========================================================================================
def snake(n,d):
    # GIVEN THE 1D POINT OF THE SNAKE CURVE IN A nxn DISCRETE LATTICE
    # IT PROVIDES THE CORRESPONDING 2D COORDINATES (x,y) OF THE POINT
    # NOTE: THE SNAKE CURVE IS BUILT BY ALWAYS COUNTING FROM 0 (NOT 1)
    #       HENCE THE POINTS OF THE 1D CURVE START FROM 0 TO (n**2)-1
    #       AND THE COORD.S x AND y ARE SUPPOSED TO GO FROM 0 TO n-1
    #       FOR MATTER OF CODE AT THE END OF THE PROCEEDING EITHER
    #       THE COORDS (x,y) EITHER THE POINTS OF THE CURVE HAVE TO
    #       BE SHIFTED BY ADDING 1
    if(d==0):
        x=0
        y=0
    elif(d<n):
        y=0
        x=d
    else:
        # COMPUTE THE REST OF THE DIVISION
        tmp1=d%n
        # COMPUTE THE INTEGER PART OF THE DIVISION
        tmp2=int(d/n)
        tmp3=((tmp2+1)%2)
        y=tmp2
        if(tmp3==0):
            x=n-1-tmp1
        else:
            x=tmp1
    return x,y
# ========================================================================================
# ========================================================================================
def inverse_snake(n,x,y):
    # INVERSE SNAKE CURVE MAPPING (from coords to the 1D points)
    # NOTE: GIVEN THE SIZE L of A SQUARE LATTICE, THE COORDS X,Y HAS TO
    #       START FROM 0 AND ARRIVE TO L-1. AT THE END, THE POINTS OF THE
    #       SNAKE CURVE START FROM 0. ADD 1 IF YOU WANT TO START FROM 1
    d=0
    tmp1=(y+1)%2
    # notice that the first (and hence odd) column is the 0^th column
    if(tmp1==0):
        # EVEN COLUMNS (1,3,5...n-1)
        d=(y*n)+n-1-x
    else:
        # ODD COLUMNS (0,2,4,...n-2)
        d=(y*n)+x
    return d





# ========================================================================================
#                                   HILBERT MAPPING
# ========================================================================================
def regions(num,x,y,s):
    if(num==0):
        # BOTTOM LEFT: CLOCKWISE ROTATE THE COORDS (x,y) OF 90 DEG
        #              THE ROTATION MAKES (x,y) INVERT (y,x)
        t=x
        x=y
        y=t
    elif(num==1):
        # TOP LEFT: TRANSLATE UPWARDS (x,y) OF THE PREVIOUS LEVE
        x=x
        y=y+s
    elif(num==2):
        # TOP RIGHT: TRANSLATE UPWARDS AND RIGHTFORWARD (x,y)
        x=x+s
        y=y+s
    elif(num==3):
        # BOTTOM RIGHT: COUNTER CLOCKWISE ROTATE OF 90 DEG THE (x,y)
        t=x
        x=(s-1)-y+s
        y=(s-1)-t
    return x,y
# ========================================================================================
# ========================================================================================
def bitconv(num):
    # GIVEN THE POSITION OF THE HILBERT CURVE IN A 2x2 SQUARE,
    # IT RETURNS THE CORRESPONDING PAIR OF COORDINATES (rx,ry)
    if(num==0):
        # BOTTOM LEFT
        rx=0
        ry=0
    elif(num==1):
        # TOP LEFT
        rx=0
        ry=1
    elif(num==2):
        # TOP RIGHT
        rx=1
        ry=1
    elif(num==3):
        # BOTTOM RIGHT
        rx=1
        ry=0
    return rx,ry
# ========================================================================================
# ========================================================================================
def hilbert(n,d):
    # MAPPING THE POSITION d OF THE HILBERT CURVE
    # LIVING IN A nxn SQUARE LATTIVE INTO THE
    # CORRESPONDING 2D (x,y) COORDINATES
    # OF A S
    s=1                         # FIX THE INITIAL LEVEL OF DESCRIPTION
    n1=d&3                      # FIX THE 2 BITS CORRESPONDING TO THE LEVEL
    x=0
    y=0
    # CONVERT THE POSITION OF THE POINT IN THE CURVE AT LEVEL 0 INTO
    # THE CORRESPONDING (x,y) COORDINATES
    x,y = bitconv(n1)
    s*=2                        # UPDATE THE LEVEL OF DESCRIPTION
    tmp=d                       # COPY THE POINT d OF THE HILBERT CURVE
    while s<n:
        tmp=tmp>>2              # MOVE TO THE RIGHT THE 2 BITS OF THE POINT dÅ
        n2=tmp&3                # FIX THE 2 BITS CORRESPONDING TO THE LEVEL
        x,y=regions(n2,x,y,s)   # UPDATE THE COORDINATES OF THAT LEVEL
        s*=2                    # UPDATE THE LEVEL OF DESCRIPTION
        s=int(s)
    return x,y
# ========================================================================================
# ========================================================================================
def inverse_regions(num,x,y,s):
    if(num==0):
        # BOTTOM LEFT
        t=x
        x=y
        y=t
    elif(num==1):
        # TOP LEFT
        x=x
        y=y-s
    elif(num==2):
        # TOP RIGHT
        x=x-s
        y=y-s
    elif(num==3):
        # BOTTOM RIGHT
        tt=x
        x=(s-1)-y
        y=(s-1)-tt+s
    return x,y
# ========================================================================================
# ========================================================================================
def inverse_bitconv(rx,ry):
    # GIVEN A PAIR OF COORDINATES (x,y) IN A 2x2 LATTICE, IT
    # RETURNS THE POINT num OF THE CORRESPONDING HILBERT CURVE
    if(rx==0):
        if(ry==0):
            # BOTTOM LEFT
            num=0
        elif(ry==1):
            # TOP LEFT
            num=1
    elif(rx==1):
        if(ry==0):
            # BOTTOM RIGHT
            num=3
        elif(ry==1):
            # TOP RIGHT
            num=2
    return num
# ========================================================================================
# ========================================================================================
def inverse_hilbert(n,x,y):
    # MAPPING THE 2D (x,y) OF A nxn SQUARE INTO THE POSITION d
    # OF THE HILBERT CURVE. REMEMBER THAT THE FINAL POINT
    # HAS TO BE SHIFTED BY 1
    d=0
    n0=0
    s=int(n/2)
    while s>1:
        rx=int(x/s)
        ry=int(y/s)
        n0=inverse_bitconv(rx,ry)
        x,y=inverse_regions(n0,x,y,s)
        d+=n0
        d=d<<2
        s/=2
        s=int(s)
    n0=inverse_bitconv(x,y)
    d+=n0
    return d
# ========================================================================================
# ========================================================================================
def coords(x,y):
    return '('+str(x+1)+','+str(y+1)+')'
