'''
Here number crunching and particular number operations will be included
'''
import numpy as np

# %% --------------------------------------------------------------
#           HEX and INTs
# ----------------------------------------------------------------


def convert_HEX2SIGNEDINT(hexin):
    '''
    hexin -> string of hex-number, shape e.g. '0x010e'
    '''
    hexint = int(hexin, 16)
    if(hexin[1] == 'x'):
        hexi = len(hexin[2:])
    else:
        hexi = len(hexin)
    hexlim = 16**hexi-1
    if (hexint >= hexlim//2):
        res = - (hexint ^ hexlim) - 1
    else:
        res = int(hexin, 16)
    return res


def convert_BYTE2INT32(datain, input_type='uint32'):
    '''
    converts a 4byte input into int32-representation.
    '''
    if input_type == 'uint32':
        # param ---------------------
        # factors = [2**x for x in range(32)]
        # factors.reverse()
        factors = np.array([2147483648, 1073741824, 536870912, 268435456, 134217728, 67108864, 33554432, 16777216, 8388608, 4194304,
                            2097152, 1048576, 524288, 262144, 131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1])
        bl = 8  # byte-length
        # convert input
        dh = ''.join(['0'*(bl-len(x))+x for x in [bin(x)[2:] for x in datain]])
        res = np.sum(np.array([int(x) for x in dh], dtype=np.uint32) * factors)
    elif input_type == 'uhex4':
        # print(datain)
        #res = int(''.join([x[:2] for x in (str(datain)[4:]).split('\\x')]), 16)
        #a = ["{:02x}".format(x) for x in datain]
        #b = ["{:02x}{:02x}{:02x}{:02x}".format(a[x],a[x+1],a[x+2],a[x+3]) for x in range(0,len(a),4)]
        res = np.array([int("{:02x}{:02x}{:02x}{:02x}".format(
            datain[x], datain[x+1], datain[x+2], datain[x+3]), 16) for x in range(0, len(datain), 4)], dtype=np.uint32)
        # group 4 entries to 1 number
    elif input_type == 'hex4':
        # print(datain)
        #res = int(''.join([x[:2] for x in (str(datain)[4:]).split('\\x')]), 16)
        #a = ["{:02x}".format(x) for x in datain]
        #b = ["{:02x}{:02x}{:02x}{:02x}".format(a[x],a[x+1],a[x+2],a[x+3]) for x in range(0,len(a),4)]
        res = np.array([convert_HEX2SIGNEDINT("{:02x}{:02x}{:02x}{:02x}".format(
            datain[x], datain[x+1], datain[x+2], datain[x+3])) for x in range(0, len(datain), 4)], dtype=np.int32)
        # group 4 entries to 1 number
    else:
        raise ValueError('Input type unknown!')
    return res


# %% --------------------------------------------------------------
#           PERMUTATIONS and COMBINATIONS
# ----------------------------------------------------------------


def generate_combinations(nbr_entries, combined_entries=2):
    '''
    Creates a set of index-lists according to "combined_entries" that complete cover all permutations that are possible within the length of the input-list (1D). For combined_entries=2 this means the classical group-combinations: n*(n-1)/2 possibilities. 
    TODO: include higher combined_entries list-combinations

    Example: 
    ========
    a = []
    '''
    if combined_entries == 2:
        id1 = []
        id2 = []
        offset = 1
        for m in range(0, nbr_entries-1):
            for n in range(offset, nbr_entries):
                id1.append(m)
                id2.append(n)
            offset += 1
    else:
        raise ValueError(
            "The expected combined_entries size is not implemented yet.")
    return id1, id2

# %% --------------------------------------------------------------
#           LOGIC OPERATIONS
# ----------------------------------------------------------------
def xor(x, y):
    '''from: https://www.delftstack.com/howto/python/xor-in-python/'''
    return bool((x and not y) or (not x and y))


# %% --------------------------------------------------------------
#               DIFFERENT SUMMING
# ----------------------------------------------------------------

def harmonic_sum(a, b):
    ''' 
    calculates the harmonic sum of two inputs. 
    '''
    return a * b / (a + b)

def harmonic_mean(a, b):
    return a*a*b*b/(a*a+b*b)
