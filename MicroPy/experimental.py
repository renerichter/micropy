'''
Experimental, new or not properly tested/used functions will be collected here. Hence: DO NOT USE!
'''
# %%
# ------------------------------------------------------------------
#                       Operation-process-tree
# ------------------------------------------------------------------


def process_image(im1, im2, operations=[]):
    '''
    Operations on channels between two images
    '''
    res = []
    if operations == []:
        print('No operation had to be done, so did nothing')
    for m in range(im1.shape[0]):
        for y in range(operations.shape[0]):
            res[y, m] = operations[y](im1[m], im2[m])
    return res
