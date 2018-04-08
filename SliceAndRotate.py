
def SliceAndRotate(arrayIn):
    # This function takes 4-channel images, slices them to quadrants,
    # rotates rotates and flips each quadrant, and scales them to the
    # original resolution
    # Assuming originally there are N images and masks
       
    N = arrayIn.shape[0]
    X_PIXELS = arrayIn.shape[1]
    Y_PIXELS = arrayIn.shape[2]
    assert X_PIXELS == Y_PIXELS
    mid = int(X_PIXELS/2) # the midpoint we will be slicing on
    
    # Preallocate memory for output, dtype = float64, 32x longer on axis 0
    arrayOut = np.ndarray([32*N]+list(arrayIn.shape[1:]), dtype = 'float64')
    
    # define a scaling funtion using skimage.transform.resise
    # with predefined arugments, purely for brevity
    def scale(q):
        shape = (X_PIXELS, Y_PIXELS)
        mode = 'constant' 
        return resize(q, shape, mode = mode, preserve_range = True)
    
        
    for n in range(N):
        # take one image from arrayIn
        im    = arrayIn[ n, :, : , :]
        
        # slice array into quadrants
        im_nw = im[ :mid, :mid, :] # upper left
        im_ne = im[ mid:, :mid, :] # upper right
        im_sw = im[ :mid, mid:, :] # lower left
        im_se = im[ :mid, :mid, :] # lower right
                
        # combine into a list of quadrants then rescale
        quads = [ im_nw, im_ne, im_sw, im_se ]
        quads = [ scale(q) for q in quads ]
                
        # do verical and horizontal flips for each quadrant
        quads_vflip = [ np.flip(q, 0) for q in quads ]
        quads_hflip = [ np.flip(q, 1) for q in quads ]
        
        # do rotations for each quadrant
        quads_90  = [ np.rot90(q) for q in quads ]
        quads_180 = [ np.rot90(q) for q in quads_90 ]
        quads_270 = [ np.rot90(q) for q in quads_180 ]
        
        # The following transformations SHOULD all be distinct
        # I believe I've thrown out all the non-distinct tranformations
        # e.g. rotate 90 and then vflip = rotate 270 then hflip
        quads_90_hflip  = [ np.flip(q, 0) for q in quads_90 ]
        quads_270_hflip = [ np.flip(q, 0) for q in quads_270 ]
        
        # expand all transforms to (1, X_PIXENS, Y_PIXELS, CHANNELS )
        # to prepare them for concatenation
        all_tfms = []
        all_tfms += [ np.expand_dims(q, 0) for q in quads ]
        all_tfms += [ np.expand_dims(q, 0) for q in quads_vflip ]
        all_tfms += [ np.expand_dims(q, 0) for q in quads_hflip ]
        all_tfms += [ np.expand_dims(q, 0) for q in quads_90 ]
        all_tfms += [ np.expand_dims(q, 0) for q in quads_180 ]
        all_tfms += [ np.expand_dims(q, 0) for q in quads_270 ]
        all_tfms += [ np.expand_dims(q, 0) for q in quads_90_hflip ]
        all_tfms += [ np.expand_dims(q, 0) for q in quads_270_hflip ]
        
        # now concatenate them along the 0th axis
        all_tfms = np.concatenate(tuple(all_tfms), axis = 0)
        arrayOut[ n : (n+32) ] = all_tfms
    
    return arrayOut
        
        
