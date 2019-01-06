import cv2
import numpy as np

original_img = cv2.imread('noise.jpg',0)

# create binary image
img = np.array(original_img/255,dtype='uint8')

# create structure element
structure = np.ones((3,3),dtype='uint8')
structure_height, structure_width = structure.shape
height,width = img.shape 

# dilation function
def dilation(img,structure):
    image_padding  = np.asarray([[0 for i in range(width+structure_width-1)] for j in range(height+structure_height-1)],dtype='uint8')
    image_padding[1:-1,1:-1] = img
    output  = np.asarray([[0 for i in range(width)] for j in range(height)],dtype='uint8')

    for x1 in range(height):
        for y1 in range(width):
            result = np.multiply(structure,image_padding[x1:x1+structure_width,y1:y1+structure_height])
            if(np.sum(result) > 0):
                output[x1,y1] = 1
            else:
                output[x1,y1] = 0
    return output
            
def errosion(img,structure):
    image_padding  = np.asarray([[0 for i in range(width+structure_width-1)] for j in range(height+structure_height-1)],dtype='uint8')
    image_padding[1:-1,1:-1] = img
    output  = np.asarray([[0 for i in range(width)] for j in range(height)],dtype='uint8')

    for x1 in range(height):
        for y1 in range(width):
            result = np.multiply(structure,image_padding[x1:x1+structure_width,y1:y1+structure_height])
            if(np.sum(result) == 9):
                output[x1,y1] = 1
            else:
                output[x1,y1] = 0
    return output


# opening
def opening(img,structure):
    errosion_output = errosion(img,structure)
    dilation_output = dilation(errosion_output,structure)
    opening_output_img = np.array(dilation_output) #,dtype='uint8')
    # cv2.imshow('Opening output',opening_output_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return opening_output_img  


# closing
def closing(img,structure):
    dilaltion_output = dilation(img,structure)
    errosion_output = errosion(dilaltion_output,structure)
    closing_output_img = np.array(errosion_output)# ,dtype='uint8')
    # cv2.imshow('Closing output',closing_output_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 
    return closing_output_img

# opening -> closing
opening_result = opening(img,structure)
closing_result = closing(opening_result,structure)
# boundary image
boundary_image = closing_result - errosion(closing_result,structure)
closing_result = closing_result*255
boundary_image = boundary_image*255 
cv2.imwrite('res_noise1.jpg',closing_result)
cv2.imwrite('res_bound1.jpg',boundary_image)
# closing -> opening 
closing_result = closing(img,structure) 
opening_result = opening(closing_result,structure)
# boundary image
boundary_image = opening_result - errosion(opening_result,structure)
opening_result = opening_result*255
boundary_image = boundary_image*255 
cv2.imwrite('res_noise2.jpg',opening_result)
cv2.imwrite('res_bound2.jpg',boundary_image)