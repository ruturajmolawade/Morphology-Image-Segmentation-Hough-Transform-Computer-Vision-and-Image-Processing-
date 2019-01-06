import cv2
import numpy as np

img = cv2.imread('point.jpg',0)
# cv2.imshow('image1',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def thresoldImage(image):
    x = 0 
    y = 0
    mean = np.mean(image)
    print(mean)
    image_new = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image_new[i][j] > 1200:
                image_new[i][j] = 250
                x = i
                y = j 
            else:
                image_new[i][j] = 0

    return image_new,x,y

def convolution(image,kernel):
    k_h = kernel.shape[0]
    k_w = kernel.shape[1]

    # calculting image height and width
    i_h = image.shape[0]
    i_w = image.shape[1]

    # zero paddding to not to miss out edges
    x = k_h-1
    y = k_w-1

    image_padding  = np.asarray([[0 for i in range(i_w+y)] for j in range(i_h+x)])
    image_padding[1:-1,1:-1] = image
    #output = image.copy()
    output  = np.asarray([[0 for i in range(i_w+y)] for j in range(i_h+x)])
    ind = 1
    for i in range(ind, i_h+ind):
        for j in range(ind, i_w+ind):

            for u in range(-ind, ind+1):
                for v in range(-ind, ind+1):
                    output[i, j] += image_padding[i+u, j+v] * kernel[1+u, 1+v] 
    
    return output[1:-1,1:-1]

# declaring kernel
Cx = [[0 for i in range(3)]for j in range(3)]
Cx[0][0] = -1
Cx[0][1] = -1
Cx[0][2] = -1
Cx[1][0] = -1
Cx[1][1] =  8
Cx[1][2] = -1
Cx[2][0] = -1
Cx[2][1] = -1
Cx[2][2] = -1

#Kernel = flipKernel(np.asarray(Cx))
output = convolution(img,np.asarray(Cx))
output = np.abs(output)
print("Max",np.max(output))
# print(np.max(output))
output,x,y = thresoldImage(output)
# cv2.imshow('Point detection',output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('point_detection.jpg',output)
output_img = cv2.imread('point_detection.jpg',0)

# drawing circle around detected point
img_final = cv2.circle(img, (y, x), 30, (0, 255, 0), 2)
detected_point = cv2.circle(np.array(output_img), (y, x), 30, (255, 255, 255), 2)

foreground_text ='(' + str(y) + ',' + str(x) + ')'
Scale = 1

lineType = 2
fontc = (255, 255, 255)
fontc1 = (0, 0, 255)
text_font = cv2.FONT_HERSHEY_DUPLEX
TextPosition = (x-30, y-30)


cv2.putText(img_final, foreground_text, TextPosition, text_font, Scale, fontc1, lineType)
cv2.putText(detected_point, foreground_text, TextPosition, text_font, Scale, fontc, lineType)

cv2.imwrite('point_detected.jpg',detected_point)
cv2.imwrite('point_detected_final.jpg',img_final)