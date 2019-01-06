import cv2
import numpy as np
import math

img = cv2.imread('hough.jpg',0)
img_color_1 = cv2.imread('hough.jpg',1)
img_color_2 = cv2.imread('hough.jpg',1)
circle_img = cv2.imread('hough.jpg',1)

def flipKernel(kernel):

     flip_kernel= [[0 for i in range(kernel.shape[1])] for j in range(kernel.shape[0])]

     for x in range(kernel.shape[0]):
         for y in range(kernel.shape[1]):
             flip_kernel[x][y]=kernel[kernel.shape[0]-1-x][kernel.shape[1]-1-y]

     return flip_kernel



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
    output  = np.asarray([[0 for i in range(i_w+y)] for j in range(i_h+x)])

    
    ind = 1
    for i in range(ind, i_h+ind):
        for j in range(ind, i_w+ind):

            for u in range(-ind, ind+1):
                for v in range(-ind, ind+1):
                    output[i, j] += image_padding[i+u, j+v] * kernel[1+u, 1+v] 
    
    return output[1:-1,1:-1]

def combinedResult(edge_x,edge_y):
    maxVal = 0
    result = np.array([[0 for x in range(edge_x.shape[1])]for y in range(edge_x.shape[0])],dtype='float32')
    for i in range(edge_x.shape[0]):
        for j in range(edge_x.shape[1]):
            sqrt = math.sqrt(edge_x[i,j]**2 + edge_y[i,j]**2)
            result[i,j] = sqrt
            if sqrt >maxVal:
                maxVal = sqrt

    print("maxVal", maxVal)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
        	result[i,j] = result[i,j]/maxVal
    
    return result

# declaring kernel
Cx = [[0 for i in range(3)]for j in range(3)]
Cx[0][0] = -1
Cx[0][1] =  0
Cx[0][2] =  1
Cx[1][0] = -2
Cx[1][1] =  0
Cx[1][2] =  2
Cx[2][0] = -1
Cx[2][1] =  0
Cx[2][2] =  1

Cy = [[0 for i in range(3)]for j in range(3)]
Cy[0][0] = -1
Cy[0][1] = -2
Cy[0][2] = -1
Cy[1][0] =  0
Cy[1][1] =  0
Cy[1][2] =  0
Cy[2][0] =  1
Cy[2][1] =  2
Cy[2][2] =  1

Cx_array=flipKernel(np.asarray(Cx))
print(Cx_array)
Cy_array=flipKernel(np.asarray(Cy))
output_x = convolution(img,np.array(Cx_array))
output_y = convolution(img,np.array(Cy_array))

result = combinedResult(output_x,output_y)
result = result*255

cv2.imwrite('result.jpg',result)
print('result max',np.max(result))

# convert (x,y) -> (d,theta)

def applyThresold(result):
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
        	if result[i,j] >= 35:
        		result[i,j] = 255
    return result


list_d_theta=[]

result = applyThresold(result)
sobel_result = applyThresold(result)
cv2.imwrite('sobel_result.jpg',result)

for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        if result[i,j] == 255:
            for theta in range(0,180):
                d = i*math.cos(math.radians(theta)) - j*math.sin(math.radians(theta))
                list_d_theta.append([d,theta])

list_d_theta = np.array(list_d_theta)
print("list_d_theta size",list_d_theta.shape)
def findMin(list_d_theta):
	minVal = 0
	for i in range(list_d_theta.shape[0]):
			if minVal > list_d_theta[i,0]:
				minVal = list_d_theta[i,0]
	return minVal

minVal = findMin(list_d_theta)
print("Min val,", minVal)

def addMinValue(minVal,list_d_theta):
    for i in range(list_d_theta.shape[0]):
        list_d_theta[i,0] += abs(minVal)
    return list_d_theta

result = addMinValue(minVal,list_d_theta)
print("minVal 2",np.min(result))

def findDimension(result):
	maxVal = 0
	for i in range(result.shape[0]):
		if maxVal < result[i,0]:
			maxVal = result[i,0]
	return maxVal

dimension = int(findDimension(result))
print('dimension - ',dimension)

array = np.array([[0 for i in range(180)] for j in range(dimension+1)])
# populating voting array
for i in range(result.shape[0]):
	d = int(result[i,0])
	# print("d ",d)
	theta = int(result[i,1])
	# print("theta " ,theta)
	array[d,theta] += 1

print(np.max(array))
cv2.imwrite("sinusoid.jpg",np.asarray(array,dtype='uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

maxVal = np.max(array)

thresold = int(0.3*maxVal)
print('Thersold ',thresold)

points = []

for i in range(array.shape[0]):
    for j in range(array.shape[1]):
        if array[i,j] > thresold:
            points.append([i,j])

# print("Points ",points)

for point in points:
    y1 = 0 
    y2 = 769
    if point[1]!=90:
        x1 = int((y1*math.sin(math.radians(point[1])))//(math.cos(math.radians(point[1])))) +((point[0]+minVal)//(math.cos(math.radians(point[1]))))
        x2 = int((y2*math.sin(math.radians(point[1])))//(math.cos(math.radians(point[1])))) +((point[0]+minVal)//(math.cos(math.radians(point[1]))))
        if(point[1]<75):
            cv2.line(img_color_1, (int(y1) ,int(x1)), (int(y2), int(x2)), (255,0,0), 1)
        else:
            cv2.line(img_color_2, (int(y1) ,int(x1)), (int(y2), int(x2)), (0,0,255), 1)
        #print(x,y)

cv2.imwrite('blue_line.jpg',img_color_1)
cv2.imwrite('red_line.jpg',img_color_2)

# detetct coins
#circle_points = np.array([[[0 for i in range(4)]for j in range(sobel_result.shape[1])]for k in range(sobel_result.shape[0])])
#print(circle_points.shape)
# Finding tuple of (a,b,r)
circle_points = []
for i in range(sobel_result.shape[0]):
    for j in range(sobel_result.shape[1]):
        if sobel_result[i,j] == 255:    
            r = [20,21,22,23,]
            for radius in r:
                for theta in range(0,361):
                    a = int(j - radius * math.cos(math.radians(theta)))
                    b = int(i + radius * math.sin(math.radians(theta)))
                    circle_points.append((a,b,radius-21))

# print('circle_points',circle_points[1])
print('out')
circle_points = np.array(circle_points)
a_max = np.max(circle_points[:,0]) + 1
b_max = np.max(circle_points[:,1]) + 1
radius_max = np.max(circle_points[:,2]) + 1

accumalator = np.zeros((a_max,b_max,radius_max))

for i in range(len(circle_points)):
    if circle_points[i][0] >= 0 and circle_points[i][1] >= 0:
    	accumalator[circle_points[i][0],circle_points[i][1],circle_points[i][2]] += 1

print('Accumalator matrix operation complete')
final_points = []
max_val = np.max(accumalator)
tr = int(0.70*max_val)

# choosing points above thresold
for i in range(accumalator.shape[0]):
    for j in range(accumalator.shape[1]):
        for k in range(accumalator.shape[2]):
            if accumalator[i,j,k] > tr :
                final_points.append([i,j,k])
print('finding points')

# drawing circles 
for point in final_points:
    cv2.circle(circle_img,(point[0],point[1]),(point[2]+21),(0,255,0),1)

cv2.imwrite('coin.jpg',circle_img)





# print(a_max,b_max,radius_max)










