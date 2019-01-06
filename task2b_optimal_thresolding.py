import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('segment.jpg',0)

T_original = 200
T_old = 0
T_final = 204
# while(True):
#     print('inside while')
#     Group_1 = []
#     Group_2 = []
#     #T_old = T_new
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             if img[i,j] > T_old:
#                 Group_1.append(img[i,j])
#             else:
#                 Group_2.append(img[i,j])
#     Mu_1 = sum(Group_1)//len(Group_1)
#     Mu_2 = sum(Group_2)//len(Group_2)
#     T_new = int((Mu_1+Mu_2)/2)
#     print('T_new',T_new)
#     #T_new = int(Mu_1)
#     if abs(T_new-T_old)<T_original:
#         T_final = T_new
#         break
#     T_old = T_new


print("T final", T_final)


hist_dict = {}
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i,j] in hist_dict.keys():
            hist_dict[img[i,j]] = hist_dict.get(img[i,j]) + 1
        else:
            hist_dict[img[i,j]] = 1

key_list = list(hist_dict.keys())
# print(type(key_list))
value_list = list(hist_dict.values())
plt.plot(key_list,value_list,lw=2)
plt.ylabel('Frequency')
plt.xlabel('Intensity')
#plt.show()
plt.savefig('histogram.png')

result_img = np.array([[0 for i in range(img.shape[1])]for j in range(img.shape[0])],dtype='uint8')
count = 0
print(np.max(img))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i,j]>=T_final:
            count += 1
            result_img[i,j] = 255
print("Count",count)
cv2.imshow('Image',result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('thresold.jpg',result_img)

point = []
top = 0
bottom = 0
left = 0
right = 0

# we will form one bounding box around object detected.
# finding corners
flag1 = False
for i in range(result_img.shape[0]):
    for j in range(result_img.shape[1]):
        if result_img[i,j] == 255:
            point.append(i)
            top = i
            flag1 = True
            break
    if flag1:
        break

flag2 = False
for j in range(result_img.shape[1]):
    for i in range(result_img.shape[0]):
        if result_img[i,j] == 255:
            point.append(j)
            left = j
            flag2 = True
            break
    if flag2:
        break

flag3 = False
for i in range(result_img.shape[0]-1,0,-1):
    for j in  range(result_img.shape[1]):
        if result_img[i,j] == 255:
            point.append(i)
            bottom = i
            flag3 = True
            break
    if flag3:
        break

flag4 = False
for j in range(result_img.shape[1]-1,0,-1):
    for i in range(result_img.shape[0]-1,0,-1):
        if result_img[i,j] == 255:
            point.append(j)
            right = j
            flag4 = True
            break
    if flag4:
    	break

print(top)
print(bottom)
print(left)
print(right)

boundary = cv2.rectangle(result_img,(left,top),(right,bottom),(255,255,255),2)
cv2.imshow('boundary img-',boundary)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('boundary.jpg',boundary)


