import matplotlib
from matplotlib import pylab as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

file = './data/FLARE22Train/imagesTr/FLARE22_Tr_0001_0000.nii.gz'  # 你的nii或者nii.gz文件路径
img = nib.load(file)

print(img)
print(img.header['db_name'])  # 输出nii的头文件
width, height, queue = img.dataobj.shape
# OrthoSlicer3D(img.dataobj).show()

num = 1
for i in range(0, queue, 10):
    img_arr = img.dataobj[:, :, i]
    plt.subplot(5, 4, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1

plt.show()