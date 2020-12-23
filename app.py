import streamlit as st
from PIL import Image
import cv2
import numpy as np
import imutils
from skimage.measure import compare_ssim


col1, col2 = st.beta_columns(2)


img1 = st.sidebar.file_uploader("original", type="jpg")#background
img2 = st.sidebar.file_uploader("compare", type="jpg")#mask


original = Image.open(img1)
original = original.resize((640, 480))
col1.header("original")
col1.image(original, use_column_width=True)

grayscale = Image.open(img2)
grayscale = grayscale.resize((640, 480))
col2.header("compare")
col2.image(grayscale, use_column_width=True)


#background
img1 = Image.open(img1)
img1 = img1.resize((640, 480))
img1 = np.array(img1.convert('RGB'))
imageA = cv2.cvtColor(img1,1)




#mask
img2 = Image.open(img2)
img2 = img2.resize((640, 480))
img2 = np.array(img2.convert('RGB'))
imageB = cv2.cvtColor(img2,1)





H, W , _= imageA.shape
# to resize and set the new width and height 
imageB = cv2.resize(imageB, (W, H))

# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)



# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))




# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)






# loop over the contours
for c in cnts:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

# cv2.imshow("Original", imageA)
# cv2.imshow("Modified", imageB)
# cv2.imshow("Diff", diff)
# cv2.imshow("Thresh", thresh)


Original=cv2.resize(imageA, (700, 480))

Modified=cv2.resize(imageB, (700, 480))
diff=cv2.resize(diff, (700, 480))
thresh=cv2.resize(thresh, (700, 480))

st.write(f"Structural Similarity Index: {score}")

st.image(Original,caption="Original")
st.image(Modified,caption="Modified")
st.image(diff,caption="Diff")
st.image(thresh,caption="Thresh")
# cv2.imshow('res',img1)
# cv2.imwrite("result.jpg",img1 )
# cv2.waitKey(10000)
# cv2.destroyAllWindows()


