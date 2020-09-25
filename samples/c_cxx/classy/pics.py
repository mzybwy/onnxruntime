from PIL import Image
import numpy as np

def preprocess(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224), Image.BILINEAR)
    img_data = np.array(img, dtype=np.float32)
    img_data = np.transpose(img_data, [2, 0, 1])
    img_data = np.expand_dims(img_data, 0)
    img_data = img_data[:,0:3,:,:] # Input is RGBA - so strip off A
    return img_data

mushroom = preprocess('cheetah.png')

print "std::vector<float> mushroom = {"
for a in mushroom:
    for b in a:
        for c in b:
            for d in c:
                print "    " + str(d) + ","
print "};"

# x=0
# y=0

# im = Image.open('mushroom.png') # Can be many different formats.
# pix = im.load()
# print im.size  # Get the width and hight of the image for iterating over

# pix[x,y] = value  # Set the RGBA Value of the image (tuple)

# print "std::vector<float> mushroom = {"
# for x in range(im.size[0]):
    # for y in range(im.size[1]):
        # print '    ' + ','.join(map(str,pix[y,x][0:3])) + ','  # Get the RGBA Value of the a pixel of an image
        # pix[x,y] = (0,0,0)
        # pass
# print "};"

