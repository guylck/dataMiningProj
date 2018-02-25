import PIL
from PIL import Image
from os import listdir
from os.path import isfile, join

baseWidth = 32
srcDir = './Resources/colored'
dstDir = './Resources/colored32'
files = [f for f in listdir(srcDir) if isfile(join(srcDir, f))]

for index in range(len(files)):
    myImage = Image.open(join(srcDir, files[index]))
    wpercent = (baseWidth / float(myImage.size[0]))
    hsize = int((float(myImage.size[1]) * float(wpercent)))
    myImage = myImage.resize((baseWidth, hsize), PIL.Image.ANTIALIAS)
    myImage.save(join(dstDir, files[index]))

