#-------------------------------------------------------------------------------
# Name:        Object bounding box label tool
# Purpose:     Label object bboxes for ImageNet Detection data
# Author:      Qiushi
# Modified:    Michael
# Created:     06/06/2014
# Date:        27/04/2020
#
#-------------------------------------------------------------------------------

# To compile python BBox_Label_Tool.py

from __future__ import division
from Tkinter import *
import tkMessageBox
from PIL import Image, ImageTk
import os
import glob
import random
import struct
import imghdr
import PIL

# colors for the bboxes
COLORS = ['red', 'blue', 'yellow', 'pink', 'cyan', 'green', 'black']
#classes = ['car','plate','keke','bus','motocycle','lorry']
classes = ['human','face','mask','no_mask','fore_head']
number_class = [0,1,2,3,4,5]
# image sizes for the examples
SIZE = 256, 256

class LabelTool():
    def __init__(self, master):
        # set up the main frame
        self.parent = master
        self.parent.title("LabelTool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width = FALSE, height = FALSE)

        # initialize global state
        self.imageDir = ''
        self.imageList= []
        self.egDir = ''
        self.egList = []
        self.outDir = ''
        self.outDir1='' #############################
        self.cur = 0
        self.total = 0
        self.category = 0
        self.imagename = ''
        self.labelfilename = ''
        self.labelfilename1 = '' #######################
        self.tkimg = None
        self.name = 'ffff'
        self.dClass = []
        

        # initialize mouse state
        self.STATE = {}
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0

        # reference to bbox
        self.bboxIdList = []
        self.bboxId = None
        self.bboxList = []
        self.bboxList1 = []  ######################
        self.hl = None
        self.vl = None

        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.label = Label(self.frame, text = "Image Dir:")
        self.label.grid(row = 0, column = 0, sticky = E)
        self.entry = Entry(self.frame)
        self.entry.grid(row = 0, column = 1, sticky = W+E)
        self.ldBtn = Button(self.frame, text = "Load", command = self.loadDir)
        self.ldBtn.grid(row = 0, column = 2, sticky = W+E)


        # main panel for labeling
        self.mainPanel = Canvas(self.frame, cursor='tcross')
        self.mainPanel.bind("<Button-1>", self.mouseClick)
        self.mainPanel.bind("<Motion>", self.mouseMove)
        self.parent.bind("<Escape>", self.cancelBBox)  # press <Espace> to cancel current bbox
        self.parent.bind("s", self.cancelBBox)
        #self.parent.bind("a", self.prevImage) # press 'a' to go backforward
        #self.parent.bind("d", self.nextImage) # press 'd' to go forward
        self.mainPanel.grid(row = 1, column = 1, rowspan = 4, sticky = W+N)

        # showing bbox info & delete bbox
        self.lb1 = Label(self.frame, text = 'Bounding boxes:')
        self.lb1.grid(row = 1, column = 2,  sticky = W+N)
        self.listbox = Listbox(self.frame, width = 22, height = 12)
        self.listbox.grid(row = 2, column = 2, sticky = N)
        self.btnDel = Button(self.frame, text = 'Delete', command = self.delBBox)
        self.btnDel.grid(row = 3, column = 2, sticky = W+E+N)
        self.btnClear = Button(self.frame, text = 'ClearAll', command = self.clearBBox)
        self.btnClear.grid(row = 4, column = 2, sticky = W+E+N)
        self.lb5 = Label(self.frame, text = self.name)
        self.lb5.grid(row = 4, column = 0,  sticky = N+E)
        

        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row = 5, column = 1, columnspan = 2, sticky = W+E)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width = 10, command = self.prevImage)
        self.prevBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.nextBtn = Button(self.ctrPanel, text='Next >>', width = 10, command = self.nextImage)
        self.nextBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.progLabel = Label(self.ctrPanel, text = "Progress:     /    ")
        self.progLabel.pack(side = LEFT, padx = 5)
        self.tmpLabel = Label(self.ctrPanel, text = "Go to Image No.")
        self.tmpLabel.pack(side = LEFT, padx = 5)
        self.idxEntry = Entry(self.ctrPanel, width = 5)
        self.idxEntry.pack(side = LEFT)
        self.goBtn = Button(self.ctrPanel, text = 'Go', command = self.gotoImage)
        self.goBtn.pack(side = LEFT)

        # example pannel for illustration
        self.egPanel = Frame(self.frame, border = 10)
        self.egPanel.grid(row = 1, column = 0, rowspan = 5, sticky = N)
        self.tmpLabel2 = Label(self.egPanel, text = "Examples:")
        self.tmpLabel2.pack(side = TOP, pady = 5)
        self.egLabels = []
        for i in range(3):
            self.egLabels.append(Label(self.egPanel))
            self.egLabels[-1].pack(side = TOP)

        # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side = RIGHT)

        self.frame.columnconfigure(1, weight = 1)
        self.frame.rowconfigure(4, weight = 1)

        # for debugging
##        self.setImage()
##        self.loadDir()

    def loadDir(self, dbg = False):
        if not dbg:
            s = self.entry.get()
            self.parent.focus()
            self.category = (s)
        else:
            s = r'D:\workspace\python\labelGUI'
##        if not os.path.isdir(s):
##            tkMessageBox.showerror("Error!", message = "The specified dir doesn't exist!")
##            return
        # get image list
        self.imageDir = os.path.join(r'./images/Face_mask')
        print(self.imageDir)
        self.imageList = sorted(glob.glob(os.path.join(self.imageDir, '*.jpg')))
        print(os.path.join(self.imageDir, '*.jpg'))
        print(self.imageList)
        if len(self.imageList) == 0:
            print ('No .jpg images found in the specified dir!')
            return

        # default to the 1st image in the collection
        self.cur = 1
        self.total = len(self.imageList)
        
         # Labels for raw data
         # set up output dir
        self.outDir = os.path.join(r'./images/FACE_MASK_LABEL') #'%03d' %(self.category)
        if not os.path.exists(self.outDir):
            os.mkdir(self.outDir)
            
        # For labels
        # Create modified directory
        self.outDir1 = os.path.join(r'./images/labels') #'%03d' %(self.category)
        if not os.path.exists(self.outDir1):
            os.mkdir(self.outDir1)
        

        # load example bboxes
        self.egDir = os.path.join(r'./Examples1')
        #if not os.path.exists(self.egDir):
        #    return
        filelist = glob.glob(os.path.join(self.egDir,'*.jpg'))
        self.tmp = []
        self.egList = []
        random.shuffle(filelist)
        for (i, f) in enumerate(filelist):
            if i == 3:
                break
            im = Image.open(f)
            r = min(SIZE[0] / im.size[0], SIZE[1] / im.size[1])
            new_size = int(r * im.size[0]), int(r * im.size[1])
            self.tmp.append(im.resize(new_size, Image.ANTIALIAS))
            self.egList.append(ImageTk.PhotoImage(self.tmp[-1]))
            self.egLabels[i].config(image = self.egList[-1], width = SIZE[0], height = SIZE[1])

        self.loadImage()
        print '%d images loaded from %s' %(self.total, s)

    def loadImage(self):
        # load image
        imagepath = self.imageList[self.cur - 1]
        self.name = imagepath
        self.lb5['text'] = imagepath
        self.img = Image.open(imagepath)
        #print(imagepath)
        #total_size = self.get_image_size(imagepath)
        #print(total_size[0])
        #print(total_size[1])
        total_size = self.get_image_size1(imagepath)
        print(total_size[0])
        print(total_size[1])
        self.tkimg = ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width = max(self.tkimg.width(), 400), height = max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)
        self.progLabel.config(text = "%04d/%04d" %(self.cur, self.total))

        # load labels
        self.clearBBox()
        self.imagename = os.path.split(imagepath)[-1].split('.')[0]
        labelname = self.imagename + '.txt'
        self.labelfilename = os.path.join(self.outDir, labelname)
        self.labelfilename1 = os.path.join(self.outDir1, labelname)
        wr = open(self.labelfilename1, 'r+') 
        wr.truncate(0) # need '0' when using r+
        wr.close()
        bbox_cnt = 0
        if os.path.exists(self.labelfilename):
            with open(self.labelfilename) as f:
                for (i, line) in enumerate(f):
                    if i == 0:
                        bbox_cnt = int(line.strip())
    
                        continue
                    tmp = [int(t.strip()) for t in line.split()]
                    #print(tmp)
                    cordinate =  self.xy_center(tmp[0],tmp[2],tmp[1],tmp[3])
                    print (cordinate)
                    process_cordinate =  self.get_cordinates(cordinate[0],cordinate[1],cordinate[2],cordinate[3],total_size[0],total_size[1])
                    self.bboxList1.append(tuple(process_cordinate))
                    print (process_cordinate)
##                    print tmp
                    
                    self.bboxList.append(tuple(tmp))
                    tmpId = self.mainPanel.create_rectangle(tmp[0], tmp[1], \
                                                            tmp[2], tmp[3], \
                                                            width = 2, \
                                                            outline = COLORS[(len(self.bboxList)-1) % len(COLORS)])
                    self.bboxIdList.append(tmpId)
                    self.listbox.insert(END, '(%d, %d) -> (%d, %d)' %(tmp[0], tmp[1], tmp[2], tmp[3]))
                    self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])

    def saveImage(self):
        self.saveImageModified()
        with open(self.labelfilename, 'w') as f:
            #print(self.labelfilename)
            f.write('%d\n' %len(self.bboxList))
            for bbox in self.bboxList:
                #print(len(self.bboxList))
                #print(bbox)
                f.write(' '.join(map(str, bbox)) + '\n')
        print 'Image No. %d saved' %(self.cur)


    def saveImageModified(self):
        print( self.entry.get())
        category = self.entry.get()
        self.dClass = category.split(',')
        i = 0
        with open(self.labelfilename1, 'w') as f:
            #f.write('%d ' %(classes.index(self.entry.get())))
            for bbox in self.bboxList1:
                #print(len(self.bboxList1))
                if i < len(self.bboxList1):
                    catgory = (self.dClass[i])
                    print(type(catgory))
                    print(catgory == "car")
                    catgory_1 =  self.get_index_number_mask(catgory)
                    #print(self.dClass.index(catgory_1))
                    f.write('%d ' %catgory_1)
                    f.write(' '.join(map(str, bbox)) + '\n')
                    i += 1
        print 'Image No. %d saved' %(self.cur)


    def mouseClick(self, event):
        if self.STATE['click'] == 0:
            self.STATE['x'], self.STATE['y'] = event.x, event.y
        else:
            x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
            y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
            self.bboxList.append((x1, y1, x2, y2))
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            self.listbox.insert(END, '(%d, %d) -> (%d, %d)' %(x1, y1, x2, y2))
            self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
        self.STATE['click'] = 1 - self.STATE['click']

    def mouseMove(self, event):
        self.disp.config(text = 'x: %d, y: %d' %(event.x, event.y))
        if self.tkimg:
            if self.hl:
                self.mainPanel.delete(self.hl)
            self.hl = self.mainPanel.create_line(0, event.y, self.tkimg.width(), event.y, width = 2)
            if self.vl:
                self.mainPanel.delete(self.vl)
            self.vl = self.mainPanel.create_line(event.x, 0, event.x, self.tkimg.height(), width = 2)
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
            self.bboxId = self.mainPanel.create_rectangle(self.STATE['x'], self.STATE['y'], \
                                                            event.x, event.y, \
                                                            width = 2, \
                                                            outline = COLORS[len(self.bboxList) % len(COLORS)])

    def cancelBBox(self, event):
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0

    def delBBox(self):
        sel = self.listbox.curselection()
        if len(sel) != 1 :
            return
        idx = int(sel[0])
        self.mainPanel.delete(self.bboxIdList[idx])
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.listbox.delete(idx)

    def clearBBox(self):
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
        self.listbox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []
        self.bboxList1 = []

    def prevImage(self, event = None):
        self.saveImage()
        if self.cur > 1:
            self.cur -= 1
            self.loadImage()

    def nextImage(self, event = None):
        self.saveImage()
        if self.cur < self.total:
            self.cur += 1
            self.loadImage()

    def gotoImage(self):
        idx = int(self.idxEntry.get())
        if 1 <= idx and idx <= self.total:
            self.saveImage()
            self.cur = idx
            self.loadImage()

##    def setImage(self, imagepath = r'test2.png'):
##        self.img = Image.open(imagepath)
##        self.tkimg = ImageTk.PhotoImage(self.img)
##        self.mainPanel.config(width = self.tkimg.width())
##        self.mainPanel.config(height = self.tkimg.height())
##        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)

    def get_image_size(self,fname):
        # Determine the kind of handle and return its size.
        with open(fname, 'rb') as fhandle:
            head = fhandle.read(24)
            if len(head) != 24:
                return
            if imghdr.what(fname) == 'png':
                check = struct.unpack('>i', head[4:8])[0]
                if check != 0x0d0a1a0a:
                    return
                width, height = struct.unpack('>ii', head[16:24])
            elif imghdr.what(fname) == 'gif':
                width, height = struct.unpack('<HH', head[6:10])
            elif imghdr.what(fname) == 'jpeg':
                try:
                    fhandle.seek(0)  # Read 0xff next
                    size = 2
                    ftype = 0
                    while not 0xc0 <= ftype <= 0xcf:
                        fhandle.seek(size,1)
                        byte = fhandle.read(1)
                        while ord(byte) == 0xff:
                            byte = fhandle.read(1)
                        ftype = ord(byte)
                        size = struct.unpack('>H', fhandle.read(2))[0] - 2
                    # we are at a SOFn block
                    fhandle.seek(1,1) # Skip precision byte
                    height, width = struct.unpack('>HH', fhandle.read(4))
                except Exception: #IGNORE: Wo703
                    return
            else:
                return
            return width, height


    def xy_center(self,x_min,x_max,y_min,y_max):
        w = x_max + x_min
        h = y_max + y_min
        x_center = w/2
        y_center = h/2
        w = x_max - x_min
        h = y_max - y_min
        return x_center, y_center,w,h

    def get_cordinates(self,x_center,y_center,w,h,width,height):
        x_center = x_center/width
        y_center = y_center/height
        w = w/width
        h = h/height
        print('Start here to check for cordinates')
        print(x_center)
        print(y_center)
        print(w)
        print(h)
        print(height)
        return x_center, y_center, w, h

    def get_classtype(self,classes):
        x = [number_classes[index]  for index in classes]
        return x

    def get_index_number(self, args):
        index_no = -1
        if args == "car"  :
            print("index no 1 %s", args )
            index_no = 0
        elif args == "plate" :
            print("index no 2 %s", args )
            index_no = 1
        elif args == "keke" :
            print("index no 3 %s", args )
            index_no = 2
        elif args == "bus" :
            print("index no 4 %s", args )
            index_no = 3
        
        return index_no

    def get_index_number_mask(self, args):
        index_no = -1
        if args == "human"  :
            print("index no 1 %s", args )
            index_no = 0
        elif args == "face" :
            print("index no 2 %s", args )
            index_no = 1
        elif args == "mask" :
            print("index no 3 %s", args )
            index_no = 2
        elif args == "no_mask" :
            print("index no 3 %s", args )
            index_no = 3
        elif args == "fore_head" :
            print("index no 4 %s", args )
            index_no = 4
        
        return index_no

    def get_image_size1(self, args):
        image = PIL.Image.open(args)
        width, height = image.size
        return width, height
                
                                 
if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    root.resizable(width =  True, height = True)
    root.mainloop()
