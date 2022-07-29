import tkinter as tk ##GUI library which provides fast and easy way to create gui appln
# powerful object orientation interface to tk gui toolkit 
from PIL import Image , ImageTk# used to create nd modify bitmap images from PIL images.
import csv # CSV= Cooma Seprated Vector which stores information in spreadsheet.
from datetime import date
import time
import numpy as np
import cv2
from tkinter.filedialog import askopenfilename # returns file name that u selected.
import os
import shutil# it copies entire directory tree

root = tk.Tk()
root.state('zoomed')

root.title("Anomally Activity Detection")

current_path = str(os.path.dirname(os.path.realpath('__file__')))

basepath=current_path  + "\\" 



img = Image.open(basepath + "back5.jpg")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()

bg = img.resize((w,h),Image.ANTIALIAS)

bg_img = ImageTk.PhotoImage(bg)

bg_lbl = tk.Label(root,image=bg_img)
bg_lbl.place(x=0,y=0)

heading = tk.Label(root,text="Anomally Activity Detection",width=25,font=("Times New Roman",45,'bold'),bg="#192841",fg="white")
heading.place(x=290,y=0)


def create_folder(FolderN):
    
    dst=os.getcwd() + "\\" + FolderN         # destination to save the images
    
    if not os.path.exists(dst):
        os.makedirs(dst)
    else:
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(dst)


def CLOSE():
    root.destroy()
    
def update_label(str_T):
   
    result_label = tk.Label(root, text=str_T, width=50, font=("bold", 25),bg='cyan',fg='black' )
    result_label.place(x=400, y=400)

def train_model():
    Train = ""
    update_label("Model Training Start...............")
    
    start = time.time()

    X=Train.main()
    
    end = time.time()
        
    ET="Execution Time: {0:.4} seconds \n".format(end-start)
    
    msg="Model Training Completed.."+'\n'+ X + '\n'+ ET

    update_label(msg)



def run_video(VPathName,XV,YV,S1,S2):

    cap = cv2.VideoCapture(VPathName)

    def show_frame():
                    
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_FPS, 30)
               
        out=cv2.transpose(frame)
    
        out=cv2.flip(out,flipCode=0)
    
        cv2image   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        img   = Image.fromarray(cv2image).resize((S1, S2))
    
        imgtk = ImageTk.PhotoImage(image = img)
        
        lmain = tk.Label(root)

        lmain.place(x=XV, y=YV)

        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame)
                
                
    show_frame()
        
def VIDEO():
    
    global fn
    
    fn=""
    fileName = askopenfilename(initialdir='/dataset', title='Select image',
                               filetypes=[("all files", "*.*")])

    fn = fileName
    Sel_F = fileName.split('/').pop()
    Sel_F = Sel_F.split('.').pop(1)
            
    if Sel_F!= 'mp4':
        print("Select Video .mp4 File!!!!!!")
    else:
        run_video(fn,560, 190,753, 485)

                
  
     
def F2V(VideoN):
    

    Video_Fname=F2V.Create_Video(basepath + 'result',VideoN)
    run_video(Video_Fname,560, 190,753, 485)
    print(Video_Fname)


def show_FDD_video(video_path):
    ''' Display FDD video with annotated bounding box and labels '''
    from keras.models import load_model
    
    img_cols, img_rows = 64,64
    
    FALLModel=load_model('D://Project//suspicios_activity//abnormalevent.h5')    #video = cv.VideoCapture(video_path);
    
    video = cv2.VideoCapture(video_path)
        
    

    if (not video.isOpened()):
        print("{} cannot be opened".format(video_path))
        # return False

    font = cv2.FONT_HERSHEY_SIMPLEX
    green = (0, 255, 0) 
    red = (0, 0, 255)
   
    line_type = cv2.LINE_AA
    i=1
    
    while True:
        ret, frame = video.read()
        
        if not ret:
            break
        img=cv2.resize(frame,(img_cols, img_rows),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(img)
     #reshape = it transforms the tensor ie; image from 2D to 1D. so that the metrix can be reperesented by array.
     # #astype = allows us to convert or cast entire datatype with existing data column
    #conversion of data takes place as float then targets to int.   
        X_img = img.reshape(-1, img_cols, img_rows, 1)
        X_img = X_img.astype('float32')
        #pixels are represented in array of no.s whose values ranges from [0,255]
    #so it should be scaled to values of type (float32) within [0,1] interval.
        X_img /= 255
        
        predicted =FALLModel.predict(X_img)

        if predicted[0][0] < 0.5:
            predicted[0][0] = 0
            predicted[0][1] = 1
            label = 1
        else:
            predicted[0][0] = 1
            predicted[0][1] = 0
            label = 0
          
        frame_num = int(i)  
        label_text = ""
        
        color = (255, 255, 255)
        
        if  label == 1 :
            label_text = "Accident Detected"
            color = red
        else:
            label_text = "Not Detected"
            color = green

        frame = cv2.putText(
            frame, "Frame: {}".format(frame_num), (5, 30),
            fontFace = font, fontScale = 1, color = color, lineType = line_type
        )
        frame = cv2.putText(
            frame, "Label: {}".format(label_text), (5, 60),
            fontFace = font, fontScale =1, color = color, lineType = line_type
        )

        i=i+1
        cv2.imshow('FDD', frame)
        if cv2.waitKey(30) == 27:
            break

    video.release()
    cv2.destroyAllWindows()
       
###################################################################################################################  
def Video_Verify():
    
    global fn
    
    fileName = askopenfilename(initialdir='/dataset', title='Select image',
                               filetypes=[("all files", "*.*")])

    fn = fileName
    Sel_F = fileName.split('/').pop()
    Sel_F = Sel_F.split('.').pop(1)
            
    if Sel_F!= 'mp4':
        print("Please Select MP4 format Video File!!!!!!")
    else:
        
        show_FDD_video(fn)
    
            

button5 = tk.Button(root,command = Video_Verify, text="Upload Video", width=20,font=("Arial Rounded MT Bold", 25, "bold"),bg="Silver",fg="black")
button5.place(x=550,y=290)

close = tk.Button(root,command = CLOSE, text="Exit", width=20,font=("Arial Rounded MT Bold", 25, "bold"),bg="red",fg="black")
close.place(x=550,y=460)


root.mainloop()






