#GUI library which provides fast and easy way to create gui appln
# powerful object orientation interface to tk gui toolkit 
import tkinter as tk 

from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox as ms
#import cv2
import sqlite3
import os
import numpy as np
import time
import random


global fn
fn = ""

root = tk.Tk()
root.configure(background="#8E8E8E")



w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Anomally Activity Detection ")

image2 = Image.open('new5.jpg')
image2 = image2.resize((1530, 900), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=70)
label_l1 = tk.Label(root, text="Anomally Activity Detection ",font=("Arial Rounded MT Bold", 37, 'bold'),
                    background="#8E8E8E", fg="#090001")
label_l1.place(x=500, y=0)


def reg():
    from subprocess import call
    call(["python","suspicious_registration.py"])

def log():
    from subprocess import call
    call(["python","suspicious_new_login.py"])
def master():
  from subprocess import call
  call(["python","GUI_Master.py"])

   
def window():
  root.destroy()


button1 = tk.Button(root, text="LOGIN", command=log, width=14, height=1,font=('Arial Rounded MT Bold', 20, ' bold '), bg="#FFEBCD", fg="brown")
button1.place(x=650, y=200)

button2 = tk.Button(root, text="REGISTER",command=reg, width=14, height=1,font=('Arial Rounded MT Bold', 20, ' bold '), bg="#FFEBCD", fg="brown")
button2.place(x=650, y=310)

root.mainloop()