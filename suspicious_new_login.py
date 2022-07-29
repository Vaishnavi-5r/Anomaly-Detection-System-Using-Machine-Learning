from tkinter import *

import sqlite3
import tkinter as tk
from PIL import Image, ImageTk


window = tk.Tk()
window.geometry("1600x1600")
window.title("Login Form")
window.resizable(True,True)


image2 = Image.open('myimg18.jpg')
image2 = image2.resize((1670,780), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(window, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)
def master():
    from subprocess import call
    call(["python","GUI_Master.py"])

#defining login function
def login():
    #getting form data
        uname=username.get()
        pwd=password.get()
    #applying empty validation
        if uname=='' or pwd=='':
            message.set("fill the empty field!!!")
        else:
      #open database
            conn = sqlite3.connect('evaluation.db')
      #select query
        cursor = conn.execute('SELECT * from registration where username="%s" and password="%s"'%(uname,pwd))
        if cursor.fetchone():
            message.set("Login success")
            window.destroy()
            from subprocess import call
            call(['python','GUI_Master.py'])
        else:
            message.set("Wrong username or password!!!")


def Loginform():

    #declaring variable
    global  message;
    global username
    global password
    username = StringVar()
    password = StringVar()
    message=StringVar()

    #Creating layout of login form

    l1=tk.Label(width="300", text="LOGIN FORM", bg="#0E6655",fg="white",font=("Arial Rounded MT Bold",28,"bold")).pack()

    l2=tk.Label(text="Username * ",width=11,bg="white",fg="black",font=("Arial Rounded MT Bold",15,"bold"))
    l2.place(x=560,y=210)
    l2=tk.Entry(textvariable=username,bg="white",fg="black",font=("Arial Rounded MT Bold",15,"bold"))
    l2.place(x=760,y=210)
    l3=tk.Label(text="Password *  ",width=11,bg="white",fg="black",font=("Arial Rounded MT Bold",15,"bold"))
    l3.place(x=560,y=290)
    l3=tk.Entry(textvariable=password ,show="*",bg="white",fg="black",font=("Arial Rounded MT Bold",15,"bold"))
    l3.place(x=760,y=290)
    
    #Label for displaying login status[success/failed]

    l4=tk.Label(text="",width = 0,textvariable=message,fg="green",font=("Arial Rounded MT Bold",15,"bold"))
    l4.place(x=650,y=380)
   
    #Login button
    b1=tk.Button(text="LOGIN", width=10, height=1, command=login, bg="green",fg="black",font=("Arial Rounded MT Bold",15,"bold"))
    b1.place(x=690,y=420)
   
    window.mainloop()


Loginform()