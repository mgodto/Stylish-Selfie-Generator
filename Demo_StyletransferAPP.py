#!/usr/bin/env python
# coding: utf-8

# In[1]:


from STModel import *


# In[2]:


import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk, ImageOps
import imageio
import cv2
import functools
import tensorflow as tf
import numpy as np
import time
import functools
import IPython.display as display
from pathlib import Path
import random
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
from tqdm import tqdm
import os


# In[16]:


class StyletransferApp(object):
    def __init__(self):
        self.photofilename = ''
        self.stylefilename = ''
        self.img_show = ''
        self.style_show = ''
        self.output_show = ''
        self.img_size = (400,400)
        self.photodirname = CONTENT_DIRS
        self.styledirname = STYLE_DIRS
        self.output = ''
        self.paths = glob.glob(self.styledirname+'/*.jpg')
        
        self.styleTransferModel = ArbitraryStyleTransferNet(img_shape=IMG_SHAPE)
        ckp = './checkpoints/cw1sw10_mscoco/ckpt_50'
        if ckp:
            self.styleTransferModel.load_weights(ckp)
            cur_epoch = int(ckp.split('_')[-1]) + 1
            print(f'ckpt from epoch {cur_epoch-1}')
    
        self.tk = tk.Tk()
        self.tk.geometry('700x500')
        self.tk.configure(background='black')
        self.tk.state('zoomed')
        
        self.title = tk.Label(self.tk, text="Stylish Selfie Generator", font=("@STLiti", 40))
        self.title.place(relx=0.5, rely = 0.05, anchor='c')
        self.title.config(bg='black', fg='white')
        
        self.picture_display = tk.Label(self.tk)
        self.picture_display.place(relx=0.2, rely = 0.6, anchor='c')
        self.picture_display.config(bg='black')
        
        self.style_display = tk.Label(self.tk)
        self.style_display.place(relx=0.5, rely = 0.6, anchor='c')
        self.style_display.config(bg='black')
        
        self.output_display = tk.Label(self.tk)
        self.output_display.place(relx=0.8, rely = 0.6, anchor='c')
        self.output_display.config(bg='black')
        
        self.inputT = tk.Label(self.tk, text="Input image", font=("@STLiti", 30))
        self.inputT.place(relx=0.15, rely = 0.815)
        self.inputT.config(bg='black', fg='white')
        
        self.styleT = tk.Label(self.tk, text="Style image", font=("@STLiti", 30))
        self.styleT.place(relx=0.45, rely = 0.815)
        self.styleT.config(bg='black', fg='white')
        
        self.outputT = tk.Label(self.tk, text="Output image", font=("@STLiti", 30))
        self.outputT.place(relx=0.75, rely = 0.815) 
        self.outputT.config(bg='black', fg='white')
        
        self.random = IntVar()
        self.israndom = tk.Checkbutton(self.tk, text="Random Choice", font=('Arial', 12), variable=self.random,
                                       bg='white', fg='black')
        self.israndom.place(relx=0.46, rely=0.25)
        
    def UploadPhoto(self):
        self.photofilename = tk.filedialog.askopenfilename(initialdir=self.photodirname)
        if self.photofilename != '':
            img = Image.open(self.photofilename)
            r = max(img.size[0]/self.img_size[0], img.size[1]/self.img_size[1])
            size = int(img.size[0]/r), int(img.size[1]/r)
            self.img_show = ImageTk.PhotoImage(img.resize(size, Image.ANTIALIAS))
            self.picture_display.image = self.img_show
            self.picture_display.config(image=self.img_show)
        
    def UploadStyle(self):
        if self.random.get() == 1:
            idx = np.random.randint(len(self.paths))
            self.stylefilename = self.paths[idx]
        else:
            self.stylefilename = tk.filedialog.askopenfilename(initialdir=self.styledirname)
        if self.stylefilename != '':
            img = Image.open(self.stylefilename)
            r = max(img.size[0]/self.img_size[0], img.size[1]/self.img_size[1])
            size = int(img.size[0]/r), int(img.size[1]/r)
            self.style_show = ImageTk.PhotoImage(img.resize(size, Image.ANTIALIAS))
            self.style_display.image = self.style_show
            self.style_display.config(image=self.style_show)
        
    def photo2pencil(self):
        if self.photofilename != '':
            photo = imageio.imread(self.photofilename)
            gray_image = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY) 

            inverted_gray_image = 255 - gray_image
            blurred_img = cv2.GaussianBlur(inverted_gray_image, (21,21),0) 
            inverted_blurred_img = 255 - blurred_img
            pencil_sketch_IMG = cv2.divide(gray_image, inverted_blurred_img, scale = 256.0)
            img = Image.fromarray(cv2.cvtColor(pencil_sketch_IMG, cv2.COLOR_BGR2RGB))
            self.output = img
            r = max(img.size[0]/self.img_size[0], img.size[1]/self.img_size[1])
            size = int(img.size[0]/r), int(img.size[1]/r)
            self.output_show = ImageTk.PhotoImage(img.resize(size, Image.ANTIALIAS))
            self.output_display.image = self.output_show
            self.output_display.config(image=self.output_show)
            
            # Refresh the style image when not using it
            self.style_show = ''
            self.stylefilename = ''
            self.style_display.image = self.style_show
            self.style_display.config(image=self.style_show)
        else:
            tk.messagebox.showwarning("Warning", "Please select photo first!")
            
    def generateStyleTransfer(self):
        #idx = np.random.randint(len(self.paths))
        #self.stylefilename = self.paths[idx]
        if self.photofilename != '' and self.stylefilename != '':
            img_init = np.array(Image.open(self.photofilename))
            ds_test = my_example(self.photofilename, self.stylefilename)
            c_batch, s_batch = next(iter(ds_test.take(1)))
            output, c_enc_c, normalized_c, out_enc_c = self.styleTransferModel((c_batch, s_batch))
            img = np_image(output[0])
            img = crop_img(img_init, img)
            img = Image.fromarray(np.uint8(img)).convert('RGB')
            self.output = img
            r = max(img.size[0]/self.img_size[0], img.size[1]/self.img_size[1])
            size = int(img.size[0]/r), int(img.size[1]/r)
            self.output_show = ImageTk.PhotoImage(img.resize(size, Image.ANTIALIAS))
            self.output_display.image = self.output_show
            self.output_display.config(image=self.output_show)
            #print(self.paths[idx].split('/')[-1])
        else:
            tk.messagebox.showwarning("Warning", "Please select photo and style first!")
     
    def photo2NPR(self):
        if self.photofilename != '':
            photo = imageio.imread(self.photofilename)
            gray_image = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY) 
            edges = cv2.Canny(gray_image,150,400)
            #edges = cv2.Canny(gray_image,100,200)    # min val, max val  
            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)

           
            pixel_values = photo.reshape((-1, 3)) # -1 means to suit the other parameter to guarantee correct size
            pixel_values = np.float32(pixel_values)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

            k = 6
            _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            labels = labels.flatten()
            segmented_image = centers[labels.flatten()]
            segmented_image = segmented_image.reshape(photo.shape)
            
            for i in range(0,segmented_image.shape[0]):
                for j in range(0,segmented_image.shape[1]):
                    if edges[i,j] == 255:
                        segmented_image[i,j] = [70,70,70] # set the line color
                        
            img = Image.fromarray(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
            self.output = img
            
            r = max(img.size[0]/self.img_size[0], img.size[1]/self.img_size[1])
            size = int(img.size[0]/r), int(img.size[1]/r)
            self.NPR_show = ImageTk.PhotoImage(img.resize(size, Image.ANTIALIAS))
            self.output_display.image = self.NPR_show
            self.output_display.config(image=self.NPR_show)
    
            # Refresh the style image when not using it
            self.style_show = ''
            self.stylefilename = ''
            self.style_display.image = self.style_show
            self.style_display.config(image=self.style_show)
        else:
            tk.messagebox.showwarning("Warning", "Please select photo first!")
        
    def Download(self):
        dirname = tk.filedialog.asksaveasfilename(defaultextension=".jpg",initialdir='./', filetypes=[('.jpg','*.jpg'), ('.png','*.png')])
        print(dirname)
        if self.output != '':
            self.output.save(dirname)
    def EnableButton(self):
        self.button_photo = tk.Button(self.tk, text='Select a photo', width=15, font=('Arial', 12),
                                      command=functools.partial(self.UploadPhoto))
        self.button_style = tk.Button(self.tk, text='Select a style', width=15, font=('Arial', 12),
                                      command=functools.partial(self.UploadStyle))
        self.button_styleTransfer = tk.Button(self.tk, text='Generate style transfer photo', width=25, font=('Arial', 12),
                                               command=functools.partial(self.generateStyleTransfer))
        self.button_pencil = tk.Button(self.tk, text='Generate pencil sketch draw', width=25, font=('Arial', 12),
                                       command=functools.partial(self.photo2pencil))
        self.button_NPR = tk.Button(self.tk, text='Generate NPR draw', width=25, font=('Arial', 12),
                                    command=functools.partial(self.photo2NPR))

        self.button_download = tk.Button(self.tk, text='Download output!', width=15, font=('Arial', 12),
                                              command=functools.partial(self.Download))
        
        self.button_photo.config(bg='white')
        self.button_style.config(bg='white')
        self.button_pencil.config(bg='white')
        self.button_styleTransfer.config(bg='white')
        self.button_NPR.config(bg='white')
        self.button_download.config(bg='white')
        
        self.button_photo.place(relx=0.16, rely=0.3)
        self.button_style.place(relx=0.46, rely=0.3)
        self.button_styleTransfer.place(relx=0.73, rely=0.3)
        self.button_pencil.place(relx=0.73, rely=0.25)
        self.button_NPR.place(relx=0.73, rely=0.2)
        self.button_download.place(relx=0.77, rely=0.9)






