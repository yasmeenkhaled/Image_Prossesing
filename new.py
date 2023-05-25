import tkinter as tk
from tkinter import *
import cv2
from tkinter import filedialog
from tkinter import colorchooser
from PIL import Image, ImageOps, ImageTk, ImageFilter
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
root = Tk()    
root.config(bg="#29738e")
left_frame = tk.Frame(root, width=260, height=300, bg="#29738e")
left_frame.pack(side="left", fill="y")

canvas = tk.Canvas(root, width=700, height=600)
canvas.pack( side=TOP,padx=10, pady=10)
scale_frame = tk.Frame(root,bd=20, width=600, height=100, bg="#29738e")
scale_frame.pack(anchor=CENTER)
display_frame = tk.Frame(root,bd=10, width=600, height=100, bg="#29738e")
display_frame.pack(anchor=CENTER)
canvas.config(bg="#FFFFFF")
variable_scale = Scale(scale_frame,fg="#FFFFFF", border=8,length=550,width=15, bg="#1e4452",troughcolor="#d6eaf0",command=COMMAND, resolution=1,from_ =1, to = 255, orient = HORIZONTAL)  
variable_scale.pack(side=TOP)
Display = tk.Button(display_frame,fg="#FFFFFF",text = "Display", height=2,width=20, bg="#1e4452" )
Display.pack(side=RIGHT)   
inputtxt = tk.Text(display_frame, height =2, width = 20)
inputtxt.pack(side=RIGHT)
variable_scale.set(127)


def getInput_txt():
    global inp
    inp = inputtxt.get(1.0, "end-1c")
    


#a4cad7
#b1deee

# Define the image processing functions
def grayscale():
    clear_canvas()
    global img
    img = original_img.copy()
    new_im = (0.3*img[: , : , 0])+(0.59*img[: , : , 1])+(0.11*img[: , : , 2])
    new_im = new_im.astype(np.uint8) 
    img=new_im
    display_image()


def hist():
    clear_canvas()
    global img,hist_original
    hist_original = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist_original)
    plt.title('Original Histogram')
    plt.show()
    display_image()

def brightness():
    clear_canvas()
    

    global img
    img = original_img.copy()
    offset=100
    row,col,ch=img.shape
    for k in range(ch):
     for i in range(row):
      for j in range(col):
              new_val=img[i,j,k]+offset
              new_val=max(0,min(255,new_val))
              img[i,j,k]=new_val
              
              
    display_image()

def power_low():
    clear_canvas()
    
    global img,gamma_transform
    img = original_img.copy()
    gamma=0.5
    gamma_transform = np.array(255*(img / 255) ** gamma, dtype = 'uint8')
    img=gamma_transform
    display_image()



def hist_equalization():
    clear_canvas()
    global img,img_eq,hist_equalized
    img = original_img.copy()
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(img)
    img=img_eq
    display_image()
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist_equalized)
    plt.title('equalized Histogram')
    plt.show()

def import_target_img():
    
    global target
    # Open a file dialog to get the image file path
    file_path = filedialog.askopenfilename()
    target= cv2.imread(file_path)
    h, w, _ = target.shape
    if h > w:
        scale_factor = canvas.winfo_height() / h
    else:
        scale_factor = canvas.winfo_width() / w
    target = cv2.resize(target, (int(w * scale_factor), int(h * scale_factor)))
    # Load the image file using OpenCV
    
   


def hist_matching():
    clear_canvas()
    global img , target ,matched_image
    img = original_img.copy()
    import_target_img()
    img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    target_hist = cv2.calcHist([target], [0], None, [256], [0, 256])

    # Normalize the histograms to have equal areas
    img_hist = cv2.normalize(img_hist, img_hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    target_hist = cv2.normalize(target_hist, target_hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Calculate the cumulative distribution functions (CDFs) of the histograms
    source_cdf = np.cumsum(img_hist)
    target_cdf = np.cumsum(target_hist)

    # Normalize the CDFs to have equal ranges
    source_cdf = cv2.normalize(source_cdf, source_cdf, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    target_cdf = cv2.normalize(target_cdf, target_cdf, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Create a lookup table that maps each source intensity value to its corresponding target intensity value
    lookup_table = np.zeros((256, 1), dtype=np.uint8)
    for i in range(256):
        j = 0
        while j < 256 and target_cdf[j] < source_cdf[i]:
            j += 1
        lookup_table[i] = j

    matched_image = cv2.LUT(img, lookup_table)
    img=matched_image
    display_image()
    hist_matching = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist_matching)
    plt.title('matched Histogram')
    plt.show()


def merge_image():
    clear_canvas()
    global img , target 
    img = original_img.copy()
    import_target_img()
    merged_img = cv2.add(img,target)
    
    img=merged_img
    display_image()

def subtract_image():
    clear_canvas()
    global img , target 
    img = original_img.copy()
    import_target_img()
    subtracted_img = cv2.subtract(img,target)
    
    img=subtracted_img
    display_image()


def negative_image():
    clear_canvas()
    global img
    img = original_img.copy()
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Compute the negative of the image by subtracting each channel from 255
    negative_array = 255 - img_array
    # Convert the numpy array back to an image
    negative_img = Image.fromarray(negative_array)
    img=negative_array
    display_image()


def Quantization(variable = variable_scale.get()):
    clear_canvas()
    global img 
    variable_scale.config(var=1,command=Quantization)
    img = original_img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    levels = int(variable)
    gap = 256 /levels 
    img_quantized = (img /gap).astype(int) * gap
    img=img_quantized
    display_image()


def Average_Filter():
    clear_canvas()
    global img
    img = original_img.copy()
    kernel_size = 9
    pad_size = kernel_size // 2
    border_img = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)
    # Create new image
    new_im = np.zeros_like(img)
    for k in range(img.shape[2]):
      for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel_sum = 0
            for x in range(i, i+kernel_size):
                for y in range(j, j+kernel_size):
                    pixel_sum += border_img[x, y, k]
            new_im[i, j, k] = pixel_sum / (kernel_size ** 2)
    img=new_im
    display_image()

def unSharpening():
    clear_canvas()
    global img
    img = original_img.copy()
    img=img.astype(np.float32)
    smooth_img = np.zeros_like(img)
    sigma = 2
    N = int(np.floor(3.7*sigma-0.5))
    mask_size = 2*N+1
    x = np.zeros((mask_size,))
    x[0] = -np.floor(mask_size/2)
    for k in range(1, mask_size):
       x[k] = x[k-1] + 1
    mask = np.zeros((mask_size, mask_size))
    for i in range(mask_size):
        for j in range(mask_size):
           mask[i, j] = np.exp(-(x[i]**2+x[j]**2)/(2*sigma**2))
    mask = mask / np.sum(mask)

    # ser border
    pad_size = mask_size // 2
    border_img = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)

    # apply gaussian filter with segma of 2
    for k in range(img.shape[2]):
       for i in range(img.shape[0]):
         for j in range(img.shape[1]):
            pixel_sum = 0
            for x in range(i, i+mask_size):
                for y in range(j, j+mask_size):
                    pixel_sum += border_img[x, y, k] * mask[x-i, y-j]
            smooth_img[i, j, k] = pixel_sum 

    # get edge image 
    edge_img =img - smooth_img

    # cut negative value
    for i in range(img.shape[0]):
       for j in range(img.shape[1]):
          for k in range(img.shape[2]):
            if edge_img[i, j, k] < 0:
               edge_img[i, j, k] = 0

    # get unsharping image
    unsharping = img + edge_img

    # cut over 255 value
    for i in range(img.shape[0]):
      for j in range(img.shape[1]):
        for k in range(img.shape[2]):
            if unsharping[i, j, k] > 255:
                unsharping[i, j, k] = 255

    # convert to uint8 to display
    unsharping = unsharping.astype(np.uint8)
    img = img.astype(np.uint8)
    edge_img = edge_img.astype(np.uint8)
    img=unsharping

    display_image()

def Sharpening():
    clear_canvas()
    global img
    img = original_img.copy()
    # initialize mask and new image
    img=img.astype(np.float32)
    mask = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    new_im = np.zeros_like(img)

    # set padding size and pad image with border
    border_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    # apply filter
    for k in range(img.shape[2]):
       for i in range(img.shape[0]):
         for j in range(img.shape[1]):
            pixel_sum = 0
            for x in range(i, i+3):
                for y in range(j, j+3):
                    pixel_sum += border_img[x, y, k] * mask[x-i, y-j]
            new_im[i, j, k] = pixel_sum 

    # cut off values that are out of range
    for k in range(img.shape[2]):
      for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if new_im[i, j, k] < 0:
                new_im[i, j, k] = 0
            elif new_im[i, j, k] > 255:
                new_im[i, j, k] = 255

    # convert back to unsigned 8-bit integer format to display
    new_im = new_im.astype(np.uint8)
    img = img.astype(np.uint8)
    img=new_im

    display_image()

def edge_detection():
    clear_canvas()
    global img
    img = original_img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    new_im = np.zeros_like(img)

    # initialize mask and new image
    mask = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    # set padding size and pad image with border
    border_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    # apply filter
    for i in range(img.shape[0]):
      for j in range(img.shape[1]):
        pixel_sum = 0
        for x in range(i, i+3):
            for y in range(j, j+3):
                pixel_sum += border_img[x, y] * mask[x-i, y-j]
        new_im[i, j] = pixel_sum

    # cut off values that are out of range
    for i in range(img.shape[0]):
      for j in range(img.shape[1]):
        if new_im[i, j] < 0:
            new_im[i, j] = 0
        elif new_im[i, j] > 255:
            new_im[i, j] = 255

    # convert to binary image
    max = 255
    threshold = 128
    new_im = cv2.threshold(new_im, threshold, max, cv2.THRESH_BINARY)[1]

    # convert to unsigned 8-bit integer format to display
    new_im = new_im.astype(np.uint8)
    img = img.astype(np.uint8)
    img=new_im
    display_image()

def Gaussian():
    clear_canvas()
    global img
    img = original_img.copy()
    img=img.astype(np.float32)
    new_im = np.zeros_like(img)

    # Calculate filter mask based on sigma
    sigma = 2
    N = int(np.floor(3.7*sigma-0.5))
    mask_size = 2*N+1
    x = np.zeros((mask_size,))
    x[0] = -np.floor(mask_size/2)
    for k in range(1, mask_size):
       x[k] = x[k-1] + 1
    mask = np.zeros((mask_size, mask_size))
    for i in range(mask_size):
       for j in range(mask_size):
         mask[i, j] = np.exp(-(x[i]**2+x[j]**2)/(2*sigma**2))


    # set padding size and pad image with border
    pad_size = mask_size // 2
    border_img = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)

    # apply filter
    for k in range(img.shape[2]):
      for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel_sum = 0
            for x in range(i, i+mask_size):
                for y in range(j, j+mask_size):
                    pixel_sum += border_img[x, y, k] * mask[x-i, y-j]
            new_im[i, j, k] = pixel_sum / np.sum(mask)

    # convert back to unsigned 8-bit integer format to display
    new_im = new_im.astype(np.uint8)
    img = img.astype(np.uint8)
    img=new_im
    display_image()


def butterworth_high():
    clear_canvas()
    global img
    img = original_img.copy()
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cutoff=10
    order=2
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    d = np.sqrt((x - ccol)**2 + (y - crow)**2)
    H = 1 / (1 + (d / cutoff)**(2 * order))
    fshift = fshift * H
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img=img_back
    display_image()


def butterworth_low():
    clear_canvas()
    global img
    img = original_img.copy()
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cutoff = 20
    order = 2
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
    d = np.sqrt(u * u + v * v)
    mask = 1 / (1 + (d / cutoff) ** (2 * order))
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img=img_back

    display_image()


def gaussian_high():
    clear_canvas()
    global img
    img = original_img.copy()
    cutoff=10
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)  
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    d = np.sqrt((x - ccol)**2 + (y - crow)**2)
    H = 1 - np.exp(-(d**2) / (2 * (cutoff**2)))
    fshift = fshift * H
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img=img_back
    display_image()


def gaussian_low():
    clear_canvas()
    global img
    img = original_img.copy()
    cutoff = 50
    sigma = 0.1
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
    d = np.sqrt(u * u + v * v)
    mask = np.exp(-(d ** 2) / (2 * (sigma * cutoff) ** 2))
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img=img_back
    
    display_image()

def ideal_high():
    clear_canvas()
    global img
    img = original_img.copy()
    cutoff=10
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img=img_back
    display_image()

def ideal_low():
    clear_canvas()
    global img
    img = original_img.copy()
    cutoff= 0.1
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.zeros((rows,cols),np.uint8)
    radius = int(cutoff * min(rows,cols)/2)
    mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 1
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img=img_back
    display_image()

def direct_mapping():
    clear_canvas()
    global img
    img = original_img.copy()
    r, c, ch = img.shape
    fact = 2
    new_r = r * fact
    new_c = c * fact
    new_im = np.zeros((new_r, new_c, ch), dtype=np.uint8)

    for k in range(ch):
      for i in range(r):
        for j in range(c):
            new_r_val = i * fact 
            new_c_val = j * fact 
            new_im[new_r_val, new_c_val, k] = img[i, j, k]
            if j != c - 1 and j + 1 < c:
                if img[i, j, k] > img[i, j + 1, k]:
                    max_val = img[i, j, k]
                    min_val = img[i, j + 1, k]
                    for it in range(1, fact):
                        new_im[new_r_val, new_c_val + it, k] = round(
                            ((max_val - min_val) / fact) * (fact - it) + min_val)
                else:
                    max_val = img[i, j + 1, k]
                    min_val = img[i, j, k]
                    for it in range(1, fact):
                        new_im[new_r_val, new_c_val + it, k] = round(((max_val - min_val) / fact) * it + min_val)
            if j == c - 1:
                new_im[new_r_val, new_c_val:new_c_val + fact, k] = img[i, j, k]
            if i == r - 1:
                for m in range(new_c):
                    new_im[new_r_val:new_r, m, k] = new_im[new_r_val, m, k]

    for k in range(ch):
      for i in range(r):
        for j in range(c):
            new_r_val = i * fact 
            if i != r - 1 and i + 1 < r:
                for m in range(new_c):
                    if new_im[new_r_val, m, k] > new_im[new_r_val + fact, m, k]:
                        max_val = new_im[new_r_val, m, k]
                        min_val = new_im[new_r_val + fact, m, k]
                        for it in range(1, fact):
                            new_im[new_r_val + it, m, k] = round(((max_val - min_val) / fact) * (fact - it) + min_val)
                    else:
                        max_val = new_im[new_r_val + fact, m, k]
                        min_val = new_im[new_r_val, m, k]
                        for it in range(1, fact):
                            new_im[new_r_val + it, m,
                                   k] = round(((max_val - min_val) / fact) * it + min_val)

    new_im = new_im.astype(np.uint8)
    img=new_im
    display_image()

def gaussian_low1():
    global img
    img = cv2.Canny(img, 100, 200)
    display_image()


 # Set the green channel to 255
# Define the display function



# Define the file dialog function

    # Display the image on the label widget
def open_image():
    global original_img ,img
    
    # Open a file dialog to get the image file path
    file_path = filedialog.askopenfilename()
    # Load the image file using OpenCV
    img = cv2.imread(file_path)
    
    h, w, _ = img.shape
    if h > w:
        scale_factor = canvas.winfo_height() / h
    else:
        scale_factor = canvas.winfo_width() / w
    img = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)))
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img =img.copy()
    display_image()
   

def display_image():
    global img,photo
    # Convert the image to a Tkinter-compatible format
    
    photo = ImageTk.PhotoImage(image=Image.fromarray(img))
    
    # Update the label widget with the new image
    # Load the image
    img_width = photo.width()
    img_height = photo.height() 
    # Calculate the center coordinates of the canvas
    center_x = canvas.winfo_width() / 2
    center_y = canvas.winfo_height() / 2
    # Calculate the top-left coordinates of the image
    top_left_x = center_x - (img_width / 2)
    top_left_y = center_y - (img_height / 2)
    canvas.create_image(top_left_x,top_left_y, image=photo, anchor="nw")
    canvas.photo = photo


def clear_canvas():
    canvas.delete("all") 
    photo = ImageTk.PhotoImage(image=Image.fromarray(original_img))
    # Update the label widget with the new image
    # Load the image
    img_width = photo.width()
    img_height = photo.height() 
    # Calculate the center coordinates of the canvas
    center_x = canvas.winfo_width() / 2
    center_y = canvas.winfo_height() / 2
    # Calculate the top-left coordinates of the image
    top_left_x = center_x - (img_width / 2)
    top_left_y = center_y - (img_height / 2)
    canvas.create_image(top_left_x,top_left_y, image=photo, anchor="nw")
    canvas.photo = photo


# Create the push buttons for the image processing functions
image_button = tk.Button(left_frame,fg="#FFFFFF", text="Add Image",command=open_image, bg="#1e4452",width=10,height=1,border=2)
image_button.pack(pady=15)
image_button.place_configure(x=2,y=5)

clear_button = tk.Button(left_frame,fg="#FFFFFF", text="clear", command=clear_canvas,  bg="#1e4452",width=10,height=1,border=2)
clear_button.pack(pady=15)
clear_button.place(x=82,y=5)

save_button = tk.Button(left_frame,fg="#FFFFFF", text="Save",command=open_image,  bg="#1e4452",width=10,height=1,border=2)
save_button.pack(pady=15)
save_button.place(x=160,y=5)

congray_button = tk.Button(left_frame, text="convert_Gray",command=grayscale,  bg="#c9e3ed",width=15,height=3)
congray_button.place(x=10,y=55)

histo_button = tk.Button(left_frame, text="histogram", command=hist,  bg="#c9e3ed",width=15,height=3)
histo_button.place(x=130,y=55)

brightness_button = tk.Button(left_frame, text="Brightness ",command=brightness,  bg="#c9e3ed",width=15,height=3)
brightness_button.place(x=10,y=120)

power_button = tk.Button(left_frame, text="power low  ",command=power_low,  bg="#c9e3ed",width=15,height=3)
power_button.place(x=130,y=120)

his_equal_button = tk.Button(left_frame, text="Histo equalization ", command=hist_equalization,  bg="#c9e3ed",width=15,height=3)
his_equal_button.place(x=10,y=190)

his_match_button = tk.Button(left_frame, text="Histo matching ", command=hist_matching,  bg="#c9e3ed",width=15,height=3)
his_match_button.place(x=130,y=190)

merge_image_button = tk.Button(left_frame, text="merge image ", command=merge_image,  bg="#c9e3ed",width=15,height=3)
merge_image_button.place(x=10,y=250)


sub_button = tk.Button(left_frame, text="subtract image ", command=subtract_image,  bg="#c9e3ed",width=15,height=3)
sub_button.place(x=130,y=250)

negative_button = tk.Button(left_frame, text="negative_image ", command=negative_image ,  bg="#c9e3ed",width=15,height=3)
negative_button.place(x=10,y=310)

quant_button = tk.Button(left_frame, text=" Quantization", command=Quantization,  bg="#c9e3ed",width=15,height=3)
quant_button.place(x=130,y=310)

avg_button = tk.Button(left_frame, text="Average Filter",command=Average_Filter,  bg="#c9e3ed",width=15,height=3)
avg_button.place(x=10,y=370)

Sharpening_button = tk.Button(left_frame, text="Sharpening", command=Sharpening,  bg="#c9e3ed",width=15,height=3)
Sharpening_button.place(x=130,y=370)

edge_button = tk.Button(left_frame, text="Edge detection", command=edge_detection,  bg="#c9e3ed",width=15,height=3)
edge_button.place(x=10,y=430)

Unsharpen_button = tk.Button(left_frame, text=" Unsharpen", command=unSharpening,  bg="#c9e3ed",width=15,height=3)
Unsharpen_button.place(x=130,y=430)

Gaussian_button = tk.Button(left_frame, text="Gaussian ", command=Gaussian,  bg="#c9e3ed",width=15,height=3)
Gaussian_button.place(x=10,y=490)

butter_H_button = tk.Button(left_frame, text="Butterworth high", command=butterworth_high,  bg="#c9e3ed",width=15,height=3)
butter_H_button.place(x=130,y=490)

butter_l_button = tk.Button(left_frame, text="ButterWorth low ", command=butterworth_low,  bg="#c9e3ed",width=15,height=3)
butter_l_button.place(x=10,y=550)

guasssion_h_button = tk.Button(left_frame, text="Gaussian high", command=gaussian_high,  bg="#c9e3ed",width=15,height=3)
guasssion_h_button.place(x=130,y=550)

guasssion_l_button = tk.Button(left_frame, text="Gaussian low ",command=gaussian_low,  bg="#c9e3ed",width=15,height=3)
guasssion_l_button.place(x=10,y=610)


ideal_h_button = tk.Button(left_frame, text="Ideal high ",command=ideal_high,  bg="#c9e3ed",width=15,height=3)
ideal_h_button.place(x=130,y=610)

ideal_h_button = tk.Button(left_frame, text="Ideal high ",command=ideal_low,  bg="#c9e3ed",width=15,height=3)
ideal_h_button.place(x=10,y=670)

direct_mapping_button = tk.Button(left_frame, text="Direct mapping",command=direct_mapping,  bg="#c9e3ed",width=15,height=3)
direct_mapping_button.place(x=130,y=670)



root.mainloop()