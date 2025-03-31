from bin.Network import Network
from bin import DataLoader
import pickle
import numpy as np
from PIL import Image
import os
import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageDraw

def main():

    training_data, validation_data, test_data = DataLoader.load_data_wrapper("data/mnist.pkl.gz")

    
    #net.StochasticGradientDescent(training_data, epochs=10, mini_batch_size=10,learning_rate=3.0, test_data=test_data)

    interactive()

    #load_from_mnist()

   

def test_network2(net: Network):
    # load the folder ./data/some and iterate through all the images in the folder

    folder_path = "./data/some/"

    #open the folder and iterate through all the images

    for image_path in os.listdir(folder_path):
        # Open the image and convert to grayscale
        img = Image.open(folder_path + image_path).convert('L')

        # Resize to 28x28 if not already that size
        img = img.resize((28, 28))

        # Convert to numpy array and normalize values to [0,1]
        img_array = np.array(img).reshape((784,1)).astype('float32') / 255.0

        result = net.feedforward(img_array)
        print("Predicted digit:", image_path, np.argmax(result))



def test_network(net: Network):
    # Load and process a 28x28 PNG image into a 784 length numpy array
    image_path = "./data/8.png"

    # Open the image and convert to grayscale
    img = Image.open(image_path).convert('L')

    # Resize to 28x28 if not already that size
    img = img.resize((28, 28))

    # Convert to numpy array and normalize values to [0,1]
    img_array = np.array(img).reshape((784,1)).astype('float32') / 255.0

    result = net.feedforward(img_array)
    print("Network prediction:", result.shape)
    print("Predicted digit:", np.argmax(result))

def save_network(net):
    #Save the net object
    with open("data/net.pkl", 'wb') as f:
        pickle.dump(net, f)

def load_network():
    #Load the net object
    with open("data/net.pkl", 'rb') as f:
        net = pickle.load(f)
    return net


def interactive():
        # Create the main window
        root = tk.Tk()
        root.title("Digit Recognition")
        
        # Variables
        drawing = False
        last_x, last_y = 0, 0
        
        # Create a canvas widget
        canvas = Canvas(root, width=280, height=280, bg="black")
        canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        
        # Functions for drawing
        def start_draw(event):
            nonlocal drawing, last_x, last_y
            drawing = True
            last_x, last_y = event.x, event.y
        
        def draw(event):
            nonlocal drawing, last_x, last_y
            if drawing:
                canvas.create_line(last_x, last_y, event.x, event.y, fill="white", width=15, capstyle=tk.ROUND, smooth=True)
                last_x, last_y = event.x, event.y
        
        def stop_draw(event):
            nonlocal drawing
            drawing = False
        
        # Bind mouse events
        canvas.bind("<Button-1>", start_draw)
        canvas.bind("<B1-Motion>", draw)
        canvas.bind("<ButtonRelease-1>", stop_draw)
        
        # Function to recognize the drawn digit
        def recognize():
            # Convert canvas to image
            img = get_image_from_canvas(canvas)

            # save the img to ./data/some as last.png

            img.resize((28,28)).save("./data/some/last.png")
            
            # Load the network
            net = load_network()
            
            # Process image for the network
            img_array = preprocess_image(img)

            
            # Get prediction
            result = net.feedforward(img_array)
            predicted = np.argmax(result)
            
            # Show result
            result_label.config(text=f"Predicted: {predicted}")
        
        # Function to clear the canvas
        def clear_canvas():
            canvas.delete("all")
        
        # Create buttons
        recognize_btn = Button(root, text="Recognize", command=recognize)
        recognize_btn.grid(row=1, column=0, padx=5, pady=5)
        
        clear_btn = Button(root, text="Clear", command=clear_canvas)
        clear_btn.grid(row=1, column=1, padx=5, pady=5)
        
        # Label for showing results
        result_label = tk.Label(root, text="Draw a digit")
        result_label.grid(row=1, column=2, padx=5, pady=5)
        
        root.mainloop()

def get_image_from_canvas(canvas):
        # Create an image from the canvas
        x = canvas.winfo_rootx() + canvas.winfo_x()
        y = canvas.winfo_rooty() + canvas.winfo_y()
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        # Create a blank image with a black background
        img = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(img)
        
        # Get all lines from the canvas and draw them on the image
        items = canvas.find_all()
        for item in items:
            if canvas.type(item) == "line":
                coords = canvas.coords(item)
                draw.line(coords, fill="white", width=15)
        
        return img

def preprocess_image(img):
        # Resize to 28x28
        img = img.resize((28, 28))
        
        # Convert to numpy array and normalize values to [0,1]
        img_array = np.array(img).reshape((784, 1)).astype('float32') / 255.0
        
        return img_array


def load_from_mnist():
    training_data, validation_data, test_data = DataLoader.load_data_wrapper("data/mnist.pkl.gz")
    
    #extract 15 image and the label from the training data

    #shuffle the training data

    np.random.shuffle(test_data)

    test_data = test_data[:15]

    #seperate the images and turn them into .png files and name them with the label
    # if such files do not exist create them

    for i, (image, label) in enumerate(test_data):
        #reshape the image
        image = image.reshape((28, 28))
        #create a new image object
        img = Image.fromarray((image * 255).astype('uint8'))
        #save the image
        img.save(f"data/some/{i}.png")
        




    
if __name__ == "__main__":
    main()