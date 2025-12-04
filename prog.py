import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

def select_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not path:
        return

    results = model(path, save=True, project=".", name="Results", exist_ok=True)
    result_path = results[0].path 

    if len(results[0].boxes) > 0:
        cls_index = int(results[0].boxes[0].cls[0])
        class_name = results[0].names[cls_index]
        class_label.config(text=f"Класс: {class_name}")
    else:
        class_label.config(text="Класс: не найдено")

    img = Image.open(result_path)
    img = img.resize((600, 600))
    img_tk = ImageTk.PhotoImage(img)

    panel.config(image=img_tk)
    panel.image = img_tk


root = tk.Tk()
root.title("ПО для распознавания одежды на изображениях")

win_width = 1000
win_height = 700

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x = (screen_width // 2) - (win_width // 2)
y = (screen_height // 2) - (win_height // 2)

root.geometry(f"{win_width}x{win_height}+{x}+{y}")

btn = tk.Button(root, text="Выбрать изображение", command=select_image)
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack()

class_label = tk.Label(root, text="Класс: неизвестно", fg="black", font=("Times New Roman", 16))
class_label.place(relx=0.95, rely=0.05, anchor="ne")

root.resizable(False, False)
root.mainloop()