import face_recognition
import cv2
import os
import shutil
import numpy as np
import threading
from tkinter import Tk, filedialog, Button, Label, Canvas, Scrollbar, Frame, Toplevel, Scale, BooleanVar, messagebox
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk

class FaceFinderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Secure Face Finder")
        self.root.geometry("500x300")
        self.root.resizable(False, False)
        
        # Configuration
        self.source_folder = ""
        self.output_folder = ""
        self.selected_face_encoding = None
        self.tolerance = 0.45  # More accurate default
        self.face_encodings_list = []
        self.face_images = []
        self.processing = False
        self.include_subfolders = BooleanVar(value=False)
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)
        
        # Header
        Label(main_frame, text="Secure Face Finder", font=("Arial", 16)).pack(pady=10)
        
        # Folder selection
        folder_frame = Frame(main_frame)
        folder_frame.pack(fill="x", pady=5)
        Button(folder_frame, text="Select Image Folder", command=self.select_folder).pack(side="left")
        self.folder_label = Label(folder_frame, text="No folder selected")
        self.folder_label.pack(side="left", padx=10)
        
        # Tolerance control
        tolerance_frame = Frame(main_frame)
        tolerance_frame.pack(fill="x", pady=5)
        Label(tolerance_frame, text="Matching Accuracy:").pack(side="left")
        self.tolerance_slider = Scale(
            tolerance_frame, 
            from_=0.3, 
            to=0.6, 
            resolution=0.01,
            orient="horizontal",
            showvalue=False,
            command=self.update_tolerance
        )
        self.tolerance_slider.set(self.tolerance)
        self.tolerance_slider.pack(side="left", fill="x", expand=True, padx=5)
        self.tolerance_label = Label(tolerance_frame, text=f"{self.tolerance:.2f}")
        self.tolerance_label.pack(side="left")
        
        # Options
        options_frame = Frame(main_frame)
        options_frame.pack(fill="x", pady=5)
        
        # Progress area
        self.progress_frame = Frame(main_frame)
        self.progress_label = Label(self.progress_frame, text="")
        self.progress_label.pack()
        self.progress_bar = Progressbar(
            self.progress_frame, 
            orient="horizontal", 
            length=300, 
            mode="determinate"
        )
        self.progress_bar.pack(pady=5)
        
        # Status area
        self.status_label = Label(main_frame, text="", fg="blue")
        self.status_label.pack(pady=10)
        
        # Action buttons
        self.action_button = Button(
            main_frame, 
            text="Detect Faces", 
            command=self.start_detection,
            state="disabled"
        )
        self.action_button.pack(pady=10)
        
    def update_tolerance(self, value):
        self.tolerance = float(value)
        self.tolerance_label.config(text=f"{self.tolerance:.2f}")

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.source_folder = folder
            self.output_folder = os.path.join(self.source_folder, "Face_Matches")
            os.makedirs(self.output_folder, exist_ok=True)
            self.folder_label.config(text=os.path.basename(folder))
            self.action_button.config(state="normal")
            self.status_label.config(text="Ready to detect faces")

    def start_detection(self):
        if self.processing:
            return
            
        self.processing = True
        self.status_label.config(text="Processing...")
        self.progress_frame.pack(pady=10)
        self.action_button.config(state="disabled")
        
        # Clear previous results
        self.face_encodings_list = []
        self.face_images = []
        
        # Start in thread to prevent GUI freeze
        threading.Thread(target=self.detect_faces, daemon=True).start()

    def detect_faces(self):
        try:
            image_files = []
            for root, _, files in os.walk(self.source_folder):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_files.append(os.path.join(root, file))
            
            total = len(image_files)
            if total == 0:
                self.root.after(0, lambda: self.status_label.config(text="No images found!", fg="red"))
                return
                
            # Setup progress
            self.root.after(0, lambda: self.progress_bar.config(maximum=total))
            
            processed = 0
            for path in image_files:
                if not self.processing:  # Allow cancellation
                    break
                    
                try:
                    image = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(image)
                    locations = face_recognition.face_locations(image)
                    
                    for encoding, location in zip(encodings, locations):
                        # Check if face already exists
                        if not any(face_recognition.compare_faces(
                            self.face_encodings_list, 
                            encoding, 
                            tolerance=self.tolerance
                        )):
                            top, right, bottom, left = location
                            face_crop = image[top:bottom, left:right]
                            face_pil = Image.fromarray(face_crop).resize((100, 100))
                            self.face_images.append(face_pil)
                            self.face_encodings_list.append(encoding)
                            
                except Exception as e:
                    print(f"Error processing {os.path.basename(path)}: {str(e)}")
                    
                processed += 1
                progress = int(100 * processed / total)
                self.root.after(0, lambda p=progress: self.progress_bar.config(value=processed))
                self.root.after(0, lambda: self.progress_label.config(
                    text=f"Processing: {processed}/{total} images ({progress}%)"
                ))
                
            if self.processing:
                self.root.after(0, self.show_faces)
                
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Error: {str(e)}", fg="red"))
        finally:
            self.root.after(0, lambda: self.action_button.config(state="normal"))
            self.processing = False

    def show_faces(self):
        if not self.face_images:
            self.status_label.config(text="No faces detected!", fg="red")
            return
            
        top = Toplevel(self.root)
        top.title("Select Target Face")
        top.geometry("800x600")
        top.transient(self.root)
        top.grab_set()
        
        canvas = Canvas(top)
        frame = Frame(canvas)
        scrollbar = Scrollbar(top, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=frame, anchor="nw")
        
        # Display faces in grid
        for idx, img in enumerate(self.face_images):
            img_tk = ImageTk.PhotoImage(img)
            btn = Button(
                frame, 
                image=img_tk, 
                command=lambda i=idx: self.select_face(i)
            )
            btn.image = img_tk
            btn.grid(row=idx // 5, column=idx % 5, padx=5, pady=5)
        
        frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
        
        # Add selection button
        Button(top, text="Select Face", command=top.destroy).pack(pady=10)
        
        self.status_label.config(text=f"Found {len(self.face_images)} unique faces")

    def select_face(self, index):
        self.selected_face_encoding = self.face_encodings_list[index]
        self.status_label.config(text="Matching faces...")
        threading.Thread(target=self.match_faces, daemon=True).start()

    def match_faces(self):
        try:
            image_files = []
            for root, _, files in os.walk(self.source_folder):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_files.append(os.path.join(root, file))
            
            total = len(image_files)
            matches = 0
            
            self.root.after(0, lambda: self.progress_bar.config(value=0, maximum=total))
            self.root.after(0, lambda: self.progress_label.config(text="Searching for matches..."))
            
            for i, path in enumerate(image_files):
                if not self.processing:
                    break
                    
                try:
                    image = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(image)
                    
                    for encoding in encodings:
                        if face_recognition.compare_faces(
                            [self.selected_face_encoding], 
                            encoding, 
                            tolerance=self.tolerance
                        )[0]:
                            filename = os.path.basename(path)
                            shutil.copy(path, os.path.join(self.output_folder, filename))
                            matches += 1
                            break
                            
                except Exception as e:
                    print(f"Error matching {os.path.basename(path)}: {str(e)}")
                
                self.root.after(0, lambda v=i+1: self.progress_bar.config(value=v))
                self.root.after(0, lambda: self.progress_label.config(
                    text=f"Checked {i+1}/{total} images, found {matches} matches"
                ))
                
            self.root.after(0, lambda: self.status_label.config(
                text=f"Found {matches} matches! Saved to 'Face_Matches' folder", 
                fg="green"
            ))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Error: {str(e)}", fg="red"))
        finally:
            self.processing = False
            self.root.after(0, lambda: self.action_button.config(state="normal"))

if __name__ == "__main__":
    root = Tk()
    app = FaceFinderApp(root)
    root.mainloop()