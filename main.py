import pandas as pd
import pickle
import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Set appearance mode and default color theme
ctk.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
ctk.set_default_color_theme("green")  # Themes: "blue", "green", "dark-blue"

class DiseasePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("💊 Advanced Disease Prediction System")
        self.root.geometry("1000x700")
        
        # Load model and features
        self.model = self.load_model("model.pkl")
        self.symptom_features = self.load_features("final_data_with_prognosis.csv")
        self.symptom_vars = {}
        self.prediction_made = False
        self.current_prediction = ""
        self.confidence_score = 0
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create left and right frames
        self.left_frame = ctk.CTkFrame(self.main_frame)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        self.right_frame = ctk.CTkFrame(self.main_frame)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        self.create_header()
        self.create_symptom_selector()
        self.create_predict_button()
        self.create_results_area()
    
    def load_model(self, path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            # Create a dummy model for demonstration purposes
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier()
    
    def load_features(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            X = df.drop("prognosis", axis=1)
            return list(X.columns)
        except FileNotFoundError:
            # Return sample symptoms for demonstration
            return ["itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", 
                   "shivering", "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue",
                   "muscle_wasting", "vomiting", "burning_micturition", "spotting_urination", "fatigue",
                   "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss"]
    
    def create_header(self):
        # Header in left frame
        self.header_frame = ctk.CTkFrame(self.left_frame, corner_radius=10)
        self.header_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            self.header_frame,
            text="Disease Predictor",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(pady=10)
        
        ctk.CTkLabel(
            self.header_frame,
            text="Select your symptoms below",
            font=ctk.CTkFont(size=16)
        ).pack(pady=5)
    
    def create_symptom_selector(self):
        # Create a scrollable frame for symptoms
        self.symptom_frame = ctk.CTkScrollableFrame(self.left_frame)
        self.symptom_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add a search box
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self.filter_symptoms)
        
        search_frame = ctk.CTkFrame(self.symptom_frame)
        search_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(search_frame, text="Search:").pack(side="left", padx=5)
        
        search_entry = ctk.CTkEntry(search_frame, textvariable=self.search_var, width=200)
        search_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        # Create Symptom Checkboxes
        self.checkbox_frames = {}
        
        for symptom in self.symptom_features:
            formatted_symptom = symptom.replace('_', ' ').title()
            frame = ctk.CTkFrame(self.symptom_frame)
            frame.pack(fill="x", pady=2)
            self.checkbox_frames[symptom] = frame
            
            var = tk.IntVar()
            checkbox = ctk.CTkCheckBox(
                frame, 
                text=formatted_symptom,
                variable=var,
                onvalue=1,
                offvalue=0
            )
            checkbox.pack(side="left", padx=10, pady=2, anchor="w")
            self.symptom_vars[symptom] = var
    
    def filter_symptoms(self, *args):
        search_text = self.search_var.get().lower()
        
        for symptom, frame in self.checkbox_frames.items():
            if search_text in symptom.lower():
                frame.pack(fill="x", pady=2)
            else:
                frame.pack_forget()
    
    def create_predict_button(self):
        # Button Frame
        button_frame = ctk.CTkFrame(self.left_frame)
        button_frame.pack(fill="x", pady=10)
        
        # Clear button
        ctk.CTkButton(
            button_frame,
            text="Clear All",
            command=self.clear_selection,
            fg_color="#FF5722",
            hover_color="#E64A19"
        ).pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        # Predict button
        ctk.CTkButton(
            button_frame,
            text="Predict Disease",
            command=self.predict_disease,
            fg_color="#4CAF50",
            hover_color="#45a049"
        ).pack(side="right", padx=10, pady=10, fill="x", expand=True)
    
    def create_results_area(self):
        # Header for results
        ctk.CTkLabel(
            self.right_frame,
            text="Prediction Results",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=10)
        
        # Initial message
        self.result_frame = ctk.CTkFrame(self.right_frame, corner_radius=10)
        self.result_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.result_label = ctk.CTkLabel(
            self.result_frame,
            text="No prediction yet.\nPlease select symptoms and click 'Predict Disease'.",
            font=ctk.CTkFont(size=16),
            wraplength=350
        )
        self.result_label.pack(pady=20)
        
        # Frame for confidence score visualization
        self.confidence_frame = ctk.CTkFrame(self.right_frame, corner_radius=10)
        self.confidence_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.confidence_header = ctk.CTkLabel(
            self.confidence_frame,
            text="Confidence Score",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.confidence_header.pack(pady=10)
        
        # Frame for the gauge
        self.gauge_frame = ctk.CTkFrame(self.confidence_frame)
        self.gauge_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Additional Information Section
        self.info_frame = ctk.CTkFrame(self.right_frame, corner_radius=10)
        self.info_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            self.info_frame,
            text="Additional Information",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        self.info_text = ctk.CTkTextbox(self.info_frame, height=150)
        self.info_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.info_text.insert("1.0", "Select symptoms to get a prediction and additional information about the disease.")
        self.info_text.configure(state="disabled")
        
        # Create initial empty gauge
        self.create_confidence_gauge(0)
    
    def clear_selection(self):
        # Clear all checkboxes
        for var in self.symptom_vars.values():
            var.set(0)
        
        # Reset the prediction area
        self.result_label.configure(text="No prediction yet.\nPlease select symptoms and click 'Predict Disease'.")
        #self.create_confidence_gauge(0)
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", "Select symptoms to get a prediction and additional information about the disease.")
        self.info_text.configure(state="disabled")
        self.prediction_made = False
    
    def create_confidence_gauge(self, confidence):
        # Clear previous gauge
        for widget in self.gauge_frame.winfo_children():
            widget.destroy()
        
        # Create matplotlib figure for the gauge
        fig = plt.Figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        # Draw the gauge
        self.draw_gauge(ax, confidence)
        
        # Embed the figure in the frame
        canvas = FigureCanvasTkAgg(fig, master=self.gauge_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    def draw_gauge(self, ax, confidence):
        # Clear axes
        ax.clear()
        
        # Gauge settings
        gauge_min = 0
        gauge_max = 100
        gauge_range = gauge_max - gauge_min
        
        # Set gauge properties
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw gauge background - CORRECTED ORIENTATION
        wedge_bg = plt.matplotlib.patches.Wedge(
            center=(0.5, 0.5),
            r=0.4,
            theta1=0,
            theta2=180,  # Correct orientation for top half
            width=0.1,
            facecolor='#E0E0E0',
            edgecolor='gray',
            linewidth=1
        )
        ax.add_patch(wedge_bg)
        
        # Calculate the angle for the confidence level - CORRECTED ORIENTATION
        angle = confidence / 100 * 180
        
        # Choose color based on confidence
        if confidence < 40:
            color = '#FF5252'  # Red
        elif confidence < 70:
            color = '#FFC107'  # Yellow/Amber
        else:
            color = '#4CAF50'  # Green
        
        # Draw the confidence level arc - CORRECTED ORIENTATION
        wedge = plt.matplotlib.patches.Wedge(
            center=(0.5, 0.5),
            r=0.4,
            theta1=0,
            theta2=angle,
            width=0.1,
            facecolor=color,
            edgecolor='gray',
            linewidth=1
        )
        ax.add_patch(wedge)
        
        # Add the pointer/needle
        pointer_length = 0.35
        pointer_angle = np.radians(angle)
        pointer_x = 0.5 + pointer_length * np.cos(pointer_angle)
        pointer_y = 0.5 + pointer_length * np.sin(pointer_angle)
        
        ax.plot([0.5, pointer_x], [0.5, pointer_y], color='black', linewidth=2)
        ax.add_patch(plt.matplotlib.patches.Circle((0.5, 0.5), radius=0.03, facecolor='black'))
        
        # Add ticks and labels - CORRECTED POSITIONS
        for i, tick in enumerate([0, 25, 50, 75, 100]):
            angle_rad = np.radians(tick / 100 * 180)
            tick_x = 0.5 + 0.45 * np.cos(angle_rad)
            tick_y = 0.5 + 0.45 * np.sin(angle_rad)
            label_x = 0.5 + 0.55 * np.cos(angle_rad)
            label_y = 0.5 + 0.55 * np.sin(angle_rad)
            
            ax.text(label_x, label_y, f"{tick}%", 
                    ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Add confidence value text - CORRECTED POSITION
        ax.text(0.5, 0.75, f"{confidence:.1f}%", 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax.text(0.5, 0.65, "Confidence", ha='center', va='center', fontsize=12)

    def predict_disease(self):
        # Get selected symptoms
        input_vector = [self.symptom_vars[symptom].get() for symptom in self.symptom_features]
    
        # Check if any symptoms are selected
        if sum(input_vector) == 0:
            self.result_label.configure(text="⚠️ Please select at least one symptom to make a prediction.")
            return
        
        try:
            # Make prediction
            prediction = self.model.predict([input_vector])[0]
            
            # Get prediction probabilities for REAL confidence score
            # Get the actual probability scores from the model
            proba = self.model.predict_proba([input_vector])[0]
            
            # Get the class indices
            class_indices = self.model.classes_
            
            # Find the index of our prediction
            pred_index = np.where(class_indices == prediction)[0][0]
            
            # Get the confidence score for the predicted class (as a percentage)
            confidence = proba[pred_index] * 100
            
            # Update the UI with prediction results
            self.show_prediction_results(prediction, confidence)
            
        except Exception as e:
            self.result_label.configure(text=f"⚠️ Error making prediction: {str(e)}\n\nPlease make sure your model is correctly trained.")
    def show_prediction_results(self, prediction, confidence):
            # Format the disease name for display
            formatted_disease = prediction.replace('_', ' ').title()
            
            # Update result label
            self.result_label.configure(
                text=f"📊 Prediction Results\n\n🔍 Predicted Disease:\n{formatted_disease}",
                font=ctk.CTkFont(size=18, weight="bold" )
            )
            
            # Update confidence gauge
            self.create_confidence_gauge(confidence)
            
            # Update additional information
            self.info_text.configure(state="normal")
            self.info_text.delete("1.0", "end")
            
            # Disease information (in a real app, this would come from a database)
            disease_info = self.get_disease_info(prediction)
            
            self.info_text.insert("1.0", disease_info)
            self.info_text.configure(state="disabled")
            
            # Save current prediction info
            self.prediction_made = True
            self.current_prediction = prediction
            self.confidence_score = confidence
    
    def get_disease_info(self, disease):
        """Get information about the disease (sample data for demonstration)"""
        diseases_info = {
            "Fungal infection": "A fungal infection, also called mycosis, is a skin disease caused by a fungus. Common types include athlete's foot, ringworm, and yeast infections. Treatment usually involves antifungal medications.",
            "Allergy": "An allergy is an immune system response to a foreign substance that's not typically harmful to your body. These foreign substances are called allergens. They can include certain foods, pollen, or pet dander.",
            "GERD": "Gastroesophageal reflux disease (GERD) is a chronic digestive disease that occurs when stomach acid or bile flows back into the food pipe and irritates the lining. Treatment includes lifestyle changes and medications.",
            "Chronic cholestasis": "Chronic cholestasis is a condition where bile flow from the liver is reduced or blocked. This can lead to a buildup of bilirubin in the blood, causing jaundice and other symptoms.",
            "Drug Reaction": "A drug reaction is an adverse effect from medication. Symptoms can range from mild to severe and include rash, hives, itching, swelling, and in severe cases, anaphylaxis.",
            "Peptic ulcer diseae": "Peptic ulcer disease is a condition where open sores develop on the inside lining of the stomach and the upper portion of the small intestine. The most common symptom is abdominal pain.",
            "AIDS": "Acquired immunodeficiency syndrome (AIDS) is a chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV). It damages your immune system and interferes with your body's ability to fight infection."
        }
        
        # Default information if disease is not in our sample database
        return diseases_info.get(disease, f"Information about {disease.replace('_', ' ').title()} is not available in our database. Please consult with a healthcare professional for accurate diagnosis and treatment options.")

if __name__ == "__main__":
    # Initialize CustomTkinter
    app = ctk.CTk()
    predictor = DiseasePredictorApp(app)
    app.mainloop()