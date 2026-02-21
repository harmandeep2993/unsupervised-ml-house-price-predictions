import os, logging
import pandas as pd

class DataLoader:

    def __init__(self, file_path):
        self.filepath = file_path
        self.data = None
    
    def load_dataset(self):
        
        if not os.path.exits(self.filepath):
            raise FileNotFoundError(f"File not found at: {self.filepath}")

        if self.filepath.endswith(".csv"):
            self.data = pd.read_csv(self.filepath)

        elif self.filepath.endswith(".xlsx"):
                self.data = pd.read_excel(self.filepath)
        
        elif self.filepath.endswidth(",json"):
                self.data = pd.read_json(self.filepath)
        
        else:
             raise ValueError("Unsupported file format. Use CSV, XLSX, or JSON")

        return self.data