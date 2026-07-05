import os
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_bearing_dataset(output_dir: str, num_files: int = 500, points_per_file: int = 20480):
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = datetime(2004, 2, 12, 10, 32, 39)
    
    print(f"Generating {num_files} synthetic bearing files...")
    
    for i in range(num_files):
        current_time = start_time + timedelta(minutes=10 * i)
        filename = current_time.strftime("%Y.%m.%d.%H.%M.%S")
        filepath = os.path.join(output_dir, filename)
        
        # Base healthy noise (N(0, 0.1))
        noise = np.random.normal(0, 0.1, points_per_file)
        
        if i > 400:
            # Degradation phase
            progress = (i - 400) / 100.0  # 0 to 1
            # Increase variance
            noise += np.random.normal(0, 0.05 + 0.5 * progress, points_per_file)
            # Add impacts (peaks)
            if progress > 0.5:
                num_peaks = int(10 * progress)
                peak_indices = np.random.randint(0, points_per_file, num_peaks)
                noise[peak_indices] += np.random.choice([-1, 1], num_peaks) * (2.0 + 3.0 * progress)
                
        # We need 4 columns to match NASA Bearing format. Column 0 is the one that degrades.
        col1 = noise
        col2 = np.random.normal(0, 0.1, points_per_file)
        col3 = np.random.normal(0, 0.1, points_per_file)
        col4 = np.random.normal(0, 0.1, points_per_file)
        
        data = np.column_stack((col1, col2, col3, col4))
        
        # Save as tab-separated values
        np.savetxt(filepath, data, delimiter='\t', fmt='%.5f')
        
    print(f"Dataset generated at {output_dir}")

if __name__ == "__main__":
    generate_synthetic_bearing_dataset("data/raw/bearing/2nd_test")
