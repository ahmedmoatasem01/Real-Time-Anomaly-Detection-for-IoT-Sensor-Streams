import os
import subprocess

REPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'final_report'))
FIGURES_DIR = os.path.join(REPORT_DIR, 'figures')

def generate_diagrams():
    print("Converting Mermaid files to PNG...")
    for filename in os.listdir(FIGURES_DIR):
        if filename.endswith(".mmd"):
            filepath = os.path.join(FIGURES_DIR, filename)
            outpath = os.path.join(FIGURES_DIR, filename.replace(".mmd", ".png"))
            
            print(f"Generating {outpath} from {filepath}")
            # Note: mmdc executable should be in node_modules/.bin/mmdc if installed via npm install
            # For Windows npx is used:
            try:
                result = subprocess.run(
                    ['npx', '--yes', 'mmdc', '-i', filepath, '-o', outpath, '-b', 'transparent'],
                    cwd=os.path.dirname(__file__) + "/..",
                    shell=True,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"Success: {filename}")
                else:
                    print(f"Failed: {filename}")
                    print(result.stderr)
            except Exception as e:
                print(f"Error running mmdc for {filename}: {e}")

if __name__ == "__main__":
    generate_diagrams()
