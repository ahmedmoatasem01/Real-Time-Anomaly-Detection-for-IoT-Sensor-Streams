import os
import subprocess

REPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'final_report'))
SECTIONS_DIR = os.path.join(REPORT_DIR, 'sections')
FINAL_MD = os.path.join(REPORT_DIR, 'FINAL_PROJECT_REPORT.md')

PARTS = [
    'part1_intro_scope.md',
    'part2_data_features.md',
    'part3_ml_eval.md',
    'part4_system_architecture.md',
    'part5_frontend_apis.md',
    'part6_conclusion_appendices.md'
]

def compile_markdown():
    print(f"Compiling markdown parts into {FINAL_MD}...")
    with open(FINAL_MD, 'w', encoding='utf-8') as outfile:
        for part_name in PARTS:
            part_path = os.path.join(SECTIONS_DIR, part_name)
            if os.path.exists(part_path):
                print(f"  -> Adding {part_name}")
                with open(part_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    outfile.write("\n\n<!-- PAGE BREAK -->\n<div style=\"page-break-after: always;\"></div>\n\n")
            else:
                print(f"  -> WARNING: {part_name} not found!")

def generate_pdf():
    print("Generating PDF via npx mdpdf...")
    try:
        result = subprocess.run(
            ['npx', '--yes', 'mdpdf', 'FINAL_PROJECT_REPORT.md'],
            cwd=REPORT_DIR,
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("PDF generation successful!")
        else:
            print("PDF generation failed.")
            print(result.stderr)
            print(result.stdout)
    except Exception as e:
        print(f"Failed to run PDF generation: {e}")

if __name__ == "__main__":
    compile_markdown()
    generate_pdf()
