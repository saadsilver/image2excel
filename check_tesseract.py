import os
import shutil
import subprocess
import sys

def check_tesseract():
    log_file = "tesseract_diagnosis.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        def log(msg):
            print(msg)
            f.write(msg + "\n")

        log("🔍 Tesseract Diagnostic Tool")
        log("-" * 30)

        # 1. Check environment variables
        log(f"OS: {os.name}")
        log(f"PATH: {os.environ.get('PATH', '')}")
        
        # 2. Search for tesseract in standard paths
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.join(os.getenv('LOCALAPPDATA', ''), r"Tesseract-OCR\tesseract.exe")
        ]
        
        found_path = None
        for path in possible_paths:
            if os.path.exists(path):
                log(f"✅ Found Tesseract at: {path}")
                found_path = path
                break
            else:
                log(f"❌ Not found at: {path}")

        # 3. Check if accessible via PATH
        which_tesseract = shutil.which("tesseract")
        if which_tesseract:
            log(f"✅ 'tesseract' command found in PATH at: {which_tesseract}")
            if not found_path:
                found_path = which_tesseract
        else:
            log("❌ 'tesseract' command NOT found in PATH")

        # 4. Try execution
        if found_path:
            try:
                result = subprocess.run([found_path, "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    log("\n✅ Tesseract execution successful!")
                    log(result.stdout.strip())
                else:
                    log("\n⚠️ Tesseract execution failed with error:")
                    log(result.stderr)
            except Exception as e:
                log(f"\n❌ Error trying to run Tesseract: {e}")
        else:
            log("\n🚫 FATAL: Tesseract executable not found anywhere.")
            log("Please install it from: https://github.com/UB-Mannheim/tesseract/wiki")

if __name__ == "__main__":
    check_tesseract()
