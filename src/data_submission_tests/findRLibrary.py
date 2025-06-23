import subprocess
import glob, os


# find Rscript.exe
def find_rscript():
    possible_directories = [
        r"C:\Users\mfbx2rdb\AppData\Local\Programs\R"
        #r"C:\Program Files\R",                   
        #r"C:\Program Files (x86)\R",
        #os.environ.get("ProgramFiles"),  
        #os.environ.get("ProgramFiles(x86)")
    ]
    for directory in possible_directories:
        if directory:
            rscript_paths = glob.glob(os.path.join(directory, "**", "Rscript.exe"), recursive=True)
            if rscript_paths:
                return rscript_paths[0]
    return None
def find_r_libraries(rscript_path):
    try:
        result = subprocess.run([rscript_path, "-e", "installed.packages()[,1]"], capture_output=True, text=True, check=True)
        libraries = result.stdout.split('\n')
        return libraries
    except subprocess.CalledProcessError as e:
        print(f"Error finding R libraries: {e}")
        return None

# Example usage
rscript_path = find_rscript()
if rscript_path:
    libraries = find_r_libraries(rscript_path)
    if libraries:
        print("Installed R libraries:")
        for lib in libraries:
            print(lib)
    else:
        print("No libraries found or an error occurred.")
else:
    print("Rscript.exe not found.")