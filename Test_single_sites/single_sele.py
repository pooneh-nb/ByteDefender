from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
import time
import shutil
from pyvirtualdisplay import Display
import requests
import os
import subprocess
import threading
import queue
from pathlib import Path
import sys
sys.path.insert(1, Path.home().cwd().as_posix())
import utilities


COUNT = 0
SLEEP = 20
PORT = 4000
site = "mercadopago.com.ar"

# virtual display
display = Display(visible=0, size=(1000, 1000))
display.start()

            
def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()


def visitWebsite(df, sleep, mouse_move):
    try:
        print("Initialzing instrumented chrome...")
        print("Visiting: ", r"https://" + site)
        # Command to start Chrome
        chrome_path = Path.home() / 'out' / 'Default' / 'chrome'
        extension_path = Path.home().cwd() / 'extension'
        # Start Chrome and capture stdout
        chrome_command = f"{str(chrome_path)} --no-sandbox \
                                              --remote-debugging-port=9222 \
                                              --load-extension={str(extension_path)} \
                                              --disable-dev-shm-usage"
        # Start Chrome and capture stdout
        chrome_process = subprocess.Popen(
                        chrome_command, 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        shell=True,
                        bufsize=1,  # Line buffered
                        universal_newlines=True)
        # Create a queue to capture the output
        output_queue = queue.Queue()
        # Start a thread to capture the output
        output_thread = threading.Thread(target=enqueue_output, args=(chrome_process.stdout, output_queue))
        output_thread.daemon = True
        output_thread.start()

        # Wait for a moment to ensure Chrome starts properly
        time.sleep(4)
    
        opt = Options()

        # initiating a Chrome process with remote debugging enabled
        opt.add_experimental_option("debuggerAddress", "localhost:9222")

        # to make Selenium not wait for the complete page load
        opt.page_load_strategy = 'eager'

        # specifies the chrome driver (matches with the chrome binary)
        driver_path = Path.home() / 'out' / 'Default' / 'chromedriver'
        service = Service(str(driver_path))
        driver = webdriver.Chrome(service=service, options=opt)
            
        requests.post(
            url="http://localhost:"+str(PORT)+"/complete", data={"website": site})
        
        # Delete all cookies
        driver.delete_all_cookies()
        # set timeout
        driver.set_page_load_timeout(20)

        driver.implicitly_wait(sleep)

        # get the url
        driver.get(url = r"https://" + site)
        # sleep
        time.sleep(sleep)
        
        print("Waking up")

        # Save the bytecodes to a file
        logs_output = "server/output/" + site + "/browser_logs.bin"
        with open(logs_output, 'a') as log_file:
            # Process output from Chrome periodically
            while not output_queue.empty():
                line = output_queue.get_nowait()
                log_file.write(line) 
        
        driver.quit()
        if len(utilities.get_files_in_a_directory("server/output/" + site)) < 2:
            shutil.rmtree("server/output/" + site)
            print("Removed ... A single file")
        else:
            print(r"Completed: " + " website: " + site)
        chrome_process.terminate()
        os.system("pkill chrome")
    
    # Exception handling
    except TimeoutException:
        try:
            driver.quit()
            chrome_process.terminate()
            print("Page took too long to respond. Skipping...")
            if len(utilities.get_files_in_a_directory("server/output/" + site)) < 2:
                shutil.rmtree("server/output/" + site)
            os.system("pkill chrome")
        except Exception as e:
            print(e)
            pass
    except Exception as e:
        print(e)
        try:
            driver.quit()
            chrome_process.terminate()
            if len(utilities.get_files_in_a_directory("server/output/" + site)) < 2:
                shutil.rmtree("server/output/" + site)
            os.system("pkill chrome")
        except Exception as e:
            print(e)
            pass

output = Path.joinpath(Path.home().cwd(), "server/output", site)
if not os.path.exists(output):
    os.mkdir(output)
try:   
    visitWebsite(site, SLEEP, False)
    os.system("pkill chrome")
except Exception as e:
    os.system("pkill chrome")
    shutil.rmtree("server/output/" + site)
    print("error:", e)
