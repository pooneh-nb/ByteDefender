import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
import time
import shutil
from pyvirtualdisplay import Display
import os
import subprocess
import threading
import queue
from pathlib import Path
import utilities
import pandas as pd
import os

# Constants
PORT = 4000

def start_virtual_display() -> Display:
    """
    Start a virtual display for headless browser operation.

    Returns:
        Display: An instance of the virtual display.
    """
    display = Display(visible=0, size=(1000, 1000))
    display.start()
    return display

def initialize_chrome_browser(chrome_path: Path, extension_path: Path) -> subprocess.Popen:
    """"
    Initialze the Chrome browser subprocess

    Args:
        chrome_path (Path): Path to the Chrome binary
        extension_path (Path): Path to the Chrome extension

    Returns:
        subprocess.Popen : The subprocess object for the Chrome browser.
    """
    print("initialize_chrome_browser")
    chrome_command = f"{str(chrome_path)} --remote-debugging-port=9222 \
                                        --load-extension={str(extension_path)} \
                                        --disable-dev-shm-usage"
    chrome_process = subprocess.Popen(
        chrome_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        bufsize=1,  # Line buffered
        universal_newlines=True
    )
    return chrome_process

def enqueue_output(out: subprocess.io.BufferedReader, 
                   output_queue: queue.Queue)-> None:
    """
    Enqueue output from the subprocess into a queue.

    Args:
        output (subprocess._io.BufferReader): BufferedReader object from the subprocess.
        output_queue (queue.Queue):Queue to hold the output lines
    """
    for line in iter(out.readline, b''):
        output_queue.put(line)
    out.close()

def setup_output_handling_thread(chrome_process: subprocess.Popen) -> tuple[queue.Queue, threading.Thread]:
    """
    setup a thread for asynchronous output handling from the Chrome subprocess.

    Args:
    chrome_process (subprocess.Popen): The chrome subprocess

    Returns:
        tuple[queue.Queue, threading.Thread]: A tread containing the output queue and the thread object.
    """
    output_queue = queue.Queue()
    output_thread = threading.Thread(target=enqueue_output, args=(chrome_process.stdout, output_queue))
    output_thread.daemon = True
    output_thread.start()
    return output_queue, output_thread

def setup_selenium_driver() -> webdriver.Chrome:
    """
    Set up the Selenium WebDriver.
    
    Args:
        sleep (int): Implicit wait time for the WebDriver

    Returns:
        webdriver.Chrome: The Chrome WebDriver instance.
    """
    print("setup_selenium_driver")
    opt = Options()
    opt.add_experimental_option("debuggerAddress", "localhost:9222")
    opt.page_load_strategy = 'normal'
    driver_path = Path.home().cwd().parent / 'out' / 'Default' / 'chromedriver'
    service = Service(str(driver_path), log_path='chromedriver.log')
    driver = webdriver.Chrome(service=service, options=opt)
    return driver

def navigate_website(driver: webdriver.Chrome, 
                     site: str, 
                     sleep: int) -> None:
    """
    Navigate to a website using the provided WebDriver

    Args:
        driver (webdriver.Chrome): The Chrome WebDriver instance
        site (str): URL of the site to visit
        sleep (int):Duration to wait after loading the site
    """
    print("navigate_website")
    driver.delete_all_cookies()
    driver.set_page_load_timeout(30) # sets the time Selenium will wait for a page load to complete before throwing a TimeoutException
    driver.get(url=r"https://" + site)
    # driver.implicitly_wait(sleep) # sets the default wait time for Selenium to locate elements
    time.sleep(sleep)
    return
    

def process_and_save_logs(output_queue: queue.Queue, 
                          output: str):
    """
    Process and save logs from the output queue

    Args:
        outputqueuq (queue.Queue): Queue containing the iutput logs.
        output (str): Path to save the output file.
    """
    print("process_and_save_logs")
    logs_output = str(output) + "/browser_logs.bin"
    with open(logs_output, 'a') as log_file:
            # Process output from Chrome periodically
            while not output_queue.empty():
                line = output_queue.get_nowait()
                log_file.write(line) 

def handle_timeout_exception()->None:
    """
    Handle the TimeoutException.
    """
    print("Page took too long to respond. Skipping...")

def handle_general_exception(exception: Exception) -> None:
    """
    Handle general exceptions.

    Args:
    """
    print(f"Exception encountered: {exception}")

def cleanup_resources(driver: webdriver.Chrome, 
                      chrome_process: subprocess.Popen)-> None:
    """
    Clean up resources like WebDriver and subprocess.

    Args:
        driver (webdriver.Chrome): The Chrome WebDriver instance.
        chrome_process (subprocess.Popen): The Chrome subprocess.
    """
    print("cleanup_resources")
    if driver:
        driver.quit()
    if chrome_process:
        chrome_process.terminate()
    os.system("pkill chrome")

def check_and_remove_output(output: str)-> None:
    """
    Check and remove the output directory if it contains less than 2 files.

    Args:
        output (str): Path to the output directory.
    """
    print("check_and_remove_output")
    if len(utilities.get_files_in_a_directory(output)) < 2:
        shutil.rmtree(output)
        print("A single file... Remove")
    else:
        print("!!Complete!!")

def visit_website(site: str, sleep: int, output: Path) -> None:
    """
    Visit a website using Selenium and Chrome

    Args:
        site (str): URL of the website to visit
        sleep(int): Duration to wait during operation
        output(Path): Path to save output logs
    """
    chrome_process = None
    driver = None
    output_thread = None
    # start_virtual_display()
    try:
        print("================================")
        chrome_path = Path.cwd().parent / 'out' / 'Default' / 'chrome'
        print(chrome_path)
        extension_path = Path.cwd() / 'extension'
        chrome_process = initialize_chrome_browser(chrome_path, extension_path)
        
        # Setup output handling thread
        output_queue, output_thread = setup_output_handling_thread(chrome_process)
        
        # Selenium WebDriver setup and navigation
        driver = setup_selenium_driver()
        print("Rest to load...")
        time.sleep(2)
        
        # Send POST request to transmit data
        requests.post(
            url=f"http://localhost:{PORT}/complete",
            data={"website": site})
        
        navigate_website(driver, site, sleep)

        # Process and save logs
        process_and_save_logs(output_queue, output)
        ...
    except TimeoutException:
        handle_timeout_exception()
    except Exception as e:
        handle_general_exception(e)
    finally:
        print("Finally")
        cleanup_resources(driver, chrome_process)
        check_and_remove_output(output)

def main(site: str, output: str, sleep: int) -> None:
    """
    Main function to control the website visiting process.

    Args:
        site (str): URL of the site to visit.
        output (str): Path to save output logs.
        sleep (int): Duration to wait
    """
    if not os.path.exists(output):
        os.mkdir(output)
        try:   
            visit_website(site, sleep, output)
        except Exception as e:
            print(f"Error occurred: {e}")
            shutil.rmtree(output)

# Starting point of the script
if __name__ == "__main__":
    # chunks = utilities.get_files_in_a_directory(Path.home().cwd() / 'corrupted_bytecode_pipeline' / 'data' / 'corrupted_sites')
    # for chunk in chunks:
    #     df = list(utilities.read_json(chunk))
    #     SLEEP = 20
    #     for i, site_to_visit in enumerate(df):
    #         output_dir = Path.joinpath(Path.home().cwd(), "server/output", site_to_visit)
    #         print("Visiting: " + site_to_visit, "#: ", i)
    #         main(site_to_visit, output_dir, SLEEP)

    df = pd.read_csv(r"sites/tranco/chunk_1.csv")
    SLEEP = 15
   
    for i in df.index:
    #   if i > 1000:
        output = Path.joinpath(Path.home().cwd(), "server/output", df["website"][i])
        site_to_visit = df["website"][i]
        print("Visiting: ", site_to_visit)
        main(site_to_visit, output, SLEEP)

    df = pd.read_csv(r"sampled_sites/chunks/chunk_6.csv")
    SLEEP = 30

