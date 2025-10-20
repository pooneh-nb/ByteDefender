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
from pathlib import Path
import utilities
import pandas as pd
import os

# Constants
PORT = 4000
PORT2 = 5000
def start_virtual_display() -> Display:
    """
    Start a virtual display for headless browser operation.

    Returns:
        Display: An instance of the virtual display.
    """
    display = Display(visible=0, size=(1000, 1000))
    display.start()
    return display

def setup_selenium_driver(extension_path: Path, binary_path: Path) -> webdriver.Chrome:
    """
    Set up the Selenium WebDriver.
    
    Args:
        extension_path (Path): Path to the Chrome extension
    Returns:
        webdriver.Chrome: The Chrome WebDriver instance.
    """
    print("setup_selenium_driver")
    opt = Options()
    opt.binary_location = str(binary_path)
    opt.page_load_strategy = 'normal'
    opt.add_argument("load-extension=" + str(extension_path))
    opt.add_argument("--no-sandbox")
    opt.add_argument("--disable-dev-shm-usage")

    driver_path = Path.home().cwd() / 'out' / 'Default' / 'chromedriver'
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
    driver.set_page_load_timeout(sleep+2) # sets the time Selenium will wait for a page load to complete before throwing a TimeoutException
    driver.get(url=r"https://" + site)
    time.sleep(sleep)
    return


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

def cleanup_resources(driver: webdriver.Chrome)-> None:
    """
    Clean up resources like WebDriver and subprocess.

    Args:
        driver (webdriver.Chrome): The Chrome WebDriver instance.
        chrome_process (subprocess.Popen): The Chrome subprocess.
    """
    print("cleanup_resources")
    if driver:
        driver.quit()
    os.system("pkill chrome")

def check_and_remove_output(output: str)-> None:
    """
    Check and remove the output directory if it contains less than 2 files.

    Args:
        output (str): Path to the output directory.
    """
    print("check_and_remove_output")
    if len(utilities.get_files_in_a_directory(output)) < 2:
        time.sleep(1)
        # shutil.rmtree(output)
        print("A single file... Remove")
    else:
        print("!!Complete!!")

def visit_website(site: str, sleep: int, output: Path, i: int) -> None:
    """
    Visit a website using Selenium and Chrome

    Args:
        site (str): URL of the website to visit
        sleep(int): Duration to wait during operation
        output(Path): Path to save output logs
    """
    driver = None
    # output_thread = None
    # start_virtual_display()
    with Display(visible=0, size=(1000, 1000)) as display:
        try:
            print("================================")
            print("Visiting: ", site, "#No: ", i)
            chrome_path = Path.cwd() / 'out' / 'Default' / 'chrome'
            extension_path = Path.cwd() / 'API_call_traces' / 'extension'
            
            # Selenium WebDriver setup and navigation
            driver = setup_selenium_driver(extension_path, chrome_path)
            print("Rest to load...")
            time.sleep(1)
            
            # Send POST request to transmit data
            requests.post(
                url=f"http://localhost:{PORT}/complete",
                data={"website": site})
            
            requests.post(
                url=f"http://localhost:{PORT2}/complete",
                data={"website": site})
            
            navigate_website(driver, site, sleep)

        except TimeoutException:
            handle_timeout_exception()
        except Exception as e:
            handle_general_exception(e)
        finally:
            print("Finally")
            # check_and_remove_output(output)
            cleanup_resources(driver)
        

def main(site: str, output: str, sleep: int, i: int) -> None:
    """
    Main function to control the website visiting process.

    Args:
        site (str): URL of the site to visit.
        output (str): Path to save output logs.
        sleep (int): Duration to wait
    """
    if not os.path.exists(output):
        os.mkdir(output)
        os.mkdir(output  / "bytecodes")
        try:   
            visit_website(site, sleep, output, i)
        except Exception as e:
            print(f"Error occurred: {e}")

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

    df = pd.read_csv(Path(Path.cwd(), "API_call_traces/tranco/chunk_0.csv"))
    SLEEP = 15
   
    for i in df.index:
        #if i > 3231:
            output = Path(Path.cwd(), "API_call_traces/server/output", df["website"][i].replace('/', '_'))
            browser_arg_path = str(Path.cwd() / 'sites.txt')
            with open(browser_arg_path, 'w') as file:
                file.write(str(output)+"/bytecodes/")
            site_to_visit = df["website"][i]
            main(site_to_visit, output, SLEEP, i)
