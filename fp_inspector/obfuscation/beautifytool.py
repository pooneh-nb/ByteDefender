import os
import sys
import time
import logging
from tqdm import tqdm
from tempfile import mkdtemp
from pathlib import Path
sys.path.insert(1, Path.home().cwd().as_posix())
proj_path = Path.home().cwd()
import utilities

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

log_path = Path(Path.cwd(), 'fp_inspector/obfuscation/beautifytool.log')
logging.basicConfig(filename= log_path, level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')


def setup_selenium_driver():
    binary_path = '/usr/bin/google-chrome-stable'
    user_data_directory = mkdtemp(prefix='chrome_user_data_', dir='chromedriver-linux64/user_data_dir')
    opt = Options()
    opt.binary_location = str(binary_path)
    opt.add_argument("--headless")
    opt.add_argument("--disable-gpu")
    opt.add_argument("--no-sandbox")
    opt.add_argument("--disable-dev-shm-usage")
    opt.add_argument(f"--user-data-dir={user_data_directory}")
    driver_path = Path(Path.cwd(), 'chromedriver-linux64/chromedriver')
    service = Service(str(driver_path), log_path='chromedriver.log')
    driver = webdriver.Chrome(service=service, options=opt)
    return driver

def run_beautify_tools(site, script, script_name, output_obfuscated, driver):  
    try:
        # print("getting driver")
        driver.get("http://beautifytools.com/javascript-obfuscator.php#")
        input_box = driver.find_element(By.ID,"src")
        driver.execute_script('document.getElementById("src").value=arguments[0]', script)
        button = driver.find_element(By.ID,"obfuscate")
        button.click()
        time.sleep(5) 
        output = driver.find_element(By.ID,"out")
        obfuscated_script = output.get_attribute('value')

        if not obfuscated_script:
            logging.error(f"Error running beautify tools: {site} : {script_name} : Obfuscation returned an empty result")
            return None        
        return obfuscated_script
    except Exception as e:
        print("Error")
        logging.error(f"Error running beautify tools: {site} : {script_name} : {e}")
    finally:
        driver.quit()
        os.system("pkill chrome")


def main():
    sites = utilities.get_directories_in_a_directory(Path(Path.home().cwd(), 'fp_inspector/data/sliced_db'))
    try:
        for site in tqdm(sites):
            print(site)
            # print(site)
            obfuscator_path = Path.home().joinpath(site, 'obfuscator')
            obfuscator_path.mkdir(exist_ok=True)
            
            beautifytools_path = Path.home().joinpath(obfuscator_path, 'beautifytools_path')
            beautifytools_path.mkdir(exist_ok=True)
            
            
            scripts = utilities.get_files_in_a_directory(Path(site, 'js'))
            
            for script_file in scripts:
                script_name = Path(script_file).name
                output_obfuscated = Path(beautifytools_path, script_name)
                if not os.path.exists(output_obfuscated):
                    # print("opening file")
                    print(script_file)
                    with open(script_file, 'r', encoding='utf-8') as file:
                        script = file.read()
                    print(script_name)
                    
                    driver = setup_selenium_driver()
                    
                    time.sleep(10)
                    obfuscated_script = run_beautify_tools(site, script, script_name, output_obfuscated, driver)
                    
                    if obfuscated_script: 
                        with open(output_obfuscated, 'w', encoding='utf-8') as file:
                            file.write(obfuscated_script)
                        print("Done")
                        logging.error(f"Done: {site} : {script_name}")   
                    else:
                        print("empty")
                        logging.error(f"Empty: {site} : {script_name}")   

    except Exception as e:
        print(e)
        logging.error(f"Error running beautify tools: {site} : {e}")    

if __name__ == "__main__":
    main()
        




