from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep

# Initialize browser driver
options = webdriver.EdgeOptions()
options.add_argument('--silent')
options.add_argument('--headless')
driver = webdriver.Edge(options=options)

wait = WebDriverWait(driver, 10)
driver.implicitly_wait(3)

# Handle captcha if needed
# captcha = input('Write captcha code:')
# driver.find_element(By.ID, '').send_keys(captcha)
# driver.find_element(By.ID, '').click()

# Load Excel data using pandas
file_path = r'File folder with search codes'
df = pd.read_excel(file_path, sheet_name='Base', usecols='M', skiprows=4)
df["A. M. Best Company"] = None
df["Standard & Poor's / FITCH"] = None
df["Moody's Investors Services"] = None

# Go to search page
# driver.get('')

# Loop though the numbers for research
for i, linha in df.iterrows():
    print(linha.iloc[0])
    if linha.iloc[0] == '-':
        continue
    code = int(linha.iloc[0])
    url_busca = f'https://www2.susep.gov.br/menuatendimento/info_resseguradoras_2011.asp?entcodigo={code}&codativo=True'
    driver.get(url_busca)
    # sleep(1)
    # wait.until(EC.presence_of_elements_located((By.CLASS_NAME, 'table-module')))
    # driver.implicitly_wait(5)

    # Extract the specific value after the query
    try:

        table = driver.find_element(By.XPATH, '/html/body/div[1]/div/table[5]/tbody')
        rows = table.find_elements(By.TAG_NAME, 'tr')

    except NoSuchElementException:
        result = 'None'
    else:
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, 'td')
            name = cols[0].text
            rating = cols[1].text

            df.at[i, name] = rating
            print(name, rating)
    # update the DataFrame
    # df.at[i, "Rating"] = result

# Save changes to the same Excel file
with pd.ExcelWriter(file_path, if_sheet_exists='overlay', mode='a') as writer:
    df['A. M. Best Company'].to_excel(writer, index=False, startrow=4, startcol=13, sheet_name='Base')
    df["Standard & Poor's / FITCH"].to_excel(writer, index=False, startrow=4, startcol=14, sheet_name='Base')
    df["Moody's Investors Services"].to_excel(writer, index=False, startrow=4, startcol=15, sheet_name='Base')

# Close the browser
driver.quit()
