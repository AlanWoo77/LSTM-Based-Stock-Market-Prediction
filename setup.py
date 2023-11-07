import pandas as pd
import os

# Donwnload NASDAQ Historical Data
offset = 0
limit = False
period = "max"  # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max

data = pd.read_csv(
    "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt", sep="|"
)
data_clean = data[data["Test Issue"] == "N"]
symbols = data_clean["NASDAQ Symbol"].tolist()
print("total number of symbols traded = {}".format(len(symbols)))
os.mkdir("hist")
limit = limit if limit else len(symbols)
end = min(offset + limit, len(symbols))
is_valid = [False] * len(symbols)
# force silencing of verbose API
with open(os.devnull, "w") as devnull:
    with contextlib.redirect_stdout(devnull):
        for i in range(offset, end):
            s = symbols[i]
            try:
                data = yf.download(s, period=period)
            except:
                continue
            if len(data.index) == 0:
                continue

            is_valid[i] = True
            data.to_csv("hist/{}.csv".format(s))

print("Total number of valid symbols downloaded = {}".format(sum(is_valid)))

valid_data = data_clean[is_valid]
valid_data.to_csv("symbols_valid_meta.csv", index=False)

os.mkdir("stocks")
os.mkdir("etfs")

etfs = valid_data[valid_data["ETF"] == "Y"]["NASDAQ Symbol"].tolist()
stocks = valid_data[valid_data["ETF"] == "N"]["NASDAQ Symbol"].tolist()

import shutil
from os.path import isfile, join


def move_symbols(symbols, dest):
    for s in symbols:
        filename = "{}.csv".format(s)
        shutil.move(join("hist", filename), join(dest, filename))


move_symbols(etfs, "etfs")
move_symbols(stocks, "stocks")

os.rmdir("hist")
