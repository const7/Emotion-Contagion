import os
import jieba
import pandas as pd
from tqdm import tqdm

start_date = "2019-11-1"
media_data_path = "../data/media_data"
stop_words = "../data/stop_words.txt"

def get_data(file_path):
    total = pd.DataFrame()
    # read all xlsx files
    for file in tqdm(os.listdir(file_path)):
        xlsx = pd.ExcelFile(os.path.join(file_path, file))
        sheet_names = xlsx.sheet_names
        # read all sheets
        for sheet in sheet_names:
            df = pd.read_excel(xlsx, sheet)
            df.drop(columns=["微博id", "原始图片url", "微博视频url", "发布位置", "发布工具", "点赞数", "转发数", "评论数"], inplace=True)
            total = total.append(df)
    # rename columns
    total.columns = ["post", "time"]
    # set date as index
    total["time"] = pd.to_datetime(total["time"].dt.date)
    total.set_index("time", inplace=True)
    total.sort_index(inplace=True)
    # concatenate posts by date
    total = total.groupby("time")["post"].sum()
    return total
    # total.to_excel(os.path.join(file_path, "media_data.xlsx"), index=False)

def process_post(post):
    pass
    
if __name__ == "__main__":
    data = get_data(media_data_path)
    for date, post in tqdm(data.items()):
        print(date)
