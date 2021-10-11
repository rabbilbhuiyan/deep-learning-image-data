# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts/preparation//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import os

path = os.chdir("C:/Users/Rabbil/Documents/New_folder_weapons_dataset/train/handgun")

i = 0
for file in os.listdir(path):
    new_file_name = "gun{}.png".format(i)
    os.rename(file, new_file_name)
    i = i + 1


