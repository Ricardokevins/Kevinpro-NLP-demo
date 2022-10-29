import json
# f = open("/Users/sheshuaijie/Downloads/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json",'r',encoding='utf-8')

# data = json.load(f)
# for i in  data:
#     print(i)
#     print(len(data[i]))
# exit()
# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""XSum dataset."""


import json
import os

import datasets


_CITATION = """
@article{Narayan2018DontGM,
  title={Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization},
  author={Shashi Narayan and Shay B. Cohen and Mirella Lapata},
  journal={ArXiv},
  year={2018},
  volume={abs/1808.08745}
}
"""

_DESCRIPTION = """
Extreme Summarization (XSum) Dataset.
There are three features:
  - document: Input news article.
  - summary: One sentence summary of the article.
  - id: BBC ID of the article.
"""

# From https://github.com/EdinburghNLP/XSum/issues/12
_URL_DATA = "http://bollin.inf.ed.ac.uk/public/direct/XSUM-EMNLP18-Summary-Data-Original.tar.gz"
_URL_SPLITS = (
    "https://raw.githubusercontent.com/EdinburghNLP/XSum/master/XSum-Dataset/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json"
)

_DOCUMENT = "document"
_SUMMARY = "summary"
_ID = "id"

_REMOVE_LINES = set(
    [
        "Share this with\n",
        "Email\n",
        "Facebook\n",
        "Messenger\n",
        "Twitter\n",
        "Pinterest\n",
        "WhatsApp\n",
        "Linkedin\n",
        "LinkedIn\n",
        "Copy this link\n",
        "These are external links and will open in a new window\n",
    ]
)



def _generate_examples(split_path = "/Users/sheshuaijie/Downloads/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json", data_dir="/Users/sheshuaijie/Downloads/bbc-summary-data"):
    with open(split_path, "r", encoding="utf-8") as f:
        split_ids = json.load(f)
    split_names = ['train','test','validation']
    split_ids = {k: set(v) for k, v in split_ids.items()}
    for split_name in split_names:
        documents = []
        summarys = []

        if not split_ids[split_name]:
            break
        for ids in split_ids[split_name]:
            path = f"{data_dir}/{ids}.summary"
            f = open(path,'rb')
            text = "".join(
                [
                    line.decode("utf-8")
                    for line in f.readlines()
                    if line.decode("utf-8") not in _REMOVE_LINES and line.strip()
                ]
            )
            segs = text.split("[SN]")
            documents.append(segs[8].strip())
            summarys.append(segs[6].strip())
        f_d = open(f"/Users/sheshuaijie/Downloads/XSUM/{split_name}.document", 'w')
        for d in documents:
            f_d.write(d+'\n')
        f_s = open(f"/Users/sheshuaijie/Downloads/XSUM/{split_name}.summary", 'w')
        for s in summarys:
            f_s.write(s+'\n')

                # Each file follows below format:
                # [SN]URL[SN]
                # http://somelink
                #
                # [SN]TITLE[SN]
                # some intro
                #
                # [SN]FIRST-SENTENCE[SN]
                # some intro
                #
                # [SN]RESTBODY[SN]
                # text line.
                # another text line.
                # "another text line."

                # According to the following issue, FIRST-SENTENCE
                # is the reference summary and TITLE is unused:
                # https://github.com/EdinburghNLP/XSum/issues/22
                

_generate_examples()