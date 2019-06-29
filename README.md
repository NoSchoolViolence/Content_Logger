# Content_Logger

The Content Logger is a Major Feature of NSV's Lantern App. This repo is collection of work that has been done on this feature.

- Trello: TBD
- The Lantern Wireframe: TBD

## NLP Library Load:

**Associated Files:**

_nlp_init_nsv_.py

**Associated Info:**

Stop having to start an NLP project by typing in every library you most commonly use at the start of every Jupyter Notebook. By placing this file in your `lib` folder then calling it in the first line of your Jupyter Notebook with `%run lib/__nlp_init_nsv__.py`, you will load all of the common libraries with one command. 

There are a number of common functions that will load also, the largest being `clean_the_text(raw_text)`. It is not the most elegant code, but it cleans text (especially text scrapped from the web or pulled in from a pdf) better than any other equal function of which I know. It will benefit you to look over the functions in the program in order to familiarize yourself with all of them in this subroutine.

When the routine first runs, it will tell you quite a bit about what is inside of the routine, but it doesn't tell you everything - at least not everything as of the current upload. This is still a work in progress. 

This whole thing, and the other file in the repository for stats library loading, come from my innate dislike of re-writing the same codes each time I need to run an analysis - whether in NLP or in Stats. It also comes from me wanting a library of functions whose names make more sense to me.

DISCLAIMER: Some of the code is mine; some of the code is not. Like many coders, I borrow from others along with writing my own code. If you see code in here that is yours, just tell me and I will attribute it to you. My laziness in keeping track of other's code has gotten me to the point where I don't remember all of the places I have borrowed code. If you are interested and want to know more about the code, please ask.

## Pulling & Processing PDFs:

**Associated Folders:**
  - test_pdfs
  - lib
  
**Associated Files:**
  - pattern.zip
  - Pulling and Processing PDFs .ipynb
  - KeywordExtract.csv
  - Requirements.txt

**Associated Info:**

This short program will walk through a directory pulling each file that ends in .pdf. It then cleans the text and returns each sentence that contains one of the keywords in which we are interested. 

This is part of a larger project in which I am scraping the web for all articles about school violence. I want to automate the research of thousands of articles to find the behaviors discussed and, more importantly, the correlations the researcher finds between behaviors and acts of violence. 

The problem this is solving is processing thousands of articles and academic research to find key elements to help in my own research.

The output of this program is a DataFrame (as a .csv file) that contains the article path, the keyword relative to the sentence, and the sentence in which that keyword is found. 

I have also included a copy of the requirements.txt in this folder. 

NOTE on the `__nlp_init_nsv__.py` file: It is a personal file that is still a work in progress. I hate re-writing code if it is not necessary. This file contains a myriad of functions that make it easier for me to process text. While it will give you a list of the functions when it first runs, that list is not complete. There are a good number of additional functions inside the file for which I have not added them to the print-out yet. 

It is not the most elegant code, or the most refined, but it works just fine for me. 

Note on pattern: You must include the decompressed version of this folder (directory) in the same directory as the notebook for Pattern to work. If you are not familiar with Pattern, this is just a quirk that it has. This is for Python 3.x. Please see the Pattern website for the Python 2.6 version of Pattern. 

## PubMed:

**Associated Folders:**
  - src
  - data

**Associated Info:**

This repository searches pubmed for key search terms related to school violence.
Search term and article ids, and article information are deposited into a sqlite database
that prevents redundancies and entry and makes it easy to query.

**Note:** Search term quality can potentially be improved by ensuring that
school is added to every search term so that it is not focused on general behaviors.

**TO DO:**

1. Implement a crontab to run code once a week to search and update database (via Azure?)
2. Develop model for data to identify key terms related to school violence.
3. Identify which articles cite a statistically significant intervention programs related to school violence. Alternatively generate meta data analysis to show share of studies which show a statistically significant effect.
4. Pull out all author names from the data.
