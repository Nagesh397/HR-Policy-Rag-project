#In preprocessing we will make sure that all the unwanted data present in the datasource will be remvoed

import re

def preprocess(text:str) -> str:
   text = text.replace("\n"," " )
   text = re.sub(r"\s+", " ", text)
   text = text.strip()

   return text   

#eg -> replace()
# hello
# my 
# name 
# is
# Pranav

# hello my name is pranav

#eg for regular expression -> sub()
# "Hello     my.     name "
# Hello my name

# ".     hello my name     " -> strip()