from preprocesing import *


text_test = "Let's dance in style, Let's dance for a while,Heaven can wait, we're only watching the skies,Hoping for the best but expecting the worst,Are you gonna drop the bomb or not?"
print(text_test)
print("Removing stop words")
text_test=remove_stopw(text_test)
print(text_test)
print("Removing numbers")
text_test=remove_num(text_test)
print(text_test)
print("Removing punctuation")
text_test=remove_pun(text_test)
preper(text_test)