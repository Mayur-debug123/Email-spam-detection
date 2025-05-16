import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
data = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\spam\spam.csv")

data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham','spam'],['Not Spam','Spam'])
mess = data['Message']
catg = data['Category']
mess_train , mess_test , catg_train, catg_test = train_test_split(mess,catg,test_size=0.2)
cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)
model = MultinomialNB()
model.fit(features, catg_train)
features_test = mess_test = cv.transform(mess_test)
#print(model.score(features_test,catg_test))
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result

#output = predict('Congratulations , you won lottery')    
#print(output)

st.header('Spam Detection')
input_message = st.text_input('Enter Message here')
if st.button('Validate'):
    output = predict(input_message)[0]  
    st.write(f"Prediction: {output}")


"""
ham: "Life is more strict than teacher... Bcoz Teacher teaches lesson & then conducts exam, 
      But Life first conducts Exam & then teaches Lessons. Happy morning. . ."

spam: "Text & meet someone sexy today. U can find a date or even flirt its up to U. 
       Join 4 just 10p. REPLY with NAME & AGE eg Sam 25. 18 -msg recd@thirtyeight pence"

ham: "New Theory: Argument wins d SITUATION, but loses the PERSON. So dont argue with ur friends 
      just.. . . . kick them & say, I'm always correct.!"

spam: "Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry 2 100 wkly draw 
       txt MUSIC to 87066 TnCs www.Ldew.com1win150ppmx3age16"

ham: "Watching tv lor..."

spam: "FREE MESSAGE Activate your 500 FREE Text Messages by replying to this message with the word FREE 
       For terms & conditions, visit www.07781482378.com"
"""
