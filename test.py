# # # import json
# # # import numpy as np
# # # from tensorflow import keras
# # # from sklearn.preprocessing import LabelEncoder
# # # import random
# # # import pickle
# # # from flask import Flask, request, jsonify, render_template

# # # app = Flask(__name__)

# # # # Load trained model
# # # model = keras.models.load_model('D:/devops/chat_model.h5')

# # # # Load tokenizer object
# # # with open('D:/devops/tokenizer.pickle', 'rb') as handle:
# # #     tokenizer = pickle.load(handle)

# # # # Load label encoder object
# # # with open('D:/devops/label_encoder.pickle', 'rb') as enc:
# # #     lbl_encoder = pickle.load(enc)

# # # # Load intents file
# # # with open("D:/devops/intents.json") as file:
# # #     data = json.load(file)

# # # # Parameters
# # # max_len = 20

# # # @app.route("/")
# # # def home():
# # #     return render_template("index.html")

# # # @app.route("/predict", methods=["POST"])
# # # def predict():
# # #     message = request.json['message']
# # #     result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([message]), truncating='post', maxlen=max_len))
# # #     tag = lbl_encoder.inverse_transform([np.argmax(result)])

# # #     for i in data['intents']:
# # #         if i['tag'] == tag:
# # #             response = np.random.choice(i['responses'])
# # #             break
# # #     else:
# # #         response = "I didn't understand that. Can you please rephrase?"

# # #     return jsonify({"response": response})

# # # if __name__ == "__main__":
# # #     app.run(debug=True)

# # # import fitz  # PyMuPDF
# # # import pandas as pd

# # # # Load your PDF file
# # # pdf_file = "azure1.pdf"
# # # doc = fitz.open(pdf_file)

# # # questions = []
# # # responses = []
# # # current_question = None
# # # current_response = []

# # # def is_question(line):
# # #     # Adjust this function to better detect questions
# # #     return line.endswith("?") or line.endswith(":")

# # # for page in doc:
# # #     text = page.get_text()

# # #     for line in text.splitlines():
# # #         line = line.strip()

# # #         if is_question(line):
# # #             if current_question:
# # #                 # Save the previous question and its response
# # #                 questions.append(current_question)
# # #                 responses.append(" ".join(current_response).strip())

# # #             current_question = line
# # #             current_response = []
# # #         elif line:  # If line is not empty, add it to the current response
# # #             current_response.append(line)

# # # # Add the last question and response
# # # if current_question:
# # #     questions.append(current_question)
# # #     responses.append(" ".join(current_response).strip())

# # # # Close the PDF
# # # doc.close()

# # # # Create a DataFrame
# # # df = pd.DataFrame({
# # #     "question": questions,
# # #     "response": responses
# # # })

# # # # Save the DataFrame as a CSV file
# # # df.to_csv("test13.csv", index=False)



# # import fitz  # PyMuPDF
# # import pandas as pd

# # # Load your PDF file
# # pdf_file = "azure1.pdf"
# # doc = fitz.open(pdf_file)

# # questions = []
# # responses = []
# # current_question = None
# # current_response = []

# # for page in doc:
# #     blocks = page.get_text("dict")["blocks"]  # Get text blocks with formatting info
    
# #     for block in blocks:
# #         for line in block["lines"]:
# #             line_text = ""
# #             is_bold = False

# #             for span in line["spans"]:
# #                 # Check if the text is bold
# #                 if "bold" in span["font"].lower():
# #                     is_bold = True
# #                     line_text += span["text"]
# #                 else:
# #                     line_text += span["text"]

# #             line_text = line_text.strip()
            
# #             if is_bold:
# #                 # If a new bold line is found, treat it as a new question
# #                 if current_question:
# #                     questions.append(current_question)
# #                     responses.append(" ".join(current_response).strip())

# #                 current_question = line_text
# #                 current_response = []
# #             elif line_text:  # If line is not empty, add it to the current response
# #                 current_response.append(line_text)

# # # Add the last question and response
# # if current_question:
# #     questions.append(current_question)
# #     responses.append(" ".join(current_response).strip())

# # # Close the PDF
# # doc.close()

# # # Create a DataFrame
# # df = pd.DataFrame({
# #     "question": questions,
# #     "response": responses
# # })

# # # Save the DataFrame as a CSV file
# # df.to_csv("test14.csv", index=False)
# import fitz  # PyMuPDF
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import LabelEncoder
# import pickle

# # Step 1: Extract questions and answers from the PDF
# pdf_file = "azure5.pdf"
# doc = fitz.open(pdf_file)

# questions = []
# responses = []
# current_question = None
# current_response = []

# for page in doc:
#     blocks = page.get_text("dict")["blocks"]  # Get text blocks with formatting info
    
#     for block in blocks:
#         for line in block["lines"]:
#             line_text = ""
#             is_bold = False

#             for span in line["spans"]:
#                 # Check if the text is bold
#                 if "bold" in span["font"].lower():
#                     is_bold = True
#                     line_text += span["text"]
#                 else:
#                     line_text += span["text"]

#             line_text = line_text.strip()
            
#             if is_bold and current_question:
#                 # If a new bold line is found, save the previous question and response
#                 questions.append(current_question)
#                 responses.append(" ".join(current_response).strip())
                
#                 # Start a new question
#                 current_question = line_text
#                 current_response = []
#             elif is_bold:
#                 # Start the first question
#                 current_question = line_text
#             elif line_text:  # If line is not empty, add it to the current response
#                 current_response.append(line_text)

# # Add the last question and response
# if current_question:
#     questions.append(current_question)
#     responses.append(" ".join(current_response).strip())

# # Close the PDF
# doc.close()

# # Step 2: Save the extracted data to a CSV file
# df = pd.DataFrame({
#     "question": questions,
#     "response": responses
# })

# csv_file_path = "test2.csv"
# df.to_csv(csv_file_path, index=False)


# import os
# secret_key = os.urandom(24)
# print(secret_key)




import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

# Step 1: Extract questions and answers from the PDF
pdf_file = "azure5.pdf"
doc = fitz.open(pdf_file)

questions = []
responses = []
current_question = None
current_response = []

for page in doc:
    blocks = page.get_text("dict")["blocks"]
    
    for block in blocks:
        for line in block["lines"]:
            line_text = ""
            is_bold = False

            for span in line["spans"]:
                if "bold" in span["font"].lower():
                    is_bold = True
                    line_text += span["text"]
                else:
                    line_text += span["text"]

            line_text = line_text.strip()
            
            if is_bold:
                if current_question:
                    questions.append(current_question)
                    responses.append(" ".join(current_response).strip())

                current_question = line_text
                current_response = []
            elif line_text:
                current_response.append(line_text)

if current_question:
    questions.append(current_question)
    responses.append(" ".join(current_response).strip())

doc.close()

# Step 2: Save the extracted data to a CSV file
df = pd.DataFrame({
    "question": questions,
    "response": responses
})

pdf_csv_file_path = "pdf_responses.csv"
df.to_csv(pdf_csv_file_path, index=False)

# Step 3: Load the custom responses CSV using df
custom_questions = ["Hi", "How are you?", "What is your name?"]
custom_responses = ["Hello!", "I'm good, thank you!", "I'm a chatbot!"]
question_types = ["greeting", "relationship", "identity"]

# Create a DataFrame for custom responses
custom_df = pd.DataFrame({
    "question": custom_questions,
    "response": custom_responses,
    "type": question_types
})

# Save the custom responses to CSV
custom_csv_file_path = "custom_responses.csv"
custom_df.to_csv(custom_csv_file_path, index=False)

# Step 4: Load the custom responses
custom_data = pd.read_csv(custom_csv_file_path)

# Step 5: Combine PDF and Custom Data
pdf_sentences = df['question'].tolist()
pdf_responses = df['response'].tolist()

custom_sentences = custom_data['question'].tolist()
custom_responses = custom_data['response'].tolist()
question_types = custom_data['type'].tolist()  # Assumes the custom CSV has a 'type' column

# Default response in case no match is found
default_response = "I'm sorry, I don't have an answer for that. Can you ask something else?"

# Combine PDF-based responses with custom responses (if needed)
training_sentences = pdf_sentences + custom_sentences
training_responses = pdf_responses + custom_responses

# Encoding the responses as labels
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_responses)
encoded_responses = lbl_encoder.transform(training_responses)

# Tokenization and Padding
vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# Step 6: Build and train the model (as before)
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(len(training_responses), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Train the model
epochs = 500
history = model.fit(padded_sequences, np.array(encoded_responses), epochs=epochs)

# Save the trained model, tokenizer, and label encoder (as before)
model.save("data1/chat_model.h5")

with open('data1/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data1/label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

# Step 7: Responding to user queries (Updated)
def respond_to_query(user_query):
    input_sentence = user_query.lower()

    # Check for custom responses (greetings, relationships, etc.)
    for idx, question in enumerate(custom_sentences):
        if question.lower() == input_sentence:
            return custom_responses[idx]

    # Check for PDF-based responses
    for idx, question in enumerate(pdf_sentences):
        if question.lower() == input_sentence:
            return pdf_responses[idx]

    # If no match, pass to the model for a predicted response
    seq = tokenizer.texts_to_sequences([input_sentence])
    padded = pad_sequences(seq, truncating='post', maxlen=max_len)
    pred = model.predict(padded)
    
    # Get the most likely response
    predicted_label = np.argmax(pred)
    response_text = lbl_encoder.inverse_transform([predicted_label])[0]

    # Confidence check for fallback
    if pred[0][predicted_label] < 0.5:  # Confidence threshold
        return default_response

    return response_text
