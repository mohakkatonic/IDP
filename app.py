import os
import openai
import requests
from io import BytesIO
from pdf2image import convert_from_path
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import date

url_azure_vision = "https://katonic-vision-service.cognitiveservices.azure.com/computervision/imageanalysis:analyze?api-version=2023-10-01&features=read&model-version=latest&language=en"

# Set Azure OpenAI credentials from environment variables
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_type = 'azure'
openai.api_version = '2023-03-15-preview'
deployment_name = 'gpt-4-32k'

def extract_text_from_image(image_data):
    headers = {
        'Ocp-Apim-Subscription-Key': '2ea3da098229408e981b23ee77b55241',
        'Content-Type': 'image/jpeg'
    }
    response = requests.post(url_azure_vision, headers=headers, data=image_data.getvalue())

    if response.status_code == 200:
        result = response.json()
        read_result = result.get("readResult", {})
        extracted_text = ""

        for block in read_result.get("blocks", []):
            for line in block.get("lines", []):
                for word in line.get("words", []):
                    extracted_text += word.get("text", "") + " "

        return extracted_text.strip()

    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

def analyze_text(extracted_text):
    today_date = date.today()

    prompt = f"""Your input will be the text from a document and you have to produce an output strictly in a JSON format, with no other explanation. The JSON format will be explained below. You have to study the text.The text is: {extracted_text}. Classify the document based on the text. There are two classes to be classified in: Handwritten document and Utility/Electricity bill. Classify based on the following definitions:
    Utility/Electricity bill: An electricity/utility bill is a periodic statement detailing the charges for consumed electrical or utility services, typically indicating usage, rates, and total amount due. 
    Handwritten document: If the document is not a Utility/Electricity bill.
    Determine the type according to above and fill it in the JSON below. Dont justify the type determined or print the type separately. The determined type should be mentioned in the below JSON only.
    If the document is a Utility/Electricity bill, extract name, address,bill period and due date from it. Also check whether the bill is valid. You can check for its validity by below formula:
    Take out the difference between today's date :{today_date} and the extracted due date mentioned in the bill.
    Formula: {today_date}- extracted due date.
    If the above difference is less than or equal to one year, fill "Valid" : "Yes", id the difference is more than one year, fill "Valid" : "No".
    Strictly stick to the above formula for checking validity, dont take out the difference between the two dates of the bill period.
    Fill Valid = Yes in below JSON, if bill is valid, otherwise, fill Valid = No, if bill is not valid.
    Also produce a short summary of the bill. Fill the extracted details and generated summary in below JSON.
    In case of Utility/Electricity bill, the output should be strictly in the below JSON format:
    {{ "Type" : "Utility Bill", "Name" : "extracted name", "Address" : "Extracted address", "Bill Period" : "Extracted bill period", "Due Date" : "Extracted due date", "Current Date": "{today_date}", "Summary" : "Generated summary of utility bill", "Valid": "Yes or No"}}.Only give this JSON as output in case of utility bill, without any explanation.In the output, i should only see the Json like above. There should not be any explantion with it.
    If the document is a handwritten document, produce a short summary of it.Fill the generated summary in JSON below.
    In case of Handwritten text, the output should be strictly in the below JSON format:
    {{ "Type": "Handwritten", "Summary": "Generated summary of handwritten document"}}. Only give this JSON as output in case of handwritten document, without any explanation. In the output, i should only see the Json like above. There should not be any explantion with it."""
    
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=[
            {"role": "system", "content": "You are an expert who adeptly discerns and interprets information from various documents, demonstrating a keen ability to differentiate nuances and extract meaningful insights."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4000, temperature=0)
    return response['choices'][0]['message']['content']

# Set the page color theme using background color in markdown
st.title("Intelligent Document Processor")

# Create buttons for options
with st.sidebar:
        st.image('image/logo.png')
        selected = option_menu(
            menu_title = "Main Menu",
            options = ["About the App","IDP"])

# Define the application description
app_description = """
An intelligent document processor for handwritten documents and utility bills is a technology that uses advanced algorithms to extract, interpret, and manage information from handwritten content and utility invoices.
"""

# Handle button clicks
if selected == "About the App" :
    st.write(app_description)
else:
    uploaded_file = st.file_uploader("Upload a multipage PDF", type="pdf")

    if uploaded_file is not None:
        pdf_images = convert_from_path(uploaded_file.name)
        
        for page_number, pdf_image in enumerate(pdf_images):
            st.image(pdf_image, caption=f"Page {page_number + 1}", use_column_width=True)
            
            # Convert PDF image to JPEG
            output_buffer = BytesIO()
            pdf_image.save(output_buffer, format="JPEG")

            # Extract text from the image
            extracted_text = extract_text_from_image(output_buffer)

            if extracted_text:
                # Perform classification and extraction
                analysis_result = analyze_text(extracted_text)

                # Display the result
                st.json({"Analysis Result for Page {}".format(page_number + 1): analysis_result})

            else:
                st.json({"Analysis Result for Page {}".format(page_number + 1) : '{"Type" : "Invalid Document"}'})
